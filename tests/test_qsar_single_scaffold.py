"""Tests for the single-scaffold (chemotype-controlled) QSAR test.

The confound this guards against is specific: a multi-scaffold QSAR can look
predictive purely because chemotypes differ in size, so molecular weight acts
as a scaffold label. These tests pin the three things that would let the
analysis lie about it -- the scaffold assignment, the subset threshold, and
the wording that turns a within-scaffold R2 into a conclusion.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from enzkin import qsar

ROOT = Path(__file__).resolve().parent.parent
SMILES_CSV = ROOT / "data" / "chembl_dhfr" / "ecoli_dhfr_smiles.csv"

needs_rdkit = pytest.mark.skipif(
    qsar.Chem is None, reason="RDKit is an optional extra (pip install rdkit)"
)


def _load_single_scaffold():
    """Import scripts/qsar_single_scaffold.py by path (scripts/ is not a pkg)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "qsar_single_scaffold", ROOT / "scripts" / "qsar_single_scaffold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Scaffold assignment and subsetting -- pure logic, no chemistry needed
# --------------------------------------------------------------------------

def test_largest_scaffold_picks_the_most_common_member():
    mod = _load_single_scaffold()
    toks = ["a"] * 5 + ["b"] * 2 + ["c"]
    assert mod.largest_scaffold(toks) == ("a", 5)


def test_largest_scaffold_ignores_unparseable_compounds():
    # None means RDKit could not read the SMILES; it must not be the biggest
    # "scaffold" just because it is frequent.
    mod = _load_single_scaffold()
    assert mod.largest_scaffold([None, None, None, "a", "a"]) == ("a", 2)


def test_largest_scaffold_of_an_empty_series_is_empty():
    mod = _load_single_scaffold()
    assert mod.largest_scaffold([None, None]) == (None, 0)


def test_subset_keeps_row_order_and_the_matching_affinities():
    mod = _load_single_scaffold()
    rows = [("c1", "S", {}), ("c2", "S", {}), ("c3", "S", {})]
    ki = np.array([10.0, 20.0, 30.0])
    sc = ["a", "b", "a"]
    sub_rows, sub_ki = mod.subset(rows, ki, sc, "a")
    assert [r[0] for r in sub_rows] == ["c1", "c3"]
    assert list(sub_ki) == [10.0, 30.0]


def test_subset_of_a_scaffold_absent_from_the_series_is_empty():
    mod = _load_single_scaffold()
    rows = [("c1", "S", {})]
    sub_rows, sub_ki = mod.subset(rows, np.array([1.0]), ["a"], "zzz")
    assert sub_rows == []
    assert sub_ki.size == 0


# --------------------------------------------------------------------------
# The n < 20 refusal -- the whole point of the threshold
# --------------------------------------------------------------------------

def test_subset_below_the_threshold_is_refused_not_fitted():
    """19 compounds must return too_small and no R2 at all.

    If this ever returns a loocv_r2, the script has started printing numbers
    that chance reaches routinely on ten descriptors.
    """
    mod = _load_single_scaffold()
    assert mod.MIN_SUBSET == 20
    rows = [("c%d" % i, "S", {}) for i in range(mod.MIN_SUBSET - 1)]
    res = mod.analyse_subset(rows, np.arange(1.0, 20.0))
    assert res["too_small"] is True
    assert res["n"] == mod.MIN_SUBSET - 1
    assert "loocv_r2" not in res


def test_the_threshold_is_checked_on_the_subset_not_the_series():
    """A 73-compound series with a 3-compound scaffold must refuse too."""
    mod = _load_single_scaffold()
    assert mod.analyse_subset([("c", "S", {})] * 3, np.ones(3))["too_small"]


def test_refusal_verdict_names_the_subset_as_too_small():
    mod = _load_single_scaffold()
    text = mod.verdict({"too_small": True, "n": 12, "min_subset": 20}, "X")
    assert "too small to test" in text
    assert "n = 12" in text
    assert "R2" not in text.replace("no R2 is reported", "")


# --------------------------------------------------------------------------
# Verdict wording must match the uncertainty
# --------------------------------------------------------------------------

def _fake(r2, p, n=66, gain=0.2, best="MW"):
    return {"too_small": False, "loocv_r2": r2, "n": n,
            "null": {"p_perm": p, "n_perm": 1000},
            "parsimony": {"best_descriptor": best, "best_single_r2": 0.3,
                          "improvement": gain}}


def test_a_real_within_scaffold_fit_is_called_a_survival():
    mod = _load_single_scaffold()
    text = mod.verdict(_fake(0.6, 0.001), "CHEMBL1")
    assert "survives" in text
    assert "chemotype spread" not in text


def test_a_negative_within_scaffold_r2_is_called_chemotype_spread():
    mod = _load_single_scaffold()
    text = mod.verdict(_fake(-0.3, 0.9), "CHEMBL1")
    assert "does NOT survive" in text
    assert "chemotype spread" in text


def test_a_positive_r2_that_chance_reaches_is_not_called_a_finding():
    """R2 above zero with p = 0.4 must say 'too small to distinguish'.

    This is the failure mode the whole script exists for: a within-scaffold
    number that looks like a result and is not one.
    """
    mod = _load_single_scaffold()
    text = mod.verdict(_fake(0.25, 0.40), "CHEMBL1")
    assert "too small to distinguish" in text
    assert "does NOT survive" not in text
    assert "survives" not in text.replace("cannot be credited", "")


def test_verdict_never_uses_proof_language():
    mod = _load_single_scaffold()
    for args in ((0.6, 0.001), (0.25, 0.4), (-0.3, 0.9)):
        text = mod.verdict(_fake(*args), "CHEMBL1")
        for word in ("proves", "confirms", "established"):
            assert word not in text


def test_a_one_descriptor_explained_fit_is_flagged_as_a_size_effect():
    mod = _load_single_scaffold()
    text = mod.verdict(_fake(0.65, 0.001, gain=0.009), "CHEMBL1")
    assert "size effect" in text


# --------------------------------------------------------------------------
# The real data, end to end
# --------------------------------------------------------------------------

@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_both_large_series_have_a_scaffold_large_enough_to_test():
    """The subsets must clear MIN_SUBSET, or the report is a refusal.

    Sizes are pinned so a change in the SMILES file cannot quietly move which
    compounds the chemotype test covers.
    """
    mod = _load_single_scaffold()
    records, _report = mod.run(series=("CHEMBL1129627", "CHEMBL1124156"),
                               n_perm=20)
    by_doc = {r["series"]["document_id"]: r for r in records}
    assert by_doc["CHEMBL1129627"]["series"]["n_total"] == 73
    assert by_doc["CHEMBL1129627"]["series"]["largest_size"] == 66
    assert by_doc["CHEMBL1124156"]["series"]["n_total"] == 72
    assert by_doc["CHEMBL1124156"]["series"]["largest_size"] == 52
    for doc, rec in by_doc.items():
        assert rec["subset"]["too_small"] is False
        assert rec["subset"]["n"] == rec["series"]["largest_size"]
        assert rec["subset"]["n"] >= mod.MIN_SUBSET


@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_scaffold_identity_dominates_the_ring_count_descriptors():
    """Rings and AromaticRings are pure chemotype labels in these series.

    eta-squared of 1.0 means all of their variance is between scaffolds, so a
    model that "predicts" from them is reading the chemotype and nothing else.
    That is the confound stated as a measurement.
    """
    mod = _load_single_scaffold()
    records, _report = mod.run(series=("CHEMBL1129627",), n_perm=20)
    spread = {d["descriptor"]: d["eta_squared"]
              for d in records[0]["spread"]}
    assert spread["Rings"] == pytest.approx(1.0)
    assert spread["AromaticRings"] == pytest.approx(1.0)
    # ...while the descriptors that do the predicting are mostly within-form.
    assert spread["MW"] < 0.5


@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_within_scaffold_analysis_reproduces_from_committed_files():
    """The published within-scaffold numbers, recomputed here, seed and all."""
    mod = _load_single_scaffold()
    records, _report = mod.run(series=("CHEMBL1129627", "CHEMBL1124156"),
                               n_perm=200)
    by_doc = {r["series"]["document_id"]: r for r in records}
    a = by_doc["CHEMBL1129627"]["subset"]
    b = by_doc["CHEMBL1124156"]["subset"]
    assert a["n"] == 66 and b["n"] == 52
    assert a["loocv_r2"] == pytest.approx(0.312, abs=0.01)
    assert b["loocv_r2"] == pytest.approx(0.653, abs=0.01)
    # Both survive the null, and the p is at the permutation floor.
    assert a["null"]["p_perm"] <= 1.0 / (200 + 1)
    assert b["null"]["p_perm"] <= 1.0 / (200 + 1)


@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_the_committed_report_matches_a_fresh_run():
    """results_single_scaffold.txt must be the output, not a transcription."""
    mod = _load_single_scaffold()
    path = ROOT / "results_single_scaffold.txt"
    if not path.exists():
        pytest.skip("figure first: python scripts/qsar_single_scaffold.py")
    committed = path.read_text().strip()
    _records, fresh = mod.run(n_perm=1000)
    assert committed == fresh.strip()
