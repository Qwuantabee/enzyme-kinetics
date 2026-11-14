"""Tests for the congeneric-series structure-affinity analysis.

Two layers. The statistical machinery (ranking, Spearman, LOOCV,
permutation null) is tested against known answers and against SciPy, which
is an independent implementation. The descriptor layer is tested against
textbook constants for molecules small enough that the values are not in
doubt -- aspirin's 180.16 g/mol and single hydrogen-bond donor, benzene's one
aromatic ring.

The tests that need RDKit are marked so they skip cleanly if it is absent;
RDKit is an optional extra here, and the rest of the package must keep
importing without it.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from enzkin import qsar

ROOT = Path(__file__).resolve().parent.parent
SMILES_CSV = ROOT / "data" / "chembl_dhfr" / "ecoli_dhfr_smiles.csv"

needs_rdkit = pytest.mark.skipif(
    qsar.Chem is None, reason="RDKit is an optional extra (pip install rdkit)"
)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

@needs_rdkit
def test_descriptors_of_aspirin_match_textbook_values():
    d = qsar.compute_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert d["MW"] == pytest.approx(180.16, abs=0.01)
    assert d["HBD"] == 1.0
    assert d["HBA"] == 3.0     # Lipinski counts the two carbonyl O and the ester O
    assert d["Rings"] == 1.0
    assert d["AromaticRings"] == 1.0
    assert d["TPSA"] == pytest.approx(63.6, abs=0.5)
    assert d["RotB"] == 2.0    # the acetyl methyl counts; the ring bond does not


@needs_rdkit
def test_descriptors_of_benzene_and_ethanol_are_ordered_as_expected():
    benzene = qsar.compute_descriptors("c1ccccc1")
    ethanol = qsar.compute_descriptors("CCO")
    assert benzene["AromaticRings"] == 1.0
    assert benzene["RotB"] == 0.0
    assert benzene["HBD"] == 0.0
    assert ethanol["MW"] == pytest.approx(46.07, abs=0.01)
    assert ethanol["HBD"] == 1.0
    assert ethanol["RotB"] == 0.0
    assert benzene["logP"] > ethanol["logP"]
    assert set(benzene) == set(qsar.DESCRIPTOR_NAMES)


@needs_rdkit
def test_unparseable_smiles_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="could not parse"):
        qsar.compute_descriptors("this is not a molecule")


@needs_rdkit
def test_descriptor_table_names_the_compounds_it_drops():
    table, skipped = qsar.descriptor_table({
        "GOOD": "CCO",
        "BAD": "Q[Q",
        "EMPTY": "",
    })
    assert [r[0] for r in table] == ["GOOD"]
    assert skipped == ["BAD", "EMPTY"]


@needs_rdkit
def test_fraction_csp3_is_zero_for_a_fully_aromatic_molecule():
    assert qsar.compute_descriptors("c1ccccc1")["FractionCSP3"] == 0.0
    assert qsar.compute_descriptors("CCCC")["FractionCSP3"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Ranks and Spearman -- checked against SciPy
# --------------------------------------------------------------------------

def test_average_ranks_handles_ties():
    r = qsar._average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(r) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_matches_scipy_on_tied_data():
    rng = np.random.default_rng(3)
    for _ in range(25):
        x = rng.integers(0, 6, size=15).astype(float)
        y = rng.normal(size=15)
        rho, p = qsar.spearman(x, y)
        ref_rho, ref_p = stats.spearmanr(x, y)
        assert rho == pytest.approx(ref_rho, abs=1e-12)
        assert p == pytest.approx(ref_p, rel=1e-6, abs=1e-9)


def test_spearman_is_one_for_a_monotone_relation_and_minus_one_reversed():
    x = np.arange(1, 21, dtype=float)
    assert qsar.spearman(x, np.exp(x))[0] == pytest.approx(1.0)
    assert qsar.spearman(x, -x ** 3)[0] == pytest.approx(-1.0)


def test_spearman_needs_at_least_three_points():
    with pytest.raises(ValueError):
        qsar.spearman([1.0, 2.0], [1.0, 2.0])


def test_spearman_constant_input_is_undefined_not_significant():
    rho, p = qsar.spearman([1.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert rho == 0.0 and p == 1.0


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def test_r2_is_one_for_a_perfect_prediction_and_negative_when_worse_than_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert qsar.r2_score(y, y) == pytest.approx(1.0)
    assert qsar.r2_score(y, np.full(3, 100.0)) < 0


def test_loocv_recovers_a_clean_linear_relation():
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=40)
    x2 = rng.normal(size=40)
    X = np.column_stack([x1, x2])
    y = 3.0 * x1 - 2.0 * x2 + 0.5
    assert qsar.loocv_r2(X, y) == pytest.approx(1.0, abs=1e-10)


def test_loocv_standardises_with_training_statistics_only():
    # A single extreme row must not be able to help predict itself. If the
    # held-out row leaked into the standardisation, its prediction would be
    # pulled towards its own value and this residual would shrink.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [1000.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pred = qsar.loocv_predictions(X, y)
    assert pred.shape == (5,)
    # The four small rows dominate the training statistics of the last fold,
    # so the outlier lands well away from its own y.
    assert abs(pred[-1] - 4.0) > 0.0
    assert (pred[-1] - y[-1]) ** 2 > (pred[1] - y[1]) ** 2


def test_loocv_predictions_have_one_entry_per_compound():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    assert qsar.loocv_predictions(X, y).shape == (6,)


def test_standardise_leaves_a_constant_column_as_zeros_not_nan():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Z = qsar.standardise(X)
    assert np.all(Z[:, 0] == 0.0)
    assert np.all(np.isfinite(Z))


# --------------------------------------------------------------------------
# Permutation null -- the honesty test
# --------------------------------------------------------------------------

def test_permutation_null_finds_signal_in_strong_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    y = 5.0 * X[:, 0] + 0.1 * rng.normal(size=60)
    r2 = qsar.loocv_r2(X, y)
    res = qsar.permutation_null(X, y, r2, n_perm=200, seed=1)
    assert r2 > 0.8
    assert res["p_perm"] < 0.02


def test_permutation_null_does_not_find_signal_in_pure_noise():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(21, 10))   # the real setting: 21 rows, 10 columns
    y = rng.normal(size=21)
    r2 = qsar.loocv_r2(X, y)
    res = qsar.permutation_null(X, y, r2, n_perm=300, seed=2)
    # Noise is routinely scored as positively as the real fit; that is the
    # whole reason this function exists. And on 10 columns of 21 rows the
    # null's mean R2 is deeply negative, which is what an 11-parameter fit
    # on 21 noise observations does when each fold refits from scratch.
    assert res["p_perm"] > 0.05
    assert res["null_mean"] < 0.0
    assert res["null_p95"] < 0.6


def test_permutation_p_is_a_proper_p_value():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 3))
    y = rng.normal(size=15)
    r2 = qsar.loocv_r2(X, y)
    res = qsar.permutation_null(X, y, r2, n_perm=99, seed=0)
    assert 0.0 < res["p_perm"] <= 1.0
    # p must equal the fraction of the null at or above the observed score.
    expected = (1 + int((res["null_r2"] >= r2).sum())) / 100.0
    assert res["p_perm"] == pytest.approx(expected)


def test_permutation_null_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    r2 = qsar.loocv_r2(X, y)
    a = qsar.permutation_null(X, y, r2, n_perm=50, seed=42)
    b = qsar.permutation_null(X, y, r2, n_perm=50, seed=42)
    c = qsar.permutation_null(X, y, r2, n_perm=50, seed=43)
    assert a["p_perm"] == b["p_perm"]
    assert np.array_equal(a["null_r2"], b["null_r2"])
    assert not np.array_equal(a["null_r2"], c["null_r2"])


def test_permutation_null_rejects_a_nonsense_permutation_count():
    X = np.arange(9, dtype=float).reshape(3, 3)
    with pytest.raises(ValueError):
        qsar.permutation_null(X, np.arange(3, dtype=float), 0.5, n_perm=0)


def test_permutation_p_counts_only_exceedances():
    # An observed R2 of 2.0 is beyond anything a shuffle will produce here,
    # so p must sit at its floor, 1/(n_perm+1).
    rng = np.random.default_rng(17)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    res = qsar.permutation_null(X, y, 2.0, n_perm=20, seed=0)
    assert res["p_perm"] == pytest.approx(1.0 / 21.0)


# --------------------------------------------------------------------------
# Collinearity and univariate scan shape
# --------------------------------------------------------------------------

def test_collinearity_reports_the_strongest_pair():
    X = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 3.0], [3.0, 3.0, 1.0],
                  [4.0, 4.0, 9.0]])
    best = qsar.collinearity(X, ["a", "b", "c"])
    assert best["pair"] == ("a", "b")
    assert best["abs_r"] == pytest.approx(1.0)


def test_univariate_scan_is_sorted_by_absolute_rho():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(30, 3))
    y = -4.0 * X[:, 1] + rng.normal(size=30) * 0.01
    scan = qsar.univariate_scan(X, ["a", "b", "c"], y)
    assert scan[0]["descriptor"] == "b"
    assert scan[0]["rho"] < 0
    assert scan[0]["loocv_r2_single"] > 0.9
    mags = [abs(d["rho"]) for d in scan]
    assert mags == sorted(mags, reverse=True)


def test_parsimony_says_one_column_was_enough():
    # y is a copy of column 0; the other nine columns are noise. A parsimony
    # check must credit column 0 and report no gain from the full model.
    rng = np.random.default_rng(29)
    X = rng.normal(size=(40, 10))
    y = X[:, 0] + 0.01 * rng.normal(size=40)
    names = [f"d{i}" for i in range(10)]
    par = qsar.parsimony(X, names, y)
    assert par["best_descriptor"] == "d0"
    assert par["best_single_r2"] > 0.95
    assert par["improvement"] < 0.05


# --------------------------------------------------------------------------
# Truncation sensitivity
# --------------------------------------------------------------------------

def test_truncation_drops_the_weak_end_and_refits():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(30, 2))
    y = 2.0 * X[:, 0] + 0.05 * rng.normal(size=30)
    ki = np.concatenate([np.full(25, 10.0), np.array([2e5, 3e5, 4e5, 5e5,
                                                      6e5])])
    res = qsar.truncation_sensitivity(X, y, ki, cutoff_nM=1e5, n_perm=40,
                                      seed=0)
    assert res["applicable"] is True
    assert res["n_dropped"] == 5
    assert res["n_kept"] == 25
    assert res["loocv_r2"] > 0.5
    assert res["p_perm"] <= 0.05


def test_truncation_declines_when_it_would_leave_too_few_rows():
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.arange(8, dtype=float)
    ki = np.full(8, 1e6)
    res = qsar.truncation_sensitivity(X, y, ki, cutoff_nM=1e5)
    assert res["applicable"] is False
    assert res["n_kept"] == 0


def test_truncation_declines_when_nothing_exceeds_the_cutoff():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    res = qsar.truncation_sensitivity(X, y, np.full(20, 5.0), cutoff_nM=1e5)
    assert res["applicable"] is False
    assert res["n_dropped"] == 0


# --------------------------------------------------------------------------
# The series itself -- the committed data, end to end
# --------------------------------------------------------------------------

@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_the_21_compound_series_is_recoverable_from_committed_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qsar_congeneric", ROOT / "scripts" / "qsar_congeneric.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.series_rows("CHEMBL1123078")
    assert len(s["rows"]) == 21
    assert s["ki_nM"].min() == pytest.approx(0.021)
    assert s["ki_nM"].max() == pytest.approx(16.0)
    assert s["ki_nM"].max() / s["ki_nM"].min() > 700
    assert not s["skipped_unparseable"]


@pytest.mark.skipif(not SMILES_CSV.exists(),
                    reason="run scripts/fetch_chembl_smiles.py first")
@needs_rdkit
def test_verdict_language_tracks_the_permutation_p():
    def fake(r2, p, n=21, gain=0.5):
        return {"loocv_r2": r2, "n": n,
                "null": {"p_perm": p, "n_perm": 1000},
                "parsimony": {"best_descriptor": "MW",
                              "best_single_r2": 0.4,
                              "improvement": gain}}
    assert "no support" in qsar_script_verdict(fake(-0.2, 0.001))
    assert "consistent with" in qsar_script_verdict(fake(0.6, 0.01))
    assert "too small" in qsar_script_verdict(fake(0.6, 0.4))
    # A significant fit that one column already explains must say so.
    assert "adds nothing over one column" in qsar_script_verdict(
        fake(0.6, 0.01, gain=0.0))


def qsar_script_verdict(result):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qsar_congeneric_verdict", ROOT / "scripts" / "qsar_congeneric.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verdict(result)


# --------------------------------------------------------------------------
# Direction of the effect -- the README got this backwards once
# --------------------------------------------------------------------------

QSAR_REPORT = ROOT / "results_qsar.txt"


def _univariate_rho(report_text, descriptor, series_index):
    """Pull one descriptor's rho out of a series block in results_qsar.txt.

    series_index is a document_chembl_id (preferred: the block order changes
    whenever a series is added or a partial run rewrites the file) or an
    integer block position.
    """
    blocks = report_text.split("--- ")[1:]
    if isinstance(series_index, str):
        matches = [b for b in blocks if b.startswith(series_index)]
        assert matches, f"{series_index} has no block in results_qsar.txt"
        block = matches[0]
    else:
        block = blocks[series_index]
    for line in block.splitlines():
        parts = line.split()
        if parts and parts[0] == descriptor and len(parts) >= 4:
            return float(parts[1])
    raise AssertionError(f"{descriptor} not listed in series block {series_index}")


@pytest.mark.skipif(not QSAR_REPORT.exists(), reason="run scripts/qsar_congeneric.py")
def test_molecular_weight_rho_is_negative_against_log10_ki():
    """Lower Ki is tighter binding, so rho(MW, log10 Ki) < 0 means heavier binds tighter.

    The README once described this as heavier compounds binding *worse*, which
    is the opposite of what the artefact says. Pin the sign so a sign flip in
    the prose, or in the analysis, cannot pass unnoticed.
    """
    text = QSAR_REPORT.read_text()
    for series_index in ("CHEMBL1129627", "CHEMBL1124156"):
        rho = _univariate_rho(text, "MW", series_index)
        assert rho < 0, f"rho(MW) = {rho} in series block {series_index}"


@pytest.mark.skipif(not QSAR_REPORT.exists(), reason="run scripts/qsar_congeneric.py")
def test_the_most_negative_descriptors_are_not_uniform_and_hbd_is_the_exception():
    """The 'every case' claim in the README was false: HBD is positive."""
    text = QSAR_REPORT.read_text()
    rho_mw = _univariate_rho(text, "MW", "CHEMBL1129627")
    rho_hbd = _univariate_rho(text, "HBD", "CHEMBL1129627")
    assert rho_mw < 0 < rho_hbd


def test_negative_rho_against_log10_ki_means_tighter_binding():
    """Arithmetic guard on the interpretation itself, independent of the data."""
    ki = np.array([100.0, 10.0, 1.0])
    mw = np.array([200.0, 300.0, 400.0])   # heavier molecule, tighter binding
    rho, _p = stats.spearmanr(mw, np.log10(ki))
    assert rho == -1.0
