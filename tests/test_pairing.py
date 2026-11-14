"""Tests for the IC50/Ki pairing module and the Cheng-Prusoff check.

These are different from the algebra tests in test_cheng_prusoff.py. There
the question was whether the formula computes what it claims. Here the
question is whether the formula's *prediction about real data* holds, and
whether the code that measures it is honest about what it does and does not
establish.

Three layers:

1. Unit tests on the pieces, with numbers small enough to check by hand.
2. A synthetic case where the truth is known: build IC50 = Ki (1 + [S]/Km)
   for two papers at different [S]/Km and confirm the statistics recover the
   structure Cheng-Prusoff predicts, and confirm they *fail* when the ratio
   instead tracks potency.
3. The real ChEMBL data, with the values this repository's files actually
   produce pinned. If a loader or a join changes, these fail -- which is the
   point. The pinned numbers are copied from
   `python scripts/validate_cheng_prusoff.py` output, not invented.
"""

import math

import numpy as np
import pytest

from enzkin import datasets, pairing


# --- 1. pieces ---------------------------------------------------------------

def test_geometric_mean_is_not_arithmetic_mean():
    """10 and 1000 average to 505 arithmetically and 100 geometrically. IC50
    and Ki are log-scale, so the geometric value is the right one, and the
    test pins the difference rather than only the value."""
    assert pairing.geometric_mean([10, 1000]) == pytest.approx(100.0)
    assert pairing.geometric_mean([100, 100, 100]) == pytest.approx(100.0)


def test_geometric_mean_refuses_non_positive():
    """A zero concentration is a bad record. Dropping it would change a
    statistic silently, so it must raise."""
    with pytest.raises(ValueError):
        pairing.geometric_mean([1.0, 0.0])
    with pytest.raises(ValueError):
        pairing.geometric_mean([1.0, -5.0])
    with pytest.raises(ValueError):
        pairing.geometric_mean([])


def test_implied_substrate_ratio_inverts_the_correction():
    """The inversion must undo the forward formula exactly."""
    for s_over_km in (0.0, 0.5, 1.0, 6.0, 19.0):
        ratio = 1.0 + s_over_km
        assert pairing.implied_substrate_ratio(ratio) == \
            pytest.approx(s_over_km, abs=1e-12)


def test_physical_ratio_flag_is_a_floor_at_one():
    """Competitive Cheng-Prusoff cannot give a ratio below 1 because [S]/Km
    cannot be negative. The flag must not clamp the value, only report it."""
    assert pairing.is_physical_ratio(1.0)
    assert pairing.is_physical_ratio(5.0)
    assert not pairing.is_physical_ratio(0.99)
    # and the arithmetic is still returned, so the caller sees the size of
    # the violation rather than a zero
    assert pairing.implied_substrate_ratio(0.7) == pytest.approx(-0.3)


def test_assay_class_flags_antibacterial_descriptions_without_asserting_a_fact():
    """One ChEMBL document DESCRIBES its 'Ki' as whole-cell antibacterial
    activity. The flag is a keyword match on that description and is
    unverified (the values look like enzyme Ki), so its name must not claim
    the record is not an enzyme measurement."""
    assert pairing.assay_class(
        "Antibacterial activity against Escherichia coli"
    ) == "antibacterial_described"
    assert pairing.assay_class("Antibacterial activity against Escherichia coli") \
        != "non_enzyme"
    assert pairing.assay_class(
        "Inhibition constant against Dihydrofolate reductase") == "enzyme"
    # a generic or empty description stays enzyme: dropping unknowns would
    # hide the problem instead of exposing it
    assert pairing.assay_class("") == "enzyme"
    assert pairing.assay_class(None) == "enzyme"


def test_predicted_ratio_matches_the_correction_factor():
    """predicted_ratio must be the same number as cheng_prusoff's factor,
    stated as a prediction about a measurable ratio."""
    from enzkin import cheng_prusoff as cp
    km = 2.5e-6
    for s in (0.0, 1e-6, 2.5e-6, 5e-5):
        assert pairing.predicted_ratio(s, km) == pytest.approx(
            cp.correction_factor(max(s, 1e-12), km, "competitive")) \
            if s > 0 else pairing.predicted_ratio(s, km) == pytest.approx(1.0)


def test_predicted_ratio_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pairing.predicted_ratio(-1.0, 1.0)
    with pytest.raises(ValueError):
        pairing.predicted_ratio(1.0, 0.0)
    with pytest.raises(ValueError):
        pairing.predicted_ratio(float("nan"), 1.0)


# --- 2. a synthetic world where the truth is known ---------------------------

def _synthetic_pairs(docs, n_per_doc, ratio_fn, ki_spread=2.0, seed=0):
    """Build IC50/Ki pairs from a known ratio function.

    `ratio_fn(ki)` returns the ratio for a compound with that Ki, so a
    per-assay constant is `lambda ki: 13.0` and a per-compound correction is
    something that varies with ki. Seeds are fixed and Ki values are drawn
    log-uniformly over four orders of magnitude, which is the range the real
    data spans.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for doc, ratio in docs.items():
        kis = 10 ** rng.uniform(-1, 3, size=n_per_doc)
        for i, ki in enumerate(kis):
            r = ratio if not callable(ratio) else ratio(ki)
            rows.append({
                "molecule_chembl_id": f"M{doc}_{i}",
                "document_chembl_id": doc,
                "ic50_nM": ki * r,
                "ki_nM": ki * math.exp(rng.normal(0, ki_spread / 10)),
                "assay_description": "Inhibition of dihydrofolate reductase",
            })
    return rows


def test_synthetic_per_assay_constant_is_recovered():
    """Two papers, one ratio each, noiseless. The analysis must report the
    within-paper spread as zero and attribute essentially all variance to
    the paper -- the signature Cheng-Prusoff predicts."""
    ratio = 13.0
    rows = [
        {"molecule_chembl_id": f"A{i}", "document_chembl_id": "DOCA",
         "ic50_nM": (10 ** i) * ratio, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(6)
    ] + [
        {"molecule_chembl_id": f"B{i}", "document_chembl_id": "DOCB",
         "ic50_nM": (10 ** i) * ratio, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(6)
    ]
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    assert len(pairs) == 12
    s = pairing.summarize_ratios([p["ratio"] for p in pairs])
    assert s["median"] == pytest.approx(ratio)
    assert s["log10_sd"] == pytest.approx(0.0, abs=1e-12)
    assert pairing.implied_substrate_ratio(s["median"]) == pytest.approx(12.0)

    label, reasons = pairing.verdict(pairs)
    assert label == "partially supported", reasons
    assert any("no assay [S]" in r for r in reasons)


def test_synthetic_different_assay_factors_stay_separated():
    """Two papers at 3x and 21x: the per-paper medians must come out as
    those numbers, and the between-paper variance must dominate."""
    rows = [
        {"molecule_chembl_id": f"A{i}", "document_chembl_id": "DOCA",
         "ic50_nM": (10 ** i) * 3.0, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(5)
    ] + [
        {"molecule_chembl_id": f"B{i}", "document_chembl_id": "DOCB",
         "ic50_nM": (10 ** i) * 21.0, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(5)
    ]
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    spread = pairing.within_document_spread(pairs)
    assert spread["per_document"]["DOCA"]["median_ratio"] == pytest.approx(3.0)
    assert spread["per_document"]["DOCB"]["median_ratio"] == pytest.approx(21.0)
    assert spread["pooled_log10_sd"] == pytest.approx(0.0, abs=1e-12)

    de = pairing.document_effect(pairs)
    assert de["document_explains"] == pytest.approx(1.0, abs=1e-9)


def test_synthetic_ratio_tracking_potency_shows_a_slope():
    """The falsification branch. If the ratio depended on the compound's
    potency, the within-paper slope would be non-zero. This is the test that
    would catch a code path silently fitting a per-compound correction and
    calling it an assay constant."""
    rows = []
    for i in range(10):
        ki = 10 ** (-1 + 0.4 * i)  # spans four orders of magnitude
        # ratio grows 1.3x per decade of potency
        r = 2.0 * (ki ** 0.3)
        rows.append({"molecule_chembl_id": f"X{i}", "document_chembl_id": "DOCX",
                     "ic50_nM": ki * r, "ki_nM": ki,
                     "assay_description": "Inhibition of DHFR"})
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    slopes = pairing.ratio_vs_potency_slope(pairs, min_pairs=5)
    assert slopes["DOCX"]["n"] == 10
    # ratio = 2 * ki^0.3, so log10 ratio = log10 2 + 0.3 * log10 ki, and
    # log10 ki is not log10 IC50 -- the fitted slope is 0.3/1.3, not 0.3.
    assert slopes["DOCX"]["slope"] == pytest.approx(0.3 / 1.3, rel=0.02)
    assert slopes["DOCX"]["pearson_r"] > 0.99

    # and a flat world must show a flat slope, which is the discriminating
    # part: the same fit must not manufacture a slope where there is none
    flat_rows = [dict(r, ic50_nM=r["ki_nM"] * 7.0) for r in rows]
    flat = pairing.ratio_vs_potency_slope(
        pairing.join_ic50_ki_pairs(flat_rows, flat_rows), min_pairs=5)
    assert flat["DOCX"]["slope"] == pytest.approx(0.0, abs=1e-9)


def test_verdict_refuses_to_validate_from_consistency_alone():
    """No amount of internal consistency may return 'validated' here: the
    absolute conversion cannot be checked without assay [S]. This is the
    guard against the analysis quietly promoting itself."""
    rows = [
        {"molecule_chembl_id": f"A{i}", "document_chembl_id": "DOCA",
         "ic50_nM": (10 ** i) * 11.0, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(8)
    ] + [
        {"molecule_chembl_id": f"B{i}", "document_chembl_id": "DOCB",
         "ic50_nM": (10 ** i) * 9.0, "ki_nM": 10 ** i,
         "assay_description": "Inhibition of DHFR"} for i in range(8)
    ]
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    label, _ = pairing.verdict(pairs)
    assert label != "validated"


def test_verdict_calls_an_impossible_median_unsupported():
    """A median ratio at or below 1 implies [S]/Km <= 0, which no competitive
    assay can have. The verdict must say so instead of reporting a
    plausible-looking distribution."""
    rows = [{"molecule_chembl_id": "A", "document_chembl_id": "DOCA",
             "ic50_nM": 0.5, "ki_nM": 2.0,
             "assay_description": "Inhibition of DHFR"}]
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    label, reasons = pairing.verdict(pairs)
    assert label == "unsupported"
    assert any("outside the range" in r for r in reasons)


def test_verdict_calls_wide_within_paper_scatter_unsupported():
    """If one paper's own compounds disagree ~56-fold in ratio, there is no
    per-assay constant: the label must say 'unsupported', not hedge with
    'partially supported'."""
    rows = []
    for i in range(8):
        r = 10 ** (i * 0.25)  # 0 to 1.75 decades of ratio spread
        rows.append({"molecule_chembl_id": f"X{i}", "document_chembl_id": "DOCX",
                     "ic50_nM": 100.0 * r, "ki_nM": 100.0,
                     "assay_description": "Inhibition of DHFR"})
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    label, reasons = pairing.verdict(pairs)
    assert label == "unsupported"
    assert any("exceeds" in r for r in reasons)


def test_join_loose_mode_pools_across_documents():
    """The loose join must pair on molecule identity even when the documents
    differ, which is exactly why it is the less defensible comparison."""
    ic = [{"molecule_chembl_id": "M1", "document_chembl_id": "D1",
           "ic50_nM": 100.0},
          {"molecule_chembl_id": "M1", "document_chembl_id": "D2",
           "ic50_nM": 400.0}]
    ki = [{"molecule_chembl_id": "M1", "document_chembl_id": "D3",
           "ki_nM": 10.0, "assay_description": "Inhibition of DHFR"}]
    strict = pairing.join_ic50_ki_pairs(ic, ki, strict_document=True)
    loose = pairing.join_ic50_ki_pairs(ic, ki, strict_document=False)
    assert strict == []
    assert len(loose) == 1
    # geometric mean of 100 and 400 is 200
    assert loose[0]["ic50_nM"] == pytest.approx(200.0)
    assert loose[0]["ratio"] == pytest.approx(20.0)
    assert loose[0]["n_ic50_records"] == 2


def test_summarize_ratios_rejects_bad_input():
    with pytest.raises(ValueError):
        pairing.summarize_ratios([])
    with pytest.raises(ValueError):
        pairing.summarize_ratios([1.0, -2.0])
    with pytest.raises(ValueError):
        pairing.summarize_ratios([float("inf")])


def test_assay_residuals_recovers_a_known_substrate_factor():
    """With [S] and Km supplied, the residual should be zero when the data
    was generated by Cheng-Prusoff with that factor."""
    rows = [{"molecule_chembl_id": f"A{i}", "document_chembl_id": "DOCA",
             "ic50_nM": (10 ** i) * 6.0, "ki_nM": 10 ** i,
             "assay_description": "Inhibition of DHFR"} for i in range(5)]
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    res = pairing.assay_residuals(pairs, s_conc=5.0, km=1.0)  # factor 6
    assert res["predicted_ratio"] == pytest.approx(6.0)
    assert res["bias_log10"] == pytest.approx(0.0, abs=1e-12)
    assert res["fold_error_median"] == pytest.approx(1.0)


# --- 3. the real ChEMBL files ------------------------------------------------

@pytest.fixture(scope="module")
def real_pairs():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    return (pairing.join_ic50_ki_pairs(ic50, ki, strict_document=True),
            pairing.join_ic50_ki_pairs(ic50, ki, strict_document=False))


def test_the_paired_set_exists_and_is_the_size_the_loader_implies(real_pairs):
    """The README used to say this comparison was impossible. It is not: the
    two files overlap. These counts are the reason the rest of the file can
    run at all, so they are pinned."""
    strict, loose = real_pairs
    assert len(loose) == 52, "52 compounds appear in both files"
    assert len({p["molecule_chembl_id"] for p in strict}) == 35
    # 36 molecule-document pairs when IC50 records are pooled across assay
    # conditions; CHEMBL1136127 measured 11 of them at two stated [S] (30 and
    # 100 uM), which the substrate-aware join keeps apart: 11 pooled pairs
    # become 22 -> 47.
    assert len(strict) == 47
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    pooled = pairing.join_ic50_ki_pairs(ic50, ki, split_substrate=False)
    assert len(pooled) == 36, "one compound pairs twice, on two documents"


def test_real_ratio_distribution_matches_the_reported_numbers(real_pairs):
    """Pinned from `python scripts/validate_cheng_prusoff.py`. A change in
    either loader, the join, or the pooling moves these."""
    strict, loose = real_pairs
    s = pairing.summarize_ratios([p["ratio"] for p in strict])
    # substrate-aware join (IC50 at 30 and 100 uM kept apart)
    assert s["n"] == 47
    assert s["median"] == pytest.approx(9.375, abs=0.01)
    assert s["q1"] == pytest.approx(4.50, abs=0.01)
    assert s["q3"] == pytest.approx(18.98, abs=0.01)
    assert s["log10_sd"] == pytest.approx(0.409, abs=0.002)
    assert s["min"] == pytest.approx(0.478, abs=0.001)
    assert s["max"] == pytest.approx(29.99, abs=0.01)

    # the earlier pooled-over-[S] join, still reproducible on request
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    old = pairing.summarize_ratios([p["ratio"] for p in
                                    pairing.join_ic50_ki_pairs(
                                        ic50, ki, split_substrate=False)])
    assert old["n"] == 36
    assert old["median"] == pytest.approx(7.47, abs=0.01)
    assert old["q1"] == pytest.approx(6.61, abs=0.01)
    assert old["q3"] == pytest.approx(20.58, abs=0.01)
    assert old["log10_sd"] == pytest.approx(0.407, abs=0.002)
    assert old["min"] == pytest.approx(0.706, abs=0.001)

    sl = pairing.summarize_ratios([p["ratio"] for p in loose])
    assert sl["n"] == 52
    assert sl["median"] == pytest.approx(6.89, abs=0.01)
    # the loose join has a far longer tail, which is what pooling across
    # assay conditions does and why the strict join is the one quoted
    assert sl["max"] > 2000
    assert sl["log10_sd"] > s["log10_sd"]


def test_real_within_paper_ratios_are_tighter_than_between(real_pairs):
    """The central result. Inside a publication the ratio is reproducible to
    about 2-fold; the papers themselves differ by more than that. That is the
    per-assay structure Cheng-Prusoff requires."""
    strict, _ = real_pairs
    spread = pairing.within_document_spread(strict)
    assert spread["pooled_log10_sd"] == pytest.approx(0.288, abs=0.005)
    assert 10 ** spread["pooled_log10_sd"] < 2.5, "within-paper spread under 2.5x"

    de = pairing.document_effect(strict)
    assert de["n_groups_used"] == 5  # CHEMBL1136127 splits into 30 and 100 uM
    assert de["document_explains"] == pytest.approx(0.513, abs=0.01)
    assert de["document_explains"] > 0.5, "paper identity explains most variance"


def test_real_per_document_medians_are_ordered(real_pairs):
    """The four papers must land at the medians measured, ascending by
    compound count is not the point -- the ordering by ratio is: 3.0, 4.2,
    6.8, 20.9. Each is a different implied [S]/Km."""
    strict, _ = real_pairs
    spread = pairing.within_document_spread(strict)
    med = {d: v["median_ratio"] for d, v in spread["per_document"].items()}
    assert med["CHEMBL5233242"] == pytest.approx(3.04, abs=0.05)
    assert med["CHEMBL1124568"] == pytest.approx(4.21, abs=0.05)
    # CHEMBL1136127 ran IC50 at two stated [S]; each condition is its own
    # group (the old single median 6.79 blended them)
    assert "CHEMBL1136127" not in med
    assert med["CHEMBL1136127@30uM"] == pytest.approx(4.63, abs=0.05)
    assert med["CHEMBL1136127@100uM"] == pytest.approx(9.74, abs=0.05)
    assert med["CHEMBL3632554"] == pytest.approx(20.95, abs=0.05)
    implied = {d: pairing.implied_substrate_ratio(v) for d, v in med.items()}
    assert implied["CHEMBL3632554"] == pytest.approx(19.95, abs=0.05)


def test_real_largest_paper_shows_no_potency_trend(real_pairs):
    """Fourteen compounds spanning 1150 to 1.55e6 nM, one assay, one ratio.
    A slope near zero over three orders of magnitude is the strongest
    single piece of evidence in this dataset that the ratio is per-assay."""
    strict, _ = real_pairs
    slopes = pairing.ratio_vs_potency_slope(strict)
    d = slopes["CHEMBL3632554"]
    assert d["n"] == 14
    assert abs(d["slope"]) < 0.1
    assert d["ic50_max_nM"] / d["ic50_min_nM"] > 1000


def test_real_verdict_is_partial_and_says_why(real_pairs):
    """The verdict must be 'partially supported', not 'validated'. Support
    for the form of the relation is real; validation of its values needs the
    assay [S], which these files do not carry."""
    strict, loose = real_pairs
    label, reasons = pairing.verdict(strict)
    assert label == "partially supported"
    assert len(reasons) >= 2
    assert all(isinstance(r, str) and r for r in reasons)
    # the missing input must be named rather than gestured at
    assert any("[S]" in r for r in reasons)

    # the loose join pairs across papers, so it has no within-paper
    # structure to test: it must say so, not borrow the overall SD as a
    # "within-paper" one
    loose_label, loose_reasons = pairing.verdict(loose)
    assert loose_label == "not testable"
    assert any("no paper" in r for r in loose_reasons)


def test_real_pairs_below_the_competitive_floor_are_flagged(real_pairs):
    """Two pairs have IC50 below measured Ki. The competitive model forbids
    that, so they must be visible as flagged rows rather than dropped or
    quietly clamped."""
    strict, _ = real_pairs
    bad = [p for p in strict if not pairing.is_physical_ratio(p["ratio"])]
    assert len(bad) == 2
    assert {p["molecule_chembl_id"] for p in bad} == {"CHEMBL82985", "CHEMBL1162149"}
    for p in bad:
        assert pairing.implied_substrate_ratio(p["ratio"]) < 0


def test_real_enzyme_subset_is_reported_separately(real_pairs):
    """Seven pairs come from a paper whose 'Ki' is whole-cell antibacterial
    potency. They stay in the data and are labelled, and the enzyme-only
    subset is available for the cleaner comparison."""
    strict, _ = real_pairs
    non = [p for p in strict if p["ki_assay_class"] == "antibacterial_described"]
    enzyme = [p for p in strict if p["ki_assay_class"] == "enzyme"]
    assert len(non) == 7
    assert len(enzyme) == 40
    assert all("antibacterial" in p["ki_assay_description"].lower() for p in non)
    se = pairing.summarize_ratios([p["ratio"] for p in enzyme])
    assert se["median"] == pytest.approx(9.99, abs=0.01)
    # pooled-over-[S] join: the sensitivity analysis that first moved the
    # median from 7.47 to 14.74
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    pooled = pairing.join_ic50_ki_pairs(ic50, ki, split_substrate=False)
    pooled_enz = [p for p in pooled if p["ki_assay_class"] == "enzyme"]
    assert len(pooled_enz) == 29
    assert pairing.summarize_ratios(
        [p["ratio"] for p in pooled_enz])["median"] == pytest.approx(14.74, abs=0.01)


def test_pinned_asset_files_exist_and_agree():
    """The results CSVs are committed as the record of this run. Their
    headline row must match what the code computes now: a stale CSV would be
    a fabricated result sitting in the repository."""
    from pathlib import Path
    import csv

    root = Path(__file__).resolve().parents[1]
    stats_path = root / "results" / "cheng_prusoff_ratio_stats.csv"
    pairs_path = root / "results" / "cheng_prusoff_pairs.csv"
    assert stats_path.exists() and pairs_path.exists()

    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    strict = pairing.join_ic50_ki_pairs(ic50, ki, strict_document=True)
    now = pairing.summarize_ratios([p["ratio"] for p in strict])

    rows = {r["subset"]: r for r in csv.DictReader(stats_path.open())}
    row = rows["same_document_all"]
    assert int(row["n"]) == now["n"]
    assert float(row["median_ratio"]) == pytest.approx(now["median"], rel=1e-4)
    assert float(row["log10_sd"]) == pytest.approx(now["log10_sd"], rel=1e-4)
    assert row["verdict"] == "partially supported"

    pair_rows = list(csv.DictReader(pairs_path.open()))
    assert len(pair_rows) == 47 + 52
    assert len([r for r in pair_rows if r["join_mode"] == "same_document"]) == 47
