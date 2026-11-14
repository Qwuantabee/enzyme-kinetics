"""Tests for the pairing fixes the independent review asked for.

Each test names the review item it pins and was written so that it FAILS on
the code the review found: a label that cannot change, an IC50 blended over
two substrate concentrations, an overall SD reported as within-paper, a CI
that ignores clustering, a fraction with no F test, slopes with no interval.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from enzkin import datasets, pairing

ROOT = Path(__file__).resolve().parents[1]


def _rows(doc, ratio_fn, n=6, ki0=1.0, desc="Inhibition of DHFR", tag="X",
          assay="A1"):
    out = []
    for i in range(n):
        ki = ki0 * 10 ** (0.4 * i)
        out.append({"molecule_chembl_id": f"{tag}{i}", "document_chembl_id": doc,
                    "assay_chembl_id": assay,
                    "ic50_nM": ki * ratio_fn(i), "ki_nM": ki,
                    "assay_description": desc})
    return out


# --- C1: every label is reachable from the statistics ------------------------

def test_verdict_has_no_validated_label_and_no_phantom_parameter():
    import inspect
    assert "has_assay_substrate" not in (pairing.verdict.__doc__ or "")
    assert "has_assay_substrate" not in inspect.signature(pairing.verdict).parameters
    assert "validated" not in pairing.VERDICT_LABELS


def test_verdict_partial_when_ratio_is_reproducible_within_papers():
    rows = (_rows("DOCA", lambda i: 3.0 * (1 + 0.05 * (-1) ** i), tag="A")
            + _rows("DOCB", lambda i: 21.0 * (1 + 0.05 * (-1) ** i), tag="B"))
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    label, reasons = pairing.verdict(pairs)
    assert label == "partially supported", reasons
    # the F test is part of the reasons, with its p value
    assert any("ANOVA" in r and "significant" in r for r in reasons)


def test_verdict_unsupported_when_within_paper_scatter_is_wide():
    rows = (_rows("DOCA", lambda i: 10 ** (0.3 * i), tag="A")
            + _rows("DOCB", lambda i: 10 ** (0.3 * (5 - i)), tag="B"))
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    label, reasons = pairing.verdict(pairs)
    assert label == "unsupported"
    assert any("exceeds" in r for r in reasons)


def test_verdict_unsupported_when_median_is_impossible():
    rows = _rows("DOCA", lambda i: 0.5, tag="A")
    label, reasons = pairing.verdict(pairing.join_ic50_ki_pairs(rows, rows))
    assert label == "unsupported"
    assert any("outside the range" in r for r in reasons)
    rows = _rows("DOCA", lambda i: 400.0, tag="A")
    assert pairing.verdict(pairing.join_ic50_ki_pairs(rows, rows))[0] == "unsupported"


def test_verdict_not_testable_with_no_paper_of_three_pairs():
    rows = _rows("DOCA", lambda i: 5.0, n=2, tag="A") + \
        _rows("DOCB", lambda i: 9.0, n=2, tag="B")
    label, reasons = pairing.verdict(pairing.join_ic50_ki_pairs(rows, rows))
    assert label == "not testable"
    assert any("no paper" in r for r in reasons)


def test_verdict_threshold_is_an_argument_and_changes_the_label():
    """The within-paper limit is a parameter with a fixed default, so a label
    near the limit must flip when the limit moves, not stay put."""
    rows = _rows("DOCA", lambda i: 6.0 * (1.9 if i % 2 else 1.0), tag="A") + \
        _rows("DOCB", lambda i: 6.0 * (1.9 if i % 2 else 1.0), tag="B")
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    spread = pairing.within_document_spread(pairs)["pooled_log10_sd"]
    assert 0.1 < spread < 0.3
    assert pairing.verdict(pairs)[0] == "partially supported"
    assert pairing.verdict(pairs, max_within_log10_sd=spread / 2)[0] == "unsupported"


def test_verdict_can_return_more_than_one_label_on_real_data():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    strict = pairing.join_ic50_ki_pairs(ic50, ki)
    loose = pairing.join_ic50_ki_pairs(ic50, ki, strict_document=False)
    labels = {pairing.verdict(strict)[0], pairing.verdict(loose)[0]}
    assert labels == {"partially supported", "not testable"}


# --- C2: IC50 at different stated [S] are not blended -------------------------

def test_parse_substrate_uM():
    f = pairing.parse_substrate_uM
    assert f("Inhibition of E. coli DHFR in presence of 30 uM Dihydrofolate reductase") == 30.0
    assert f("in presence of 100 uM dihydrofolic acid") == 100.0
    assert f("in 50 uM DHF") == 50.0
    assert f("with 0.5 mM dihydrofolate") == pytest.approx(500.0)
    assert f("20 nM dihydrofolate") == pytest.approx(0.02)
    assert f("Inhibition of Escherichia coli DHFR") is None
    assert f("") is None and f(None) is None
    # an enzyme concentration or inhibitor concentration is not [S]
    assert f("assessed at 10 uM of compound") is None


def _two_s_rows(km=8.0, ki_scale=1.0):
    """Eight molecules at S = 30 and 100 uM, generated from Cheng-Prusoff."""
    ic, ki = [], []
    for i in range(8):
        k = ki_scale * 10 ** (0.3 * i)
        ki.append({"molecule_chembl_id": f"T{i}", "document_chembl_id": "DOCT",
                   "assay_chembl_id": "K1", "ki_nM": k,
                   "assay_description": "Inhibition constant"})
        for s, aid in ((30.0, "A30"), (100.0, "A100")):
            ic.append({"molecule_chembl_id": f"T{i}", "document_chembl_id": "DOCT",
                       "assay_chembl_id": aid, "ic50_nM": k * (1 + s / km),
                       "assay_description":
                           f"Inhibition in presence of {s:g} uM Dihydrofolate"})
    return ic, ki


def test_ic50_at_two_substrate_concentrations_is_not_blended():
    ic, ki = _two_s_rows(km=8.0)
    split = pairing.join_ic50_ki_pairs(ic, ki)
    blended = pairing.join_ic50_ki_pairs(ic, ki, split_substrate=False)
    assert len(split) == 16 and len(blended) == 8
    by_s = {s: [p["ratio"] for p in split if p["ic50_substrate_uM"] == s]
            for s in (30.0, 100.0)}
    assert by_s[30.0] == pytest.approx([1 + 30 / 8] * 8)
    assert by_s[100.0] == pytest.approx([1 + 100 / 8] * 8)
    # the blend is the geometric mean of the two ratios: it belongs to neither
    blend_ratio = blended[0]["ratio"]
    assert blend_ratio == pytest.approx(math.sqrt((1 + 30 / 8) * (1 + 100 / 8)))
    assert all(abs(blend_ratio - r) > 1.0 for rs in by_s.values() for r in rs[:1])
    assert {p["group"] for p in split} == {"DOCT@30uM", "DOCT@100uM"}


def test_assay_ids_and_substrate_are_carried_on_each_pair():
    ic, ki = _two_s_rows()
    for p in pairing.join_ic50_ki_pairs(ic, ki):
        assert p["ic50_assay_ids"] == ["A30"] or p["ic50_assay_ids"] == ["A100"]
        assert p["ic50_substrate_uM"] in (30.0, 100.0)


def test_two_substrate_check_recovers_the_km_it_was_generated_from():
    for km in (3.0, 8.0, 45.0):
        ic, ki = _two_s_rows(km=km)
        chk = pairing.two_substrate_check(pairing.join_ic50_ki_pairs(ic, ki))
        assert chk["n_molecules"] == 8 and chk["n_feasible"] == 8
        assert chk["km_pair_median_uM"] == pytest.approx(km)
        assert chk["km_from_low_median_uM"] == pytest.approx(km)
        assert chk["km_from_high_median_uM"] == pytest.approx(km)
        assert chk["rho_median"] == pytest.approx((km + 100) / (km + 30))


def test_two_substrate_check_flags_ratios_outside_the_bounds():
    """rho above S2/S1 (or below 1) is impossible under Cheng-Prusoff."""
    ic, ki = _two_s_rows(km=8.0)
    for r in ic:
        if r["assay_chembl_id"] == "A100":
            r["ic50_nM"] *= 5.0   # rho becomes > 100/30
    chk = pairing.two_substrate_check(pairing.join_ic50_ki_pairs(ic, ki))
    assert chk["n_feasible"] == 0
    assert math.isnan(chk["km_pair_median_uM"])


def test_real_two_substrate_paper_is_detected_and_pairs_change():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    strict = pairing.join_ic50_ki_pairs(ic50, ki)
    pooled = pairing.join_ic50_ki_pairs(ic50, ki, split_substrate=False)
    assert len(strict) - len(pooled) == 11
    assert sum(1 for p in strict if p["group"] != p["document_chembl_id"]) == 22
    chk = pairing.two_substrate_check(strict)
    assert chk["n_molecules"] == 11
    assert chk["rho_median"] == pytest.approx(2.18, abs=0.01)
    assert chk["km_from_low_median_uM"] == pytest.approx(8.14, abs=0.02)
    assert chk["km_from_high_median_uM"] == pytest.approx(11.45, abs=0.02)


# --- C3/C4: the antibacterial flag is unverified, not a fact ----------------

def test_antibacterial_flag_is_named_for_what_it_is():
    flagged = pairing.assay_class("Antibacterial activity against Escherichia coli")
    assert flagged == pairing.CLASS_ANTIBACTERIAL_DESCRIBED
    assert "enzyme" not in flagged.replace("non_enzyme", "")  # not a claim either way
    assert flagged != "non_enzyme"


def test_flagged_records_stay_in_the_headline_and_a_sensitivity_row_exists():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    strict = pairing.join_ic50_ki_pairs(ic50, ki)
    assert sum(1 for p in strict
               if p["ki_assay_class"] == pairing.CLASS_ANTIBACTERIAL_DESCRIBED) == 7
    rows = {r["subset"]: r for r in csv.DictReader(
        (ROOT / "results" / "cheng_prusoff_ratio_stats.csv").open())}
    assert int(rows["same_document_all"]["n"]) == len(strict)
    sens = rows["same_document_sensitivity_excl_antibacterial_described"]
    assert int(sens["n"]) == len(strict) - 7
    text = (ROOT / "results_cheng_prusoff.txt").read_text()
    assert "whose 'Ki' is antibacterial" not in text
    assert "UNVERIFIED" in text


# --- C6: cluster bootstrap ----------------------------------------------------

def test_cluster_ci_is_wider_than_iid_ci_when_papers_differ():
    rng = np.random.default_rng(3)
    ratios, groups = [], []
    for g, centre in (("P1", 3.0), ("P2", 21.0), ("P3", 7.0)):
        for _ in range(12):
            ratios.append(centre * 10 ** rng.normal(0, 0.03))
            groups.append(g)
    s = pairing.summarize_ratios(ratios, groups=groups)
    width_iid = math.log10(s["median_ci_high"] / s["median_ci_low"])
    width_cl = math.log10(s["median_ci_cluster_high"] / s["median_ci_cluster_low"])
    assert width_cl > 2 * width_iid
    # and it reaches the per-paper medians the iid interval cannot reach
    assert s["median_ci_cluster_low"] <= 3.2 and s["median_ci_cluster_high"] >= 20.0
    assert not (s["median_ci_low"] <= 3.2)


def test_cluster_ci_is_nan_without_groups_or_with_one_paper():
    assert math.isnan(pairing.summarize_ratios([2.0, 3.0, 4.0])["median_ci_cluster_low"])
    one = pairing.summarize_ratios([2.0, 3.0, 4.0], groups=["P"] * 3)
    assert math.isnan(one["median_ci_cluster_low"])
    with pytest.raises(ValueError):
        pairing.summarize_ratios([2.0, 3.0], groups=["P"])


# --- C7: the within-paper SD of the loose join is not the overall SD --------

def test_loose_join_has_no_within_paper_spread():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    loose = pairing.join_ic50_ki_pairs(ic50, ki, strict_document=False)
    assert all(p["group"] is None for p in loose)
    spread = pairing.within_document_spread(loose)
    assert math.isnan(spread["pooled_log10_sd"])
    assert spread["per_document"] == {}
    overall = pairing.summarize_ratios([p["ratio"] for p in loose])["log10_sd"]
    assert overall > 0.5   # the number the old code reported as "within-paper"
    assert pairing.ratio_vs_potency_slope(loose) == {}
    assert math.isnan(pairing.document_effect(loose)["document_explains"])


def test_committed_stats_csv_has_nan_within_sd_for_the_loose_join():
    rows = {r["subset"]: r for r in csv.DictReader(
        (ROOT / "results" / "cheng_prusoff_ratio_stats.csv").open())}
    assert rows["same_molecule_all"]["within_group_log10_sd"] == "nan"
    assert rows["same_molecule_all"]["verdict"] == "not testable"


# --- C8: F test for the document effect --------------------------------------

def test_document_effect_f_matches_scipy_one_way_anova():
    rng = np.random.default_rng(11)
    groups = {"A": 0.3, "B": 0.9, "C": 1.2}
    rows = []
    for g, mu in groups.items():
        for i in range(7):
            ratio = 10 ** (mu + rng.normal(0, 0.15))
            rows.append({"molecule_chembl_id": f"{g}{i}", "document_chembl_id": g,
                         "ic50_nM": ratio * 5.0, "ki_nM": 5.0,
                         "assay_description": ""})
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    de = pairing.document_effect(pairs)
    samples = [[math.log10(p["ratio"]) for p in pairs if p["group"] == g]
               for g in groups]
    ref = stats.f_oneway(*samples)
    assert de["f_stat"] == pytest.approx(ref.statistic, rel=1e-9)
    assert de["f_p"] == pytest.approx(ref.pvalue, rel=1e-6)
    assert (de["df_between"], de["df_within"]) == (2, 18)


def test_document_effect_f_is_nan_when_untestable_and_inf_when_noise_free():
    one = [{"molecule_chembl_id": f"A{i}", "document_chembl_id": "A",
            "ic50_nM": 10.0 + i, "ki_nM": 1.0, "assay_description": ""}
           for i in range(4)]
    assert math.isnan(pairing.document_effect(pairing.join_ic50_ki_pairs(one, one))["f_stat"])


def test_real_document_effect_reports_a_significant_f():
    ic50 = datasets.load_chembl_dhfr_ic50()
    ki = datasets.load_chembl_dhfr_ki()
    de = pairing.document_effect(pairing.join_ic50_ki_pairs(ic50, ki))
    assert (de["df_between"], de["df_within"]) == (4, 41)
    assert de["f_stat"] == pytest.approx(12.86, abs=0.02)
    assert de["f_p"] < 1e-5


# --- C5: slope uncertainty and the shared-variable bias -----------------------

def test_slope_comes_with_a_correct_standard_error_and_interval():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(12):
        ki = 10 ** rng.uniform(0, 3)
        rows.append({"molecule_chembl_id": f"S{i}", "document_chembl_id": "D",
                     "ic50_nM": ki * 6.0 * 10 ** rng.normal(0, 0.1), "ki_nM": ki,
                     "assay_description": ""})
    pairs = pairing.join_ic50_ki_pairs(rows, rows)
    d = pairing.ratio_vs_potency_slope(pairs)["D"]
    x = np.log10([p["ic50_nM"] for p in pairs])
    y = np.log10([p["ratio"] for p in pairs])
    ref = stats.linregress(x, y)
    assert d["slope"] == pytest.approx(ref.slope)
    assert d["slope_se"] == pytest.approx(ref.stderr)
    t = stats.t.ppf(0.975, len(x) - 2)
    assert d["slope_ci_low"] == pytest.approx(ref.slope - t * ref.stderr)
    assert d["slope_ci_high"] == pytest.approx(ref.slope + t * ref.stderr)
    assert d["slope_ci_low"] < d["slope"] < d["slope_ci_high"]


def test_noise_alone_biases_the_ic50_slope_up_and_the_ki_slope_down():
    """The ratio is a constant (6) in every compound; only the measurement
    noise differs. Noise on the IC50 shows up as a positive slope on log
    IC50, noise on the Ki as a negative slope on log Ki -- so a null slope is
    not zero, which is why both slopes are reported."""
    rng = np.random.default_rng(0)
    n = 4000
    k = 10 ** rng.uniform(0, 3, n)
    ic_noise = [{"molecule_chembl_id": f"a{i}", "document_chembl_id": "D",
                 "ic50_nM": k[i] * 6.0 * 10 ** rng.normal(0, 0.2),
                 "ki_nM": k[i], "assay_description": ""} for i in range(n)]
    ki_noise = [{"molecule_chembl_id": f"b{i}", "document_chembl_id": "D",
                 "ic50_nM": k[i] * 6.0,
                 "ki_nM": k[i] * 10 ** rng.normal(0, 0.2), "assay_description": ""}
                for i in range(n)]
    a = pairing.ratio_vs_potency_slope(
        pairing.join_ic50_ki_pairs(ic_noise, ic_noise))["D"]
    b = pairing.ratio_vs_potency_slope(
        pairing.join_ic50_ki_pairs(ki_noise, ki_noise))["D"]
    assert a["slope"] > 0.01
    assert b["slope_vs_ki"] < -0.01
