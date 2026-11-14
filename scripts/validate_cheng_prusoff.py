"""Validate (or fail to validate) Cheng-Prusoff against measured Ki values.

    python scripts/validate_cheng_prusoff.py

The question
------------
scripts/ic50_to_ki.py applies Cheng-Prusoff to the measured IC50s for E. coli
dihydrofolate reductase and reports that it cannot check the result, because
ChEMBL records carry no assay [S] or [E] as fields. That remains true for most
assays. But it is not the end of the story: this repository also holds
measured Ki values for the same target, and a subset of compounds appears in
both tables; for most of those the IC50 and the Ki come from the same
publication. (The counts are printed by the run, not written here.)

So the conversion CAN be checked -- against measured Ki, not against a
re-derived IC50. What it cannot be checked against is each compound's own
assay [S], except for the few assays whose free-text description states it.
This script draws that line as precisely as the data allows.

What is tested, and what it means
---------------------------------
Cheng-Prusoff for a competitive inhibitor is IC50 = Ki (1 + [S]/Km), so

    IC50 / Ki = 1 + [S]/Km

which is a property of the *assay*, not the compound. Consequences testable
from these files:

1. The ratio is at least 1 (since [S]/Km >= 0).
2. The ratio is roughly constant within one assay condition -- one protocol,
   one [S]/Km, one ratio -- however much the compounds' potency varies.
3. The ratio may differ between papers, because their [S]/Km may differ. The
   relation permits this; it does not require it.
4. Where the same compounds were run at two stated [S], IC50(S2)/IC50(S1)
   must equal (Km + S2)/(Km + S1): a direct, Ki-free check.

A ratio that instead tracked potency (tight-binding compounds corrected more
than weak ones) would falsify the relation; that is examined by fitting
log(ratio) against potency within each paper, with the interval, and with the
caveat that the response shares a variable with the regressor.

What these files still cannot do
--------------------------------
Pin the value of [S]/Km for most assays. Any observed ratio above 1 is
consistent with *some* substrate concentration, so consistency is necessary
and not sufficient. The verdict is a statement about the *shape* of the data;
every sentence of the report that states a number or a reading is built from
the numbers computed in the same run, not typed in.

Output
------
    results/cheng_prusoff_pairs.csv        every pair, both join modes
    results/cheng_prusoff_ratio_stats.csv  headline statistics
    results_cheng_prusoff.txt              the readable report
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import datasets, pairing  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "results_cheng_prusoff.txt"

#: What a Cheng-Prusoff test needs and most of the files do not have.
MISSING = ("assay substrate concentration [S] (stated for only some IC50 "
           "assays, in free text)",
           "assay enzyme concentration [E]",
           "the Km of the assay's substrate under the assay's conditions")

ANTIBACTERIAL = pairing.CLASS_ANTIBACTERIAL_DESCRIBED


def _fmt(x, w=8, p=2):
    return f"{x:>{w}.{p}f}"


def _kind(group, pairs):
    return ("antibacterial-described Ki" if any(
        _doc(p) == group.split("@")[0] and p["ki_assay_class"] == ANTIBACTERIAL
        for p in pairs) else "enzyme Ki")


def _doc(p):
    return p["document_chembl_id"]


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    ic50 = datasets.load_chembl_dhfr_ic50()
    ki_all = datasets.load_chembl_dhfr_ki(wild_type_only=False)
    ki_wt = datasets.load_chembl_dhfr_ki()

    strict = pairing.join_ic50_ki_pairs(ic50, ki_wt, strict_document=True)
    pooled = pairing.join_ic50_ki_pairs(ic50, ki_wt, strict_document=True,
                                        split_substrate=False)
    loose = pairing.join_ic50_ki_pairs(ic50, ki_wt, strict_document=False)
    enzyme = [p for p in strict if p["ki_assay_class"] == "enzyme"]
    flagged_keys = {(r["molecule_chembl_id"], r["document_chembl_id"])
                    for r in ic50 if r["outside_typical_range"]}
    no_flag = pairing.join_ic50_ki_pairs(
        datasets.load_chembl_dhfr_ic50(exclude_implausible=True), ki_wt)
    n_flag_rows = sum(1 for r in ic50 if r["outside_typical_range"])
    n_flag_pairs = sum(1 for p in strict
                       if (p["molecule_chembl_id"], _doc(p)) in flagged_keys)

    out("=" * 78)
    out("Cheng-Prusoff vs measured Ki: E. coli dihydrofolate reductase")
    out("=" * 78)
    out(f"target        ChEMBL {datasets.CHEMBL_DHFR_TARGET}")
    out(f"IC50 records  {len(ic50)} over "
        f"{len({r['molecule_chembl_id'] for r in ic50})} compounds and "
        f"{len({r['document_chembl_id'] for r in ic50})} publications; "
        f"{n_flag_rows} are above {datasets.IC50_IMPLAUSIBLE_NM:g} nM "
        f"(flagged outside_typical_range, kept)")
    out(f"Ki records    {len(ki_all)} total, {len(ki_wt)} wild-type "
        f"({len(ki_all) - len(ki_wt)} point-mutant records excluded by the "
        f"load_chembl_dhfr_ki mutant filter)")
    out(f"compounds in both files        "
        f"{len({r['molecule_chembl_id'] for r in ic50} & {r['molecule_chembl_id'] for r in ki_wt})}")
    out(f"  ... sharing a publication    {len({p['molecule_chembl_id'] for p in strict})} "
        f"({len(strict)} pairs; {len(pooled)} if IC50 records of different "
        f"stated [S] are pooled)")
    out(f"  ... paired on molecule only  {len(loose)}")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(a) observed IC50/Ki ratio distribution")
    out("-" * 78)
    out("Ratios are formed per (molecule, document, stated IC50 [S]) after")
    out("geometric-mean pooling of replicate measurements, because IC50 and Ki")
    out("are log-scale quantities.")
    out()
    out(f"{'set':<34}{'n':>4}{'median':>9}{'Q1':>8}{'Q3':>8}{'log10SD':>9}{'range':>16}")
    stats = {}
    for label, ps in (("same publication (strict)", strict),
                      ("same molecule (loose)", loose),
                      ("strict, IC50 pooled over [S]", pooled),
                      ("strict, excl. antibacterial-desc.", [p for p in strict if p["ki_assay_class"] == "enzyme"]),
                      ("strict, excl. IC50 > 1e6 nM", no_flag)):
        s = pairing.summarize_ratios([p["ratio"] for p in ps],
                                     groups=[p.get("group") for p in ps])
        stats[label] = s
        out(f"{label:<34}{s['n']:>4}{_fmt(s['median'], 9)}"
            f"{_fmt(s['q1'], 8)}{_fmt(s['q3'], 8)}"
            f"{_fmt(s['log10_sd'], 9, 3)}"
            f"{s['min']:>7.2f}-{s['max']:<8.1f}")

    s = stats["same publication (strict)"]
    sub_one = [p for p in strict if not pairing.is_physical_ratio(p["ratio"])]
    out()
    out(f"Median ratio, same-publication pairs: {s['median']:.2f}, IQR "
        f"{s['q1']:.2f} to {s['q3']:.2f}, n = {s['n']}.")
    out(f"  95% bootstrap CI, i.i.d. (treats pairs as independent): "
        f"{s['median_ci_low']:.2f} to {s['median_ci_high']:.2f}")
    out(f"  95% bootstrap CI, cluster (resamples papers, then pairs; "
        f"{len({p['group'] for p in strict})} clusters): "
        f"{s['median_ci_cluster_low']:.2f} to {s['median_ci_cluster_high']:.2f}")
    out("  The cluster interval is the one to read: pairs from one paper share a")
    out("  protocol. With this few clusters it is coarse and may under-cover.")
    out(f"{len(sub_one)} of {s['n']} pairs sit below 1, which the competitive model")
    out("forbids (" + ", ".join(f"{p['molecule_chembl_id']} {p['ratio']:.2f}"
                                for p in sub_one) + "). A ratio below 1 can come")
    out("from a non-competitive mechanism, mismatched conditions or a record")
    out("error; this data cannot say which. They stay in the headline number.")
    out(f"The full span is {s['fold_span']:.1f}-fold across papers; the per-paper")
    out("spreads below are the more informative numbers.")

    n_changed = sum(1 for p in strict if p["group"] != _doc(p))
    out()
    out(f"Effect of keeping IC50 records at different stated [S] apart "
        f"(C2): {len(pooled)} pooled pairs -> {len(strict)} pairs; "
        f"{n_changed} pairs "
        f"({len({p['molecule_chembl_id'] for p in strict if p['group'] != _doc(p)})} "
        f"molecules) come from the split document.")
    out(f"Effect of excluding the {n_flag_pairs} pair(s) whose IC50 is above "
        f"{datasets.IC50_IMPLAUSIBLE_NM:g} nM: n {s['n']} -> {len(no_flag)}.")

    spread = pairing.within_document_spread(strict)
    out()
    out("Per assay condition (publication, or publication@[S] where one "
        "publication")
    out("ran IC50 at two stated substrate concentrations):")
    out(f"{'group':<24}{'n':>4}{'median ratio':>14}{'within spread':>16}  implied [S]/Km")
    for doc, d in sorted(spread["per_document"].items(),
                         key=lambda t: -t[1]["n"]):
        out(f"{doc:<24}{d['n']:>4}{_fmt(d['median_ratio'], 14)}"
            f"{_fmt(d['fold_spread'], 14, 2)}x"
            f"{pairing.implied_substrate_ratio(d['median_ratio']):>15.1f}")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(b) what Cheng-Prusoff implies, and what assumption reconciles it")
    out("-" * 78)
    out("For a competitive inhibitor, IC50 = Ki (1 + [S]/Km). Inverting the")
    out("observed ratio gives the assay's substrate factor directly:")
    out()
    out(f"{'observed ratio':>16}{'implied [S]/Km':>18}")
    meds = [d["median_ratio"] for d in spread["per_document"].values()]
    for r in (1.0, 2.0, s["median"], min(meds), max(meds)):
        out(f"{r:>16.2f}{pairing.implied_substrate_ratio(r):>18.2f}")
    out()
    out(f"Per-group median ratios run from {min(meds):.1f} to {max(meds):.1f}, "
        f"i.e. implied [S]/Km from")
    out(f"{pairing.implied_substrate_ratio(min(meds)):.1f} to "
        f"{pairing.implied_substrate_ratio(max(meds)):.1f}. The pooled median "
        f"({s['median']:.1f}) is an average over those and")
    out("is not a conversion factor for any one assay.")
    out()
    out("The assumptions behind reading a ratio as an assay factor: every pair")
    out("is a competitive inhibitor; each group ran a single [S]; and each IC50")
    out("is a true 50% inhibition point rather than a fitted midpoint off a")
    out("partial curve. Relax any of them and the ratio stops being an assay")
    out("constant. Nothing in these files constrains [S] for most assays, so for")
    out("those no pair can confirm or refute the correction it implies.")

    # ---- two-[S] direct check
    chk = pairing.two_substrate_check(strict)
    out()
    out("Direct (1 + [S]/Km) check: compounds measured at two stated [S]")
    if chk["n_molecules"] == 0:
        out("  none found in this data.")
    else:
        r0 = chk["rows"][0]
        s_lo, s_hi = r0["s_low_uM"], r0["s_high_uM"]
        docs = sorted({r["document_chembl_id"] for r in chk["rows"]})
        ki_desc = sorted({r["assay_description"] for r in ki_wt
                          if r["document_chembl_id"] in docs})
        out(f"  {chk['n_molecules']} compounds in {', '.join(docs)} have IC50 "
            f"at {s_lo:g} and {s_hi:g} uM")
        out(f"  substrate (assay descriptions); their Ki assay description(s): "
            f"{ki_desc}")
        out(f"  (no [S] stated for the Ki).")
        out(f"  Cheng-Prusoff bounds IC50({s_hi:g})/IC50({s_lo:g}) between 1 "
            f"(Km >> S) and {s_hi / s_lo:.2f} (Km << S).")
        out(f"  Measured ratio: median {chk['rho_median']:.2f}, range "
            f"{chk['rho_min']:.2f} to {chk['rho_max']:.2f}; "
            f"{chk['n_feasible']} of {chk['n_molecules']} inside the bounds.")
        out(f"  Km from the IC50 pair alone (Ki-free), median: "
            f"{chk['km_pair_median_uM']:.1f} uM.")
        out(f"  Km from each single condition's IC50/Ki ratio, median: "
            f"{chk['km_from_low_median_uM']:.1f} uM at {s_lo:g} uM, "
            f"{chk['km_from_high_median_uM']:.1f} uM at {s_hi:g} uM.")
        pred_lo = (chk["km_from_low_median_uM"] + s_hi) / (chk["km_from_low_median_uM"] + s_lo)
        pred_hi = (chk["km_from_high_median_uM"] + s_hi) / (chk["km_from_high_median_uM"] + s_lo)
        out(f"  If Km were the single-condition value, the predicted IC50 ratio "
            f"would be {pred_lo:.2f} ({s_lo:g}-based Km) or {pred_hi:.2f} "
            f"({s_hi:g}-based Km),")
        out(f"  against the measured median {chk['rho_median']:.2f}. "
            f"Per-compound values are in results/cheng_prusoff_two_substrate.csv.")
        out("  The pair-derived Km is noisy (it divides by rho - 1 and rho is "
            "near 1 for some compounds),")
        out("  so its median is a coarse number; both Km estimates are reported "
            "as measured, with no interval.")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(c) assay conditions vs genuinely different compounds")
    out("-" * 78)
    out("Same-publication pairs are the tighter comparison: one lab, one")
    out("protocol family, one enzyme preparation. Same-molecule pairs that span")
    out("publications mix all of those, so their scatter has more in it than")
    out("assay variation alone.")
    out()
    de = pairing.document_effect(strict)
    out("log10(ratio) variance decomposition, strict pairs (>=3 pairs/group):")
    out(f"  total   {de['total_var']:.4f}")
    out(f"  between {de['between_var']:.4f}  ({100 * de['document_explains']:.1f}% "
        f"of total)")
    out(f"  within  {de['within_var']:.4f}  ({100 * (1 - de['document_explains']):.1f}%)")
    out(f"  {de['n_groups_used']} groups, {de['n_pairs_used']} pairs")
    out(f"  one-way ANOVA: F({int(de['df_between'])},{int(de['df_within'])}) = "
        f"{de['f_stat']:.2f}, p = {de['f_p']:.2g}")
    for lab, ps in (("excl. antibacterial-described Ki", enzyme),
                    ("IC50 pooled over [S]", pooled)):
        d2 = pairing.document_effect(ps)
        out(f"  same test, {lab}: F({int(d2['df_between'])},{int(d2['df_within'])}) "
            f"= {d2['f_stat']:.2f}, p = {d2['f_p']:.2g}, "
            f"{100 * d2['document_explains']:.1f}% explained")
    out()
    sig = de["f_p"] < 0.05
    out(f"Between-group differences are a "
        f"{10 ** (np.sqrt(de['between_var'])):.1f}-fold effect on the ratio "
        f"(SD-scale);")
    out(f"within-group scatter is {10 ** np.sqrt(de['within_var']):.1f}-fold. "
        f"The ANOVA {'finds' if sig else 'does not find'} the group means "
        f"{'different' if sig else 'distinguishable'} (p = {de['f_p']:.2g}), and "
        f"{100 * de['document_explains']:.0f}% of the variance is attributable")
    out("to which paper/condition a pair came from. That is the pattern a")
    out("per-assay [S]/Km would produce; any per-assay systematic difference")
    out("would produce it too, so it does not single out Cheng-Prusoff from a")
    out("rival explanation.")

    out()
    out("Does the ratio track potency within a group? Slope of log10(ratio) on")
    out("log10(IC50), with the 95% t-interval; and on log10(Ki). The response")
    out("shares a variable with both regressors, so independent noise biases")
    out("the IC50 slope upward and the Ki slope downward: read the interval and")
    out("the pair of signs, not the point estimate alone.")
    out()
    slopes = pairing.ratio_vs_potency_slope(strict)
    big = sorted(slopes.items(), key=lambda t: -t[1]["n"])
    for doc, d in big:
        span = math.log10(d["ic50_max_nM"] / d["ic50_min_nM"])
        out(f"  {doc:<22} n={d['n']:>2}  slope(IC50)={d['slope']:+.3f} "
            f"[{d['slope_ci_low']:+.3f},{d['slope_ci_high']:+.3f}] p={d['slope_p']:.2f}  "
            f"slope(Ki)={d['slope_vs_ki']:+.3f} p={d['slope_vs_ki_p']:.2f}  "
            f"IC50 span {span:.1f} decades  [{_kind(doc, strict)}]")
    excl0 = [doc for doc, d in big if d["slope_ci_low"] > 0 or d["slope_ci_high"] < 0]
    out()
    out(f"{len(big) - len(excl0)} of {len(big)} groups with >=5 pairs have a "
        f"slope interval that includes 0; "
        + (f"the interval excludes 0 for {', '.join(excl0)}."
           if excl0 else "none excludes 0."))
    out("An interval that includes 0 is the absence of a detected trend, not")
    out("evidence of none: groups of 7-14 points have wide intervals, and")
    out("the shortest IC50 span here is "
        f"{min(math.log10(d['ic50_max_nM'] / d['ic50_min_nM']) for _, d in big):.1f} "
        f"decades.")
    out()
    enz_s = stats["strict, excl. antibacterial-desc."]
    out("Sensitivity analysis: the Ki records whose assay description contains")
    out("'antibacterial' (a keyword match; UNVERIFIED label).")
    abx = [p for p in strict if p["ki_assay_class"] == ANTIBACTERIAL]
    tmp = sorted({(r["document_chembl_id"], r["ki_nM"]) for r in ki_wt
                  if r["molecule_chembl_id"] == "CHEMBL22"})
    out(f"  {len(abx)} pairs; their Ki values run {min(p['ki_nM'] for p in abx):.2g}"
        f" to {max(p['ki_nM'] for p in abx):.2g} nM.")
    out("  Trimethoprim (CHEMBL22) Ki by document: "
        + ", ".join(f"{d} {v:g}" for d, v in tmp) + " nM")
    ref = [v for d, v in tmp if d == "CHEMBL1124568"]
    near = [d for d, v in tmp if ref and abs(v - ref[0]) <= 0.05 * ref[0]]
    out(f"  {len(near)} of {len(tmp)} documents (including CHEMBL1124568 itself) "
        f"record trimethoprim within 5% of the")
    out("  value in the antibacterial-described document, several under "
        "enzyme-style descriptions: the wording may be")
    out("  ChEMBL's rather than the measurement's, which this data cannot settle.")
    out("  The rows are not deleted. Excluding them moves the")
    out(f"  median from {s['median']:.2f} to {enz_s['median']:.2f} and the log10 SD from "
        f"{s['log10_sd']:.3f} to {enz_s['log10_sd']:.3f}.")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(d) verdict")
    out("-" * 78)
    label, reasons = pairing.verdict(strict)
    out(f"VERDICT (strict join): {label}")
    for r in reasons:
        out(f"  - {r}")
    for lab, ps in (("loose join", loose), ("excl. antibacterial-desc.", enzyme),
                    ("IC50 pooled over [S]", pooled)):
        lb, rs = pairing.verdict(ps)
        out(f"verdict ({lab}): {lb}")
        out(f"  - {rs[0]}")
    wsd = spread["pooled_log10_sd"]
    lab25, _ = pairing.verdict(strict, max_within_log10_sd=0.25)
    out()
    out(f"Threshold sensitivity: the pooled within-group log10 SD is {wsd:.3f} "
        f"against the fixed")
    out(f"limit 0.30; with a limit of 0.25 the strict-join label would be "
        f"'{lab25}'.")
    out()
    out("What the label means: the ratio is reproducible inside a publication to")
    out(f"within the stated limit and the groups differ (F-test above); that is the")
    out("structure Cheng-Prusoff requires, and support for the relation's *form*.")
    out("It is not validation of its *values*: for most assays the file that would")
    out("settle those -- one carrying [S] alongside IC50 and Ki -- is not here.")
    out()
    per_fold = [d["fold_spread"] for d in spread["per_document"].values()]
    out(f"Qualifications from the numbers: {len(sub_one)} pairs fall below the "
        f"model's floor of ratio 1;")
    out(f"the pooled within-group SD is {10 ** wsd:.1f}-fold, and per-group max/min "
        f"ratio spreads run")
    out(f"from {min(per_fold):.1f}x to {max(per_fold):.1f}x (table above), so the "
        f"ratio is a per-group constant")
    out("with real noise on it rather than a tight one.")
    out()
    out("What is missing, stated exactly:")
    for m in MISSING:
        out(f"  - {m}")
    out()
    out("With [S] and the assay Km, pairing.assay_residuals computes the")
    out("predicted ratio and the per-compound log residual directly; that is")
    out("the check this data cannot run for most assays and a suitable paper")
    out("could.")

    # ------------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    pair_path = RESULTS / "cheng_prusoff_pairs.csv"
    with pair_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["join_mode", "molecule_chembl_id", "document_chembl_id",
                    "group", "ic50_substrate_uM",
                    "ic50_nM", "ki_nM", "ic50_over_ki", "implied_S_over_Km",
                    "physical", "n_ic50_records", "n_ki_records",
                    "ki_assay_class", "ki_assay_description"])
        for mode, ps in (("same_document", strict), ("same_molecule", loose)):
            for p in ps:
                w.writerow([mode, p["molecule_chembl_id"],
                            p["document_chembl_id"] or "",
                            p["group"] or "",
                            "" if p["ic50_substrate_uM"] is None
                            else f"{p['ic50_substrate_uM']:g}",
                            f"{p['ic50_nM']:.6g}", f"{p['ki_nM']:.6g}",
                            f"{p['ratio']:.6g}",
                            f"{pairing.implied_substrate_ratio(p['ratio']):.6g}",
                            pairing.is_physical_ratio(p["ratio"]),
                            p["n_ic50_records"], p["n_ki_records"],
                            p["ki_assay_class"],
                            p["ki_assay_description"]])

    stat_path = RESULTS / "cheng_prusoff_ratio_stats.csv"
    with stat_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subset", "n", "median_ratio", "q1", "q3", "iqr",
                    "log10_sd", "min_ratio", "max_ratio", "fold_span",
                    "median_ci_iid_low", "median_ci_iid_high",
                    "median_ci_cluster_low", "median_ci_cluster_high",
                    "within_group_log10_sd", "document_explains",
                    "anova_F", "anova_p",
                    "implied_S_over_Km_at_median", "verdict"])
        for label, ps in (("same_document_all", strict),
                          ("same_molecule_all", loose),
                          ("same_document_sensitivity_excl_antibacterial_described", enzyme),
                          ("same_document_ic50_pooled_over_S", pooled)):
            st = pairing.summarize_ratios([p["ratio"] for p in ps],
                                          groups=[p["group"] for p in ps])
            sp = pairing.within_document_spread(ps)
            dd = pairing.document_effect(ps)
            vb = pairing.verdict(ps)[0]
            w.writerow([label, st["n"], f"{st['median']:.6g}",
                        f"{st['q1']:.6g}", f"{st['q3']:.6g}", f"{st['iqr']:.6g}",
                        f"{st['log10_sd']:.6g}", f"{st['min']:.6g}",
                        f"{st['max']:.6g}", f"{st['fold_span']:.6g}",
                        f"{st['median_ci_low']:.6g}",
                        f"{st['median_ci_high']:.6g}",
                        f"{st['median_ci_cluster_low']:.6g}",
                        f"{st['median_ci_cluster_high']:.6g}",
                        f"{sp['pooled_log10_sd']:.6g}",
                        f"{dd['document_explains']:.6g}",
                        f"{dd['f_stat']:.6g}", f"{dd['f_p']:.6g}",
                        f"{pairing.implied_substrate_ratio(st['median']):.6g}",
                        vb])

    two_path = RESULTS / "cheng_prusoff_two_substrate.csv"
    with two_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_chembl_id", "document_chembl_id", "s_low_uM",
                    "s_high_uM", "ic50_ratio_high_over_low", "rho_max_possible",
                    "inside_bounds", "km_from_ic50_pair_uM",
                    "km_from_low_uM", "km_from_high_uM"])
        for r in chk["rows"]:
            w.writerow([r["molecule_chembl_id"], r["document_chembl_id"],
                        f"{r['s_low_uM']:g}", f"{r['s_high_uM']:g}",
                        f"{r['rho']:.6g}", f"{r['rho_max_possible']:.6g}",
                        r["feasible"], f"{r['km_from_ic50_pair_uM']:.6g}",
                        f"{r['km_from_low_uM']:.6g}",
                        f"{r['km_from_high_uM']:.6g}"])

    out()
    out(f"wrote {pair_path.relative_to(ROOT)}  ({len(strict) + len(loose)} rows)")
    out(f"wrote {stat_path.relative_to(ROOT)}")
    out(f"wrote {two_path.relative_to(ROOT)}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
