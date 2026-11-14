"""Validate (or fail to validate) Cheng-Prusoff against measured Ki values.

    python scripts/validate_cheng_prusoff.py

The question
------------
`scripts/ic50_to_ki.py` applies Cheng-Prusoff to 482 measured IC50s for E. coli
dihydrofolate reductase and reports that it cannot check the result, because
ChEMBL records carry no assay [S] or [E]. That is true and remains true. But
it is not the end of the story: this repository also holds 266 measured Ki
values for the same target, and 52 compounds appear in both. For 35 of those
the IC50 and the Ki come from the same publication.

So the conversion *can* be checked -- against measured Ki, not against a
re-derived IC50. What it cannot be checked against is each compound's own
assay [S]. This script draws that line as precisely as the data allows.

What is tested, and what it means
---------------------------------
Cheng-Prusoff for a competitive inhibitor is IC50 = Ki (1 + [S]/Km), so

    IC50 / Ki = 1 + [S]/Km

which is a property of the *assay*, not the compound. Three consequences are
testable from these files:

1. The ratio is at least 1 (since [S]/Km >= 0).
2. The ratio is roughly constant within one publication -- one protocol, one
   [S]/Km, one ratio -- however much the compounds' potency varies.
3. The ratio differs between publications, because their [S]/Km differ.

A ratio that instead tracked potency (tight-binding compounds corrected more
than weak ones) would falsify the relation, and that is checked directly in
(row data, below) by fitting log(ratio) against log(IC50) within each paper.

What these files still cannot do
--------------------------------
Pin the value of [S]/Km. Any observed ratio above 1 is consistent with *some*
substrate concentration, so consistency is necessary and not sufficient. A
compound whose ratio is 20 is consistent with Cheng-Prusoff and also with any
other monotone story that produces a per-assay constant. The honest verdict
is therefore a statement about the *shape* of the data, and the report says
so in those words instead of claiming the conversion is validated.

Output
------
    results/cheng_prusoff_pairs.csv        every pair, both join modes
    results/cheng_prusoff_ratio_stats.csv  headline statistics
    results_cheng_prusoff.txt              the readable report
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import cheng_prusoff as cp  # noqa: E402
from enzkin import datasets, pairing  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "results_cheng_prusoff.txt"

#: What the README says a Cheng-Prusoff test needs and the files do not have.
MISSING = ("assay substrate concentration [S]", "assay enzyme concentration [E]",
           "the Km of the assay's substrate under the assay's conditions")


def _fmt(x, w=8, p=2):
    return f"{x:>{w}.{p}f}"


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    ic50 = datasets.load_chembl_dhfr_ic50()
    ki_all = datasets.load_chembl_dhfr_ki(wild_type_only=False)
    ki_wt = datasets.load_chembl_dhfr_ki()

    strict = pairing.join_ic50_ki_pairs(ic50, ki_wt, strict_document=True)
    loose = pairing.join_ic50_ki_pairs(ic50, ki_wt, strict_document=False)
    enzyme = [p for p in strict if p["ki_assay_class"] == "enzyme"]

    out("=" * 78)
    out("Cheng-Prusoff vs measured Ki: E. coli dihydrofolate reductase")
    out("=" * 78)
    out(f"target        ChEMBL {datasets.CHEMBL_DHFR_TARGET}")
    out(f"IC50 records  {len(ic50)}")
    out(f"Ki records    {len(ki_all)} total, {len(ki_wt)} wild-type "
        f"({len(ki_all) - len(ki_wt)} point-mutant records excluded by the "
        f"load_chembl_dhfr_ki mutant filter)")
    out(f"compounds in both files        "
        f"{len({r['molecule_chembl_id'] for r in ic50} & {r['molecule_chembl_id'] for r in ki_wt})}")
    out(f"  ... sharing a publication    {len({p['molecule_chembl_id'] for p in strict})} "
        f"({len(strict)} molecule-document pairs)")
    out(f"  ... paired on molecule only  {len(loose)}")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(a) observed IC50/Ki ratio distribution")
    out("-" * 78)
    out("Ratios are formed per (molecule, document) after geometric-mean "
        "pooling of")
    out("replicate measurements, because IC50 and Ki are log-scale quantities.")
    out()
    out(f"{'set':<26}{'n':>4}{'median':>9}{'Q1':>8}{'Q3':>8}{'log10SD':>9}{'range':>16}")
    stats = {}
    for label, ps in (("same publication (strict)", strict),
                      ("same molecule (loose)", loose),
                      ("strict, enzyme assays only", enzyme)):
        s = pairing.summarize_ratios([p["ratio"] for p in ps])
        stats[label] = s
        out(f"{label:<26}{s['n']:>4}{_fmt(s['median'], 9)}"
            f"{_fmt(s['q1'], 8)}{_fmt(s['q3'], 8)}"
            f"{_fmt(s['log10_sd'], 9, 3)}"
            f"{s['min']:>7.2f}-{s['max']:<8.1f}")

    s = stats["same publication (strict)"]
    sub_one = [p for p in strict if not pairing.is_physical_ratio(p["ratio"])]
    out()
    out(f"Median ratio for the defensible pairs: {s['median']:.2f} "
        f"(95% bootstrap CI {s['median_ci_low']:.2f} to {s['median_ci_high']:.2f}),")
    out(f"IQR {s['q1']:.2f} to {s['q3']:.2f}, n = {s['n']}.")
    out(f"{len(sub_one)} of {s['n']} pairs sit below 1, which the competitive model")
    out("forbids (" + ", ".join(f"{p['molecule_chembl_id']} {p['ratio']:.2f}"
                              for p in sub_one) + "). Those are not evidence")
    out("against the conversion -- they are evidence that a few paired records")
    out("are not what their labels claim. They stay in the headline number.")
    out(f"The full span is {s['fold_span']:.1f}-fold, which sounds alarming and is")
    out("the wrong number to quote: the ratio is not one number across papers,")
    out("and section (c) shows why.")

    # per-document breakdown
    spread = pairing.within_document_spread(strict)
    out()
    out("Per publication:")
    out(f"{'document':<16}{'n':>4}{'median ratio':>14}{'within-paper spread':>22}"
        f"  implied [S]/Km")
    for doc, d in sorted(spread["per_document"].items(),
                         key=lambda t: -t[1]["n"]):
        out(f"{doc:<16}{d['n']:>4}{_fmt(d['median_ratio'], 14)}"
            f"{_fmt(d['fold_spread'], 20, 2)}x"
            f"{pairing.implied_substrate_ratio(d['median_ratio']):>13.1f}")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(b) what Cheng-Prusoff implies, and what assumption reconciles it")
    out("-" * 78)
    out("For a competitive inhibitor, IC50 = Ki (1 + [S]/Km). Inverting the")
    out("observed ratio gives the assay's substrate factor directly:")
    out()
    out(f"{'observed ratio':>16}{'implied [S]/Km':>18}")
    for r in (1.0, 2.0, s["median"], 21.0, 30.0):
        out(f"{r:>16.2f}{pairing.implied_substrate_ratio(r):>18.2f}")
    out()
    out("Quoting one implied [S]/Km for the whole set would be misleading, and")
    out("section (a) is the reason. The pooled median of "
        f"{s['median']:.1f} is an average")
    out("over papers whose own median ratios run from "
        f"{min(d['median_ratio'] for d in spread['per_document'].values()):.1f} to "
        f"{max(d['median_ratio'] for d in spread['per_document'].values()):.1f},")
    out("i.e. implied [S]/Km from about "
        f"{min(pairing.implied_substrate_ratio(d['median_ratio']) for d in spread['per_document'].values()):.1f} "
        f"to {max(pairing.implied_substrate_ratio(d['median_ratio']) for d in spread['per_document'].values()):.0f}.")
    out("Quoted per paper the factor is a concrete number of the size")
    out("inhibition assays of this enzyme plausibly use; quoted across the")
    out("whole set it is an average of incompatible things and means little.")
    out()
    out("A single global [S]/Km CANNOT reconcile this data, and that is worth")
    out("stating plainly: with one factor, every paper's median ratio would")
    out("have to be the same number. They are not. What reconciles the data is")
    out("a per-assay factor -- which is exactly what Cheng-Prusoff says the")
    out("ratio is. So the median is reported because it summarises the set, not")
    out("because it is the conversion factor.")
    out()
    out("The assumptions that make that per-assay reading consistent: every")
    out("pair is a competitive inhibitor; each paper ran a single [S]; and each")
    out("IC50 is a true 50% inhibition point rather than a fitted midpoint off a")
    out("partial curve. Relax any of them and the ratio stops being an assay")
    out("constant.")
    out()
    out("The assumption is NOT independently checkable here. Nothing in these")
    out("files constrains [S], so no pair can confirm or refute the "
        "correction it")
    out("implies. That is the ceiling of this test.")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(c) assay conditions vs genuinely different compounds")
    out("-" * 78)
    out("Same-publication pairs are the tighter comparison: one lab, one")
    out("protocol family, one enzyme preparation, and typically one substrate")
    out("concentration. Same-molecule pairs that span publications mix all of")
    out("those, so their scatter has more in it than assay variation alone.")
    out()
    de = pairing.document_effect(strict)
    out(f"log10(ratio) variance decomposition, strict pairs (>=3 pairs/paper):")
    out(f"  total   {de['total_var']:.4f}")
    out(f"  between {de['between_var']:.4f}  ({100 * de['document_explains']:.1f}% "
        f"of total)")
    out(f"  within  {de['within_var']:.4f}  ({100 * (1 - de['document_explains']):.1f}%)")
    out(f"  {de['n_groups_used']} papers, {de['n_pairs_used']} pairs")
    out()
    out(f"Between-paper differences are a "
        f"{10 ** (np.sqrt(de['between_var'])):.1f}-fold effect on the ratio;")
    out(f"within-paper scatter is "
        f"{10 ** np.sqrt(de['within_var']):.1f}-fold. So roughly half the total")
    out("variance in the ratio is 'which paper this came from', i.e. assay")
    out("conditions, not 'which compound this is'. That is the pattern")
    out("Cheng-Prusoff predicts. It is also what any per-assay systematic")
    out("difference would produce, so it discriminates the conversion from")
    out("noise, not from a rival theory.")

    # falsification: does the ratio track potency?
    out()
    out("Rejecting the alternative: if the ratio were really a per-compound")
    out("correction rather than a per-assay constant, it would track potency")
    out("within a paper. Fitting log10(ratio) on log10(IC50) per document:")
    out()
    slopes = pairing.ratio_vs_potency_slope(strict)
    big = sorted(slopes.items(), key=lambda t: -t[1]["n"])
    out("Per paper (>=5 pairs), with the IC50 range each slope was fitted over:")
    for doc, d in big:
        kind = ("whole-cell" if any(p["document_chembl_id"] == doc and
                                    p["ki_assay_class"] == "non_enzyme"
                                    for p in strict) else "enzyme")
        out(f"  {doc:<16} n={d['n']:>2}  slope={d['slope']:+.3f}  "
            f"r={d['pearson_r']:+.2f}  {d['ic50_min_nM']:>8.3g}-{d['ic50_max_nM']:<9.3g}"
            f" [{kind}]")
    flat = [(d, v) for d, v in big if abs(v["slope"]) < 0.2]
    out()
    out(f"{len(flat)} of {len(big)} papers with >=5 pairs have |slope| < 0.2 while")
    out("spanning 2-3 orders of magnitude of potency: the ratio does not track")
    out("potency, so it is not a per-compound correction. The exception is the")
    out("whole-cell paper (CHEMBL1124568), whose 'Ki' is antibacterial potency")
    out("rather than a binding constant -- pairing it with a purified-enzyme")
    out("IC50 compares two different quantities, which is why the enzyme-only")
    out("column exists.")
    out()
    enz_s = stats["strict, enzyme assays only"]
    out(f"Restricting to enzyme-assay Ki records ({enz_s['n']} pairs) moves the")
    out(f"median to {enz_s['median']:.2f} and the log10 SD to "
        f"{enz_s['log10_sd']:.3f}. Both are reported;")
    out("the cell-based rows are not deleted, because dropping data to improve")
    out("a statistic is how a validation becomes a demonstration.")

    # ------------------------------------------------------------------
    out()
    out("-" * 78)
    out("(d) verdict")
    out("-" * 78)
    label, reasons = pairing.verdict(strict)
    out(f"VERDICT: {label}")
    for r in reasons:
        out(f"  - {r}")
    out()
    out("In plain terms: the ratio is reproducible within a paper and differs")
    out("between papers, which is the structure Cheng-Prusoff requires. That is")
    out("real support for the conversion's *form*. It is not validation of the")
    out("conversion's *values*: the file that would settle those -- one carrying")
    out("[S] alongside IC50 and Ki -- is not in this repository.")
    out()
    out(f"Nor is the support unqualified. {len(sub_one)} pairs fall below the")
    out("model's floor of ratio 1, and the pooled within-paper spread is")
    out(f"{(10 ** spread['pooled_log10_sd']):.1f}-fold even in the best paper, so the ratio")
    out("is a per-assay constant with real noise on it rather than a tight one.")
    out("Both facts are in the tables above instead of being smoothed away: a")
    out("validation that hides its own residuals is a demonstration.")
    out()
    out("What is missing, stated exactly:")
    for m in MISSING:
        out(f"  - {m}")
    out()
    out("With [S] and the assay Km, `pairing.assay_residuals` computes the")
    out("predicted ratio and the per-compound log residual directly; that is")
    out("the check this data cannot run and a suitable paper could.")

    # ------------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    pair_path = RESULTS / "cheng_prusoff_pairs.csv"
    with pair_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["join_mode", "molecule_chembl_id", "document_chembl_id",
                    "ic50_nM", "ki_nM", "ic50_over_ki", "implied_S_over_Km",
                    "physical", "n_ic50_records", "n_ki_records",
                    "ki_assay_class", "ki_assay_description"])
        for mode, ps in (("same_document", strict), ("same_molecule", loose)):
            for p in ps:
                w.writerow([mode, p["molecule_chembl_id"],
                            p["document_chembl_id"] or "",
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
                    "median_ci_low", "median_ci_high",
                    "within_paper_log10_sd", "document_explains",
                    "implied_S_over_Km_at_median", "verdict"])
        for label, ps in (("same_document_all", strict),
                          ("same_molecule_all", loose),
                          ("same_document_enzyme_only", enzyme)):
            st = pairing.summarize_ratios([p["ratio"] for p in ps])
            sp = pairing.within_document_spread(ps)
            dd = pairing.document_effect(ps)
            vb = pairing.verdict(ps)[0]
            w.writerow([label, st["n"], f"{st['median']:.6g}",
                        f"{st['q1']:.6g}", f"{st['q3']:.6g}", f"{st['iqr']:.6g}",
                        f"{st['log10_sd']:.6g}", f"{st['min']:.6g}",
                        f"{st['max']:.6g}", f"{st['fold_span']:.6g}",
                        f"{st['median_ci_low']:.6g}",
                        f"{st['median_ci_high']:.6g}",
                        f"{sp['pooled_log10_sd']:.6g}",
                        f"{dd['document_explains']:.6g}",
                        f"{pairing.implied_substrate_ratio(st['median']):.6g}",
                        vb])

    out()
    out(f"wrote {pair_path.relative_to(ROOT)}  ({len(strict) + len(loose)} rows)")
    out(f"wrote {stat_path.relative_to(ROOT)}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
