"""Does structure predict affinity in a congeneric DHFR inhibitor series?

    python scripts/qsar_congeneric.py [--series CHEMBL1123078] [--n-perm 1000]

Computes ten RDKit 2D descriptors for every compound in a ChEMBL paper series,
fits them against measured Ki with leave-one-out cross-validation, and then
tries to destroy the result by shuffling the affinities.

Why the shuffle is the point. A 21-compound series with ten descriptors and
an intercept gives eleven parameters for 21 observations. In that regime a
leave-one-out R2 of 0.5 is not evidence of anything on its own -- noise, run
through the same folds, produces numbers of that size routinely. Reporting
the R2 without its permutation null would be the kind of result that looks
like a finding and evaporates on a larger set, so this script reports both
and refuses to call the second one a formality.

Output goes to results_qsar.txt and results/qsar_series.csv, every number
computed here, none transcribed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import qsar  # noqa: E402
from enzkin.datasets import load_chembl_dhfr_ki  # noqa: E402

DATA_DIR = ROOT / "data" / "chembl_dhfr"
SMILES_CSV = DATA_DIR / "ecoli_dhfr_smiles.csv"

#: Paper (document_chembl_id) -> label for the report.
SERIES_LABELS = {
    "CHEMBL1123078": "21 compounds, Ki 0.021-16 nM",
    "CHEMBL1129627": "73 compounds",
    "CHEMBL1124156": "72 compounds",
}


def read_smiles(path=None):
    path = Path(path) if path is not None else SMILES_CSV
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} -- run "
            "`python scripts/fetch_chembl_smiles.py` first"
        )
    with path.open(newline="") as fh:
        return {r["molecule_chembl_id"]: (r["canonical_smiles"] or None)
                for r in csv.DictReader(fh)}


def series_rows(document_id, ki_rows=None, smiles=None, wild_type_only=True):
    """Compounds in one paper with both a Ki and a parseable structure.

    One row per compound: if a compound appears in the paper more than once
    the geometric mean Ki is taken, which is the natural average for
    affinities (they are log-scaled quantities). The count of such
    duplicates is returned so the report can say how many there were.
    """
    ki_rows = load_chembl_dhfr_ki() if ki_rows is None else ki_rows
    smiles = read_smiles() if smiles is None else smiles
    by_id = {}
    for r in ki_rows:
        if r["document_chembl_id"] != document_id:
            continue
        if wild_type_only and r["assay_description"]:
            pass  # load_chembl_dhfr_ki already applied the mutant filter
        by_id.setdefault(r["molecule_chembl_id"], []).append(r["ki_nM"])
    dupes = sum(1 for v in by_id.values() if len(v) > 1)
    wanted = {cid: float(np.exp(np.log(v).mean())) for cid, v in by_id.items()}
    matched = {cid: smiles[cid] for cid in wanted
               if smiles.get(cid)}
    rows, skipped = qsar.descriptor_table(matched)
    ki = []
    for cid, _smi, _desc in rows:
        ki.append(wanted[cid])
    return {
        "document_id": document_id,
        "rows": rows,
        "ki_nM": np.array(ki, dtype=float),
        "n_compounds_with_ki": len(wanted),
        "n_duplicate_compounds": dupes,
        "n_missing_structure": len(wanted) - len(matched),
        "skipped_unparseable": skipped,
    }


def analyse(document_id, n_perm=1000, seed=0, ki_rows=None, smiles=None):
    """Run the whole analysis for one paper and return a plain dict."""
    s = series_rows(document_id, ki_rows=ki_rows, smiles=smiles)
    if len(s["rows"]) < 5:
        raise ValueError(
            f"{document_id}: only {len(s['rows'])} compounds with both a Ki "
            "and a structure; nothing can be fitted"
        )
    X, names = qsar.descriptor_matrix(s["rows"])
    y = np.log10(s["ki_nM"])          # affinities are handled on log scale
    loo = qsar.loocv_predictions(X, y)
    r2 = qsar.r2_score(y, loo)
    null = qsar.permutation_null(X, y, r2, n_perm=n_perm, seed=seed)
    return {
        "series": s,
        "X": X,
        "names": names,
        "y": y,
        "n": len(y),
        "n_descriptors": len(names),
        "loocv_r2": float(r2),
        "loocv_pred": loo,
        "null": null,
        "univariate": qsar.univariate_scan(X, names, y),
        "collinearity": qsar.collinearity(X, names),
        "pearson_in_sample": float(np.corrcoef(y, loo)[0, 1]),
    }


def verdict(a):
    """Plain-language reading of the analysis, driven by p_perm.

    The thresholds are stated rather than hidden: p < 0.05 to be called
    support, and the R2 has to be positive in the first place, because a
    permutation p on a negative R2 means "chance also produces negative
    numbers", not "the model works".
    """
    p = a["null"]["p_perm"]
    r2 = a["loocv_r2"]
    n = a["n"]
    if r2 <= 0:
        return (
            f"no support: cross-validated R2 = {r2:.3f} is worse than "
            f"predicting the mean, so the descriptors do not predict Ki at "
            f"all (permutation p = {p:.3f})"
        )
    if p < 0.05:
        return (
            f"consistent with a structure-affinity relationship: "
            f"LOO R2 = {r2:.3f}, permutation p = {p:.3f} over "
            f"{a['null']['n_perm']} shuffles with n = {n}"
        )
    return (
        f"too small to support a descriptor model: LOO R2 = {r2:.3f} looks "
        f"promising but shuffled Ki reaches it in {p * 100:.1f}% of "
        f"{a['null']['n_perm']} permutations (p = {p:.3f}), so with n = {n} "
        f"that fit is indistinguishable from chance"
    )


def format_report(results, lines_out=None):
    L = lines_out if lines_out is not None else []
    add = L.append
    add("Structure-affinity analysis of congeneric E. coli DHFR inhibitors")
    add("=" * 70)
    add("")
    add("Descriptors: RDKit 2D only (no conformers, no 3D pose). Ki modelled")
    add("as log10(nM); one row per compound, geometric mean if repeated.")
    add("")
    for a in results:
        s = a["series"]
        add(f"--- {s['document_id']}: {SERIES_LABELS.get(s['document_id'], '')}")
        add(f"compounds with a Ki: {s['n_compounds_with_ki']}"
            f"  (repeated in the paper: {s['n_duplicate_compounds']})")
        add(f"missing structure:   {s['n_missing_structure']}"
            f"  unparseable SMILES: {len(s['skipped_unparseable'])}")
        add(f"fitted:              n = {a['n']}, "
            f"{a['n_descriptors']} descriptors + intercept = "
            f"{a['n_descriptors'] + 1} parameters")
        ratio = a["n"] / (a["n_descriptors"] + 1)
        add(f"observations per parameter: {ratio:.2f}"
            + ("   <-- below the ~5 usually wanted for a stable fit"
               if ratio < 5 else ""))
        add(f"Ki span: {s['ki_nM'].min():g} to {s['ki_nM'].max():g} nM "
            f"({s['ki_nM'].max() / s['ki_nM'].min():.0f}x)")
        add("")
        add("Univariate Spearman against log10 Ki (sorted by |rho|):")
        add(f"{'descriptor':<16}{'rho':>8}{'p':>10}")
        for d in a["univariate"]:
            add(f"{d['descriptor']:<16}{d['rho']:>8.3f}{d['p_value']:>10.4f}")
        add("")
        c = a["collinearity"]
        add(f"Most collinear descriptor pair: {c['pair'][0]} vs {c['pair'][1]}, "
            f"|rho| = {c['abs_r']:.3f}")
        add("")
        add(f"Multivariable model, leave-one-out: R2 = {a['loocv_r2']:.3f}, "
            f"corr(observed, LOO-predicted) = {a['pearson_in_sample']:.3f}")
        nl = a["null"]
        add(f"Permutation null ({nl['n_perm']} shuffles, seed {nl['seed']}): "
            f"mean R2 = {nl['null_mean']:.3f}, "
            f"95th pct = {nl['null_p95']:.3f}, max = {nl['null_max']:.3f}")
        add(f"p_perm = {nl['p_perm']:.3f}")
        add("")
        add("VERDICT: " + verdict(a))
        add("")
    return "\n".join(L)


def write_predictions(results, path):
    """Per-compound observed vs LOO-predicted log10 Ki, for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["document_chembl_id", "molecule_chembl_id",
                    "ki_nM", "log10_ki", "log10_ki_loocv_pred", "residual"])
        for a in results:
            for i, (cid, smi, desc) in enumerate(a["series"]["rows"]):
                w.writerow([a["series"]["document_id"], cid,
                            f"{a['series']['ki_nM'][i]:g}",
                            f"{a['y'][i]:.6f}",
                            f"{a['loocv_pred'][i]:.6f}",
                            f"{a['y'][i] - a['loocv_pred'][i]:.6f}"])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="+", default=["CHEMBL1123078"],
                    help="document_chembl_id(s) to analyse")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results_qsar.txt"))
    args = ap.parse_args(argv)

    ki_rows = load_chembl_dhfr_ki()
    smiles = read_smiles()
    results = [analyse(d, n_perm=args.n_perm, seed=args.seed,
                       ki_rows=ki_rows, smiles=smiles)
               for d in args.series]
    report = format_report(results)
    Path(args.out).write_text(report + "\n")
    write_predictions(results, ROOT / "results" / "qsar_series.csv")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
