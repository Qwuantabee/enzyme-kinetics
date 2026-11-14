"""Is the structure-affinity signal carried by the weakest binders?

    python scripts/qsar_tail_sweep.py [--n-perm 1000]

The single cutoff in results_qsar.txt (drop Ki >= 100 uM, which removes only
1-2 compounds per series) is the mildest test of the tail. A medicinal
chemist would not care about compounds beyond roughly 100 nM, and many of the
weak compounds sit at the far end of every size descriptor, so this sweeps the
cutoff downward and reports what the leave-one-out R2 does, on the whole
series and on the largest Murcko scaffold, with the same permutation null.

Nothing is tuned: the cutoffs (1e5, 1e4, 1e3, 1e2 nM) are fixed here, every
one is reported, and a cutoff that leaves fewer than 10 compounds is reported
as not applicable rather than skipped.

Output: results_qsar_tail.txt, results/qsar_tail_sweep.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.qsar_congeneric import SERIES_LABELS, read_smiles, series_rows  # noqa: E402
from scripts.qsar_single_scaffold import (  # noqa: E402
    largest_scaffold, scaffold_assignments, subset)
from enzkin import qsar  # noqa: E402
from enzkin.datasets import load_chembl_dhfr_ki  # noqa: E402

SERIES = ("CHEMBL1129627", "CHEMBL1124156")
CUTOFFS_NM = (1e5, 1e4, 1e3, 1e2)


def affinity_bands(ki_nM):
    """Counts of compounds by Ki band, so 'the rest weaker' is a number."""
    k = np.asarray(ki_nM, dtype=float)
    return {"below_1nM": int((k < 1).sum()),
            "1_to_100nM": int(((k >= 1) & (k < 100)).sum()),
            "100nM_to_100uM": int(((k >= 100) & (k < 1e5)).sum()),
            "100uM_and_above": int((k >= 1e5).sum())}


def sweep(rows, ki_nM, n_perm, seed=0):
    X, _names = qsar.descriptor_matrix(rows)
    y = np.log10(ki_nM)
    r2 = qsar.loocv_r2(X, y)
    out = [{"cutoff_nM": None, "applicable": True, "n_kept": len(rows),
            "n_dropped": 0, "loocv_r2": float(r2),
            "best_single_r2": float(max(qsar.loocv_r2(X[:, [j]], y)
                                        for j in range(X.shape[1])))}]
    for c in CUTOFFS_NM:
        out.append(qsar.truncation_sensitivity(X, y, ki_nM, cutoff_nM=c,
                                               n_perm=n_perm, seed=seed))
    return out


def run(n_perm=1000, ki_rows=None, smiles=None):
    ki_rows = load_chembl_dhfr_ki() if ki_rows is None else ki_rows
    smiles = read_smiles() if smiles is None else smiles
    L, recs = [], []
    add = L.append
    add("Tail sensitivity of the QSAR signal (leave-one-out R2)")
    add("=" * 60)
    add("")
    add("Compounds are dropped from the weak end (Ki >= cutoff) and the same")
    add("ten-descriptor LOO fit and permutation null are rerun. Cutoffs are")
    add("fixed in advance and all are shown.")
    for doc in SERIES:
        s = series_rows(doc, ki_rows=ki_rows, smiles=smiles)
        sc = scaffold_assignments(s["rows"])
        largest, size = largest_scaffold(sc)
        sub_rows, sub_ki = subset(s["rows"], s["ki_nM"], sc, largest)
        for label, rows, ki in (("whole series", s["rows"], s["ki_nM"]),
                                ("largest scaffold", sub_rows, sub_ki)):
            b = affinity_bands(ki)
            add("")
            add(f"--- {doc} ({SERIES_LABELS.get(doc, '')}) -- {label}, n = {len(rows)}")
            add("Ki bands: " + ", ".join(f"{k} {v}" for k, v in b.items()))
            add(f"{'cutoff (nM)':>12}{'kept':>6}{'dropped':>9}{'LOO R2':>9}"
                f"{'best single':>13}{'p_perm':>9}")
            for r in sweep(rows, ki, n_perm):
                c = "none" if r["cutoff_nM"] is None else f"{r['cutoff_nM']:g}"
                if not r["applicable"]:
                    add(f"{c:>12}{r['n_kept']:>6}{r['n_dropped']:>9}"
                        f"{'n/a: fewer than 10 left or none dropped':>40}")
                else:
                    p = r.get("p_perm")
                    add(f"{c:>12}{r['n_kept']:>6}{r['n_dropped']:>9}"
                        f"{r['loocv_r2']:>9.3f}{r['best_single_r2']:>13.3f}"
                        f"{('' if p is None else format(p, '.3f')):>9}")
                recs.append({"document": doc, "subset": label,
                             "cutoff_nM": "" if r["cutoff_nM"] is None else r["cutoff_nM"],
                             "applicable": r["applicable"], "n_kept": r["n_kept"],
                             "n_dropped": r["n_dropped"],
                             "loocv_r2": r.get("loocv_r2", ""),
                             "best_single_r2": r.get("best_single_r2", ""),
                             "p_perm": r.get("p_perm", "")})
    return "\n".join(L), recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args(argv)
    report, recs = run(n_perm=args.n_perm)
    (ROOT / "results_qsar_tail.txt").write_text(report + "\n")
    with (ROOT / "results" / "qsar_tail_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
