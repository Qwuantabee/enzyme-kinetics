"""Does the QSAR signal survive inside a single chemotype?

    python scripts/qsar_single_scaffold.py [--n-perm 1000]

`qsar_congeneric.py` reports that the 72- and 73-compound ChEMBL series beat
a permutation null, and it also reports that those series are not one
chemotype: a 73-compound set spanning 6 Murcko scaffolds and a 72-compound set
spanning 9. Those two facts can coexist for a boring reason. If the scaffolds
differ systematically in size -- a phenyl series being small and an
adamantyl series being large -- then molecular weight ranks the chemotypes,
and a descriptor model "predicts" Ki by recognising which paper sub-series a
compound came from rather than by seeing any local chemistry. The
cross-validated R2 would be honest about the shuffled labels and still say
nothing causal about binding.

The test is the obvious one that nobody runs: throw away every compound
outside the largest single scaffold and redo exactly the same analysis on what
is left. Same descriptors, same leave-one-out folds, same permutation null.
If the signal is chemotype spread, it disappears when the chemotypes do. If
it survives, the variation being fitted is variation between isomers and
homologues sharing a ring system, which is what a QSAR is supposed to explain.

A subset below 20 compounds is reported as too small and not fitted. That
threshold is not shyness: with ten descriptors plus an intercept, an LOO R2
on 19 points is a number chance reaches routinely, and printing it invites
exactly the misreading this script exists to prevent.

Output: printed report plus results_single_scaffold.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.qsar_congeneric import (  # noqa: E402
    SERIES_LABELS,
    read_smiles,
    series_rows,
)
sys.path.insert(0, str(ROOT / "src"))

from enzkin import qsar  # noqa: E402
from enzkin.datasets import load_chembl_dhfr_ki  # noqa: E402

#: Below this many compounds in the single-scaffold subset, the analysis is
#: not run. Ten descriptors plus an intercept against fewer than 20 rows is
#: a parameter-rich fit whose LOO R2 cannot be separated from chance.
MIN_SUBSET = 20

#: The two large series from the QSAR report.
SERIES = ("CHEMBL1129627", "CHEMBL1124156")


def scaffold_of(smiles):
    """Murcko scaffold SMILES for one compound, or None if unparseable."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = qsar.parse_smiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_assignments(rows):
    """Murcko scaffold per row, in row order, for a series' compound rows."""
    out = []
    for _cid, smi, _desc in rows:
        out.append(scaffold_of(smi))
    return out


def largest_scaffold(scaffolds):
    """(scaffold SMILES, size) of the biggest chemotype in a list."""
    from collections import Counter
    counts = Counter(s for s in scaffolds if s is not None)
    if not counts:
        return None, 0
    return counts.most_common(1)[0]


def subset(rows, ki_nM, scaffolds, scaffold):
    """Rows/affinities/scaffolds restricted to one scaffold, order preserved."""
    keep = [i for i, s in enumerate(scaffolds) if s == scaffold]
    kept_rows = [rows[i] for i in keep]
    kept_ki = np.asarray([ki_nM[i] for i in keep], dtype=float)
    return kept_rows, kept_ki


def analyse_subset(rows, ki_nM, n_perm=1000, seed=0):
    """The same LOO + permutation analysis as the main script, on a subset."""
    if len(rows) < MIN_SUBSET:
        return {
            "n": len(rows),
            "too_small": True,
            "min_subset": MIN_SUBSET,
        }
    X, names = qsar.descriptor_matrix(rows)
    y = np.log10(ki_nM)
    loo = qsar.loocv_predictions(X, y)
    r2 = qsar.r2_score(y, loo)
    null = qsar.permutation_null(X, y, r2, n_perm=n_perm, seed=seed)
    par = qsar.parsimony(X, names, y)
    return {
        "n": len(rows),
        "too_small": False,
        "X": X,
        "names": names,
        "y": y,
        "loocv_r2": float(r2),
        "loocv_pred": loo,
        "null": null,
        "parsimony": par,
        "univariate": qsar.univariate_scan(X, names, y),
        "ki_nM": ki_nM,
        "constant_descriptors": [n for j, n in enumerate(names)
                                 if np.std(X[:, j]) == 0.0],
    }


def descriptor_spread(rows, scaffolds):
    """How much of each descriptor's variance is between chemotypes.

    The confounding mechanism put in numbers: the fraction of total variance
    of each descriptor explained by scaffold identity (eta-squared from a
    one-way ANOVA-style decomposition). High values mean the descriptor is
    mostly a chemotype label.
    """
    X, names = qsar.descriptor_matrix(rows)
    labels = np.asarray([s if s is not None else "" for s in scaffolds])
    out = []
    for j, name in enumerate(names):
        col = X[:, j]
        ss_tot = float(((col - col.mean()) ** 2).sum())
        ss_between = 0.0
        for lab in set(labels.tolist()):
            grp = col[labels == lab]
            ss_between += grp.size * (grp.mean() - col.mean()) ** 2
        eta2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        out.append({"descriptor": name, "eta_squared": float(eta2)})
    out.sort(key=lambda d: -d["eta_squared"])
    return out


def verdict(a, label):
    """Plain-language reading, with the chemotype answer stated either way."""
    if a["too_small"]:
        return (
            f"{label}: only n = {a['n']} compounds on this scaffold, below "
            f"the {a['min_subset']} needed for the ten-descriptor fit; the "
            f"subset is too small to test and no R2 is reported"
        )
    p = a["null"]["p_perm"]
    r2 = a["loocv_r2"]
    par = a["parsimony"]
    if r2 <= 0:
        return (
            f"{label}: the signal does NOT survive -- within one scaffold "
            f"LOO R2 = {r2:.3f} on n = {a['n']}, no better than predicting "
            f"the mean (p_perm = {p:.3f}). That is the absence of a "
            f"within-scaffold fit at this n. It is compatible with the "
            f"cross-scaffold correlation reflecting differences between "
            f"chemotypes, but a negative R2 cannot establish that: a weak "
            f"real signal on a small n gives the same number"
        )
    if p < 0.05:
        if par["improvement"] <= 0.02:
            tail = (
                f"; note that {par['best_descriptor']} alone reaches "
                f"{par['best_single_r2']:.3f}, so the within-scaffold fit is "
                f"a size effect and the other nine descriptors add nothing"
            )
        else:
            tail = (
                f"; best single descriptor {par['best_descriptor']} reaches "
                f"{par['best_single_r2']:.3f} "
                f"(gain {par['improvement']:+.3f})"
            )
        return (
            f"{label}: the signal survives -- within one scaffold LOO "
            f"R2 = {r2:.3f} on n = {a['n']}, p_perm = {p:.3f}{tail}"
        )
    return (
        f"{label}: too small to distinguish -- within one scaffold LOO "
        f"R2 = {r2:.3f} on n = {a['n']} is reached by shuffled Ki in "
        f"{p * 100:.1f}% of {a['null']['n_perm']} permutations "
        f"(p = {p:.3f}); the subset is too small to separate the fit from "
        f"chance and the cross-scaffold signal cannot be credited to local "
        f"chemistry on this evidence"
    )


def format_report(records):
    L = []
    add = L.append
    add("Single-scaffold (chemotype-controlled) QSAR test")
    add("=" * 60)
    add("")
    add("Same descriptors, same leave-one-out folds, same permutation null,")
    add("restricted to the largest Murcko scaffold of each series. If the")
    add("cross-series signal is chemotype spread, it vanishes here.")
    add("")
    for r in records:
        s = r["series"]
        add(f"--- {s['document_id']}: {SERIES_LABELS.get(s['document_id'], '')}")
        add(f"all compounds fitted: n = {s['n_total']}, "
            f"scaffolds = {s['n_scaffolds']}")
        add(f"largest scaffold: {s['largest']}")
        add(f"scaffold size: {s['largest_size']}/{s['n_total']} compounds")
        add("")
        add("Between-scaffold share of each descriptor's variance "
            "(eta-squared):")
        for d in r["spread"][:5]:
            add(f"  {d['descriptor']:<16}{d['eta_squared']:>7.3f}")
        add("")
        a = r["subset"]
        if a["too_small"]:
            add(f"SUBSET n = {a['n']} -- below {a['min_subset']}; "
                f"not fitted (see verdict)")
            add("")
            add("VERDICT: " + verdict(a, s["document_id"]))
            add("")
            continue
        add(f"SUBSET n = {a['n']} (constant descriptors: "
            f"{', '.join(a['constant_descriptors']) or 'none'})")
        add(f"Ki span in subset: {a['ki_nM'].min():g} to "
            f"{a['ki_nM'].max():g} nM "
            f"({a['ki_nM'].max() / a['ki_nM'].min():.0f}x)")
        add(f"LOO R2 = {a['loocv_r2']:.3f}; best single descriptor "
            f"{a['parsimony']['best_descriptor']} gives "
            f"{a['parsimony']['best_single_r2']:.3f} "
            f"(gain {a['parsimony']['improvement']:+.3f})")
        nl = a["null"]
        add(f"Permutation null ({nl['n_perm']} shuffles, seed {nl['seed']}): "
            f"mean = {nl['null_mean']:.3f}, 95th pct = {nl['null_p95']:.3f}, "
            f"max = {nl['null_max']:.3f}")
        add(f"p_perm = {nl['p_perm']:.3f}")
        add("")
        add("VERDICT: " + verdict(a, s["document_id"]))
        add("")
    return "\n".join(L)


def run(series=SERIES, n_perm=1000, seed=0, ki_rows=None, smiles=None):
    """Whole analysis for the given series; returns (records, report text)."""
    ki_rows = load_chembl_dhfr_ki() if ki_rows is None else ki_rows
    smiles = read_smiles() if smiles is None else smiles
    records = []
    for doc in series:
        s = series_rows(doc, ki_rows=ki_rows, smiles=smiles)
        sc = scaffold_assignments(s["rows"])
        largest, size = largest_scaffold(sc)
        from collections import Counter
        n_scaffolds = len({x for x in sc if x is not None})
        sub_rows, sub_ki = subset(s["rows"], s["ki_nM"], sc, largest)
        records.append({
            "series": {
                "document_id": doc,
                "n_total": len(s["rows"]),
                "n_scaffolds": n_scaffolds,
                "scaffold_counts": Counter(x for x in sc if x is not None),
                "largest": largest,
                "largest_size": size,
            },
            "spread": descriptor_spread(s["rows"], sc),
            "subset": analyse_subset(sub_rows, sub_ki, n_perm=n_perm,
                                     seed=seed),
        })
    return records, format_report(records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="+", default=list(SERIES))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results_single_scaffold.txt"))
    args = ap.parse_args(argv)
    _records, report = run(series=args.series, n_perm=args.n_perm,
                           seed=args.seed)
    Path(args.out).write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
