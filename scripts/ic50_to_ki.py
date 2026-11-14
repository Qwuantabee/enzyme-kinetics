"""Apply Cheng-Prusoff to the real ChEMBL DHFR IC50 dataset.

    python scripts/ic50_to_ki.py

Answers a question the dataset can actually support: given 482 measured IC50s
for E. coli dihydrofolate reductase, what does the conversion do to them, and
where does it stop being valid?

It cannot answer "is the converted Ki right", because ChEMBL's records for this
target do not carry the assay's [S] or [E]. That is stated in the output rather
than glossed over. What it can do is show the size of the correction as a
function of [S]/Km, and show that the tight-binding failure case is common
enough to matter rather than hypothetical.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import cheng_prusoff as cp  # noqa: E402
from enzkin import datasets  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
OUT = ROOT / "results_ic50.txt"


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    table = datasets.load_chembl_dhfr_ic50()
    ics = np.array([r["ic50_nM"] for r in table])

    out("=" * 74)
    out("Cheng-Prusoff conversion, real E. coli DHFR IC50 data")
    out("=" * 74)
    out(f"source     ChEMBL target {datasets.CHEMBL_DHFR_TARGET}, "
        f"standard_type=IC50, relation '='")
    out(f"records    {len(table)} values, "
        f"{len({r['molecule_chembl_id'] for r in table})} compounds, "
        f"{len({r['document_chembl_id'] for r in table})} publications")
    out(f"range      {ics.min():.3g} to {ics.max():.3g} nM")
    out(f"median     {np.median(ics):.3g} nM")

    # --- how big is the correction, and what does it do --------------------
    out("\n--- correction factor vs [S]/Km (competitive) ---")
    out(f"{'[S]/Km':>7} {'factor':>8} {'Ki from median IC50':>22}")
    med = float(np.median(ics))
    for ratio in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        f = cp.correction_factor(ratio, 1.0, "competitive")  # ratio only
        out(f"{ratio:7.1f} {f:8.2f} {med / f:22.1f} nM")

    out("\n--- the same IC50 under three mechanisms ([S] = 2 Km) ---")
    km, s = 25e3, 50e3   # nM
    for model in cp.SINGLE_KI_MODELS:
        ki = cp.ki_from_ic50(med, s, km, model=model)
        out(f"{model:16} Ki = {ki:10.1f} nM")

    # --- convert the dataset and describe the result ----------------------
    out("\n--- converted dataset (competitive, assuming [S] = 2 Km) ---")
    kis = np.array([cp.ki_from_ic50(r["ic50_nM"], s, km) for r in table])
    out(f"Ki range   {kis.min():.3g} to {kis.max():.3g} nM")
    out(f"Ki median  {np.median(kis):.3g} nM")
    out("The conversion divides every value by the same factor, so it shifts")
    out("the distribution without changing its shape or the rank order. That")
    out("is the honest summary: Cheng-Prusoff rescales a series, it does not")
    out("reorganise it.")

    # --- where would it fail ---------------------------------------------
    out("\n--- tight-binding risk across the series ---")
    out("For each assumed enzyme concentration, how many of these compounds")
    out("fall below the 10x threshold where Cheng-Prusoff stops being valid?")
    out(f"{'[E] nM':>9} {'IC50 < 10[E]':>13} {'% of series':>12}")
    for e_nm in (0.1, 1.0, 10.0, 100.0):
        thr = cp.tight_binding_threshold(e_nm)
        n = int(np.sum(ics < thr))
        out(f"{e_nm:9.1f} {n:13d} {100 * n / len(ics):11.1f}%")

    out("\n--- assumption check on the median compound ---")
    rep = cp.check_assumptions(med, s, km, model="competitive",
                               enzyme_conc=1.0)
    out(rep.summary())

    out("\n--- what this does NOT establish ---")
    out("ChEMBL's activity records for this target carry no [S] and no [E].")
    out("Every Ki above is therefore illustrative of the arithmetic, not a")
    out("measurement. To validate the conversion against measured Ki you need")
    out("a paper reporting both for the same compound under stated conditions.")
    out("That is the open item in README.md.")

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ratios = np.logspace(-1, 2, 200)
    ax.plot(ratios, [cp.correction_factor(r, 1.0, "competitive")
                     for r in ratios], lw=2, label="competitive")
    ax.plot(ratios, [cp.correction_factor(r, 1.0, "uncompetitive")
                     for r in ratios], lw=2, label="uncompetitive")
    ax.axhline(1.0, ls=":", color="grey", lw=1, label="noncompetitive")
    ax.axvline(1.0, ls="--", color="k", lw=0.8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("[S] / Km")
    ax.set_ylabel("IC50 / Ki")
    ax.set_title("correction factor: the size of the conversion", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.hist(np.log10(ics), bins=40, color="C0", alpha=0.8)
    ax.set_xlabel("log10 IC50 (nM)")
    ax.set_ylabel("compounds")
    ax.set_title(f"E. coli DHFR IC50 series (n={len(ics)})", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "ic50_conversion.png", dpi=150)
    out(f"\nfigure -> {FIG / 'ic50_conversion.png'}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"summary -> {OUT}")


if __name__ == "__main__":
    main()
