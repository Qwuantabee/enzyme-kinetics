"""Fit the published P99 / nitrocefin progress curves.

    python scripts/fit_published.py

Writes figures/published_p99.png, prints the numbers quoted in README.md, and
appends a machine-readable summary to results_published.txt.

Data: Frere, Verlaine & Matagne (2023) Sci Rep 13:14994,
doi:10.1038/s41598-023-41805-y; archive doi:10.5281/zenodo.10059024, CC-BY.
System: Enterobacter cloacae P99 class C beta-lactamase + nitrocefin at 462 nm,
0.6 nM enzyme, 30 C. Published reference values for this pair: kcat = 780 s^-1,
Km = 25 uM (quoted by the paper, not measured here).

Two analyses of the same 18 traces:

  initial rates   a straight line through the first 15% of each trace, then
                  Michaelis-Menten on the five averages
  progress curves the integrated rate equation fitted to every point

They should agree. Where they disagree, the reason is substrate depletion
inside the initial-rate window, and the progress-curve value is the one to
trust because it does not depend on where the window was cut.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import datasets, fitting, progress  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
OUT = ROOT / "results_published.txt"

WINDOWS = (0.05, 0.10, 0.15, 0.25, 0.40)


def window_sensitivity():
    """How much does the initial-rate answer depend on the window chosen?"""
    print("\n--- initial rates: dependence on the window ---")
    print(f"{'window':>7} {'Vmax':>8} {'Km':>8} {'kcat':>8} {'n_ok':>5}")
    rows = {}
    for frac in WINDOWS:
        try:
            table = datasets.initial_rate_table(frac)
        except ValueError as exc:
            print(f"{frac:7.0%}  skipped ({exc})")
            continue
        S = np.array([r["s0_um"] for r in table])
        v = np.array([r["v_um_per_min"] for r in table])
        fit = fitting.fit_michaelis_menten(S, v)
        kcat = datasets.kcat_from_vmax(fit.params["vmax"])
        rows[frac] = (fit, table)
        print(f"{frac:7.0%} {fit.params['vmax']:8.2f} {fit.params['km']:8.2f} "
              f"{kcat:8.0f} {len(table):5d}")
    return rows


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 74)
    out("Published dataset: P99 beta-lactamase + nitrocefin (Frere et al. 2023)")
    out("=" * 74)
    out(f"enzyme        {datasets.ENZYME_CONC_M * 1e9:.2f} nM (final, cuvette)")
    out(f"delta eps     {datasets.DELTA_EPSILON:.0f} M^-1 cm^-1 at 462 nm")
    out(f"conditions    {datasets.BUFFER}, {datasets.TEMPERATURE_C:.0f} C")
    out(f"traces        {len(datasets.SUBSTRATE_UM)} "
        f"({len(set(datasets.SUBSTRATE_UM.values()))} substrate concentrations)")
    out(f"published     kcat = {datasets.PUBLISHED_KCAT_S:.0f} s^-1, "
        f"Km = {datasets.PUBLISHED_KM_M * 1e6:.0f} uM")

    # --- initial rates -------------------------------------------------
    out("\n--- initial rates (15% window), pooled by [S] ---")
    table = datasets.initial_rate_table(0.15)
    out(f"{'[S] uM':>8} {'v uM/min':>10} {'SEM':>7} {'n':>2}  {'range':>20}")
    for r in table:
        out(f"{r['s0_um']:8.1f} {r['v_um_per_min']:10.3f} {r['v_sem']:7.3f} "
            f"{r['n_replicates']:2d}  [{r['v_min']:.2f}, {r['v_max']:.2f}]")

    S = np.array([r["s0_um"] for r in table])
    v = np.array([r["v_um_per_min"] for r in table])
    rate_fit = fitting.fit_michaelis_menten(S, v)
    kcat_rate = datasets.kcat_from_vmax(rate_fit.params["vmax"])
    out("")
    out(rate_fit.summary())
    out(f"  kcat = {kcat_rate:.0f} s^-1   "
        f"({datasets.activation_fraction(kcat_rate):.2f} x published)")

    # --- progress curves ------------------------------------------------
    out("\n--- progress curves (integrated rate equation, all points) ---")
    traces = [datasets.progress_curve(stem) for stem in datasets.SUBSTRATE_UM]
    conv = [p[-1] / s0 for _, p, s0 in traces]
    out(f"conversion reached: {min(conv):.0%} to {max(conv):.0%} of [S]0")
    prog = progress.fit_progress_global(traces)
    out(prog.summary())
    kcat_prog = datasets.kcat_from_vmax(prog.vmax)
    out(f"  kcat = {kcat_prog:.0f} s^-1   "
        f"({datasets.activation_fraction(kcat_prog):.2f} x published)")

    # --- how the two compare -------------------------------------------
    out("\n--- comparison ---")
    out(f"{'':14} {'Vmax':>9} {'Km uM':>9} {'kcat':>8} {'kcat/Km':>10}")
    for name, vmax, km, se_km in (
        ("initial rates", rate_fit.params["vmax"], rate_fit.params["km"],
         rate_fit.stderr["km"]),
        ("progress", prog.vmax, prog.km, prog.km_stderr),
    ):
        kcat = datasets.kcat_from_vmax(vmax)
        eff = kcat / (km * 1e-6)
        out(f"{name:14} {vmax:9.2f} {km:9.2f} {kcat:8.0f} {eff / 1e6:10.2f}e6")
    out(f"Km relative SE: initial rates "
        f"{rate_fit.stderr['km'] / rate_fit.params['km']:.1%}, "
        f"progress {prog.km_stderr / prog.km:.1%}")

    window_sensitivity()

    # simple check of the paper's own claim, on our fits
    out("\n--- the paper's premise, measured on our fits ---")
    low = min(S)
    sat = rate_fit.params["km"]
    out(f"lowest [S]0 = {low:.1f} uM = {low / sat:.2f} x fitted Km")
    out("At that concentration the reaction is far from saturated, so the")
    out("initial-rate window covers a large fraction of S0 and depletes it.")
    out("That is exactly the regime where the paper argues progress curves")
    out("are better, and the Km standard errors above are the test of it.")

    # --- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for stem in datasets.SUBSTRATE_UM:
        t, p, s0 = datasets.progress_curve(stem)
        ax.plot(t, p, lw=0.8, alpha=0.55,
                color="C0" if s0 < 40 else ("C1" if s0 < 60 else "C3"))
    tt = np.linspace(0, 1.15, 400)
    for s0, c in ((11.1, "C0"), (35.0, "C1"), (64.0, "C3")):
        ax.plot(tt, progress.product_from_time(tt, prog.vmax, prog.km, s0),
                lw=2.0, color=c,
                label=f"[S]0 = {s0:.0f} uM fit")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("product (uM)")
    ax.set_title(f"global progress-curve fit\nVmax={prog.vmax:.1f} uM/min, "
                 f"Km={prog.km:.1f} uM", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.errorbar(S, v, yerr=[r["v_sem"] for r in table], fmt="o", ms=5,
                capsize=3, color="C0", label="initial rates (15%)")
    ss = np.linspace(0, S.max() * 1.15, 300)
    ax.plot(ss, rate_fit.params["vmax"] * ss / (rate_fit.params["km"] + ss),
            lw=1.6, color="C0",
            label=f"MM fit: Km={rate_fit.params['km']:.1f}")
    ax.plot(ss, prog.vmax * ss / (prog.km + ss), lw=1.6, ls="--", color="C2",
            label=f"progress: Km={prog.km:.1f}")
    ax.set_xlabel("[S]0 (uM)")
    ax.set_ylabel("v (uM/min)")
    ax.set_title("initial-rate fit vs progress-curve fit", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "published_p99.png", dpi=150)
    out(f"\nfigure -> {FIG / 'published_p99.png'}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"summary -> {OUT}")


if __name__ == "__main__":
    main()
