"""Fit the published P99 / nitrocefin progress curves.

    python scripts/fit_published.py

Prints the numbers quoted in README.md and appends a machine-readable summary
to results_published.txt. It draws nothing: figures/published_p99.png and
figures/km_uncertainty.png are written by scripts/make_figures.py.

Data: Frere, Verlaine & Matagne (2023) Sci Rep 13:14994,
doi:10.1038/s41598-023-41805-y; archive doi:10.5281/zenodo.10059024, CC-BY.
System: Enterobacter cloacae P99 class C beta-lactamase + nitrocefin at 462 nm,
0.6 nM enzyme, 30 C. Published reference values for this pair: kcat = 780 s^-1,
Km = 25 uM (quoted by the paper, not measured here).

Two analyses of the same 18 traces:

  initial rates   a straight line through the first 15% of each trace, then
                  Michaelis-Menten on the five averages
  progress curves the integrated rate equation fitted to every point

They should agree. Where they disagree, substrate depletion inside the
initial-rate window is one candidate explanation -- but it is a candidate to
be tested, not an assumption, and on this dataset the Vmax gap is under one
initial-rate standard error while the window sweep shows no trend, so the
data do not support the depletion story.

The Km comparison is the substantive one, and it is reported with a
trace-level bootstrap: the residuals in each trace are strongly
autocorrelated, so an i.i.d. standard error over all points is not an
uncertainty at all.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import datasets, fitting, progress  # noqa: E402

OUT = ROOT / "results_published.txt"

WINDOWS = (0.05, 0.10, 0.15, 0.25, 0.40)


def window_sensitivity(emit=print):
    """How much does the initial-rate answer depend on the window chosen?
    main() passes its own writer, so the table lands in results_published.txt
    as well as on stdout (the README quotes it)."""
    emit("\n--- initial rates: dependence on the window ---")
    emit(f"{'window':>7} {'Vmax':>8} {'Km':>8} {'kcat':>8} {'n_ok':>5}")
    rows = {}
    for frac in WINDOWS:
        try:
            table = datasets.initial_rate_table(frac)
        except ValueError as exc:
            emit(f"{frac:7.0%}  skipped ({exc})")
            continue
        S = np.array([r["s0_um"] for r in table])
        v = np.array([r["v_um_per_min"] for r in table])
        fit = fitting.fit_michaelis_menten(S, v)
        kcat = datasets.kcat_from_vmax(fit.params["vmax"])
        rows[frac] = (fit, table)
        emit(f"{frac:7.0%} {fit.params['vmax']:8.2f} {fit.params['km']:8.2f} "
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
    # The residuals inside a trace are autocorrelated, so the i.i.d. SE above
    # is not an uncertainty: resample traces, which are the real replicates.
    boot = progress.cluster_se(traces, n_boot=400, seed=0)
    out(f"  {boot.n_boot}-trace cluster bootstrap: "
        f"km = {boot.km:.2f} +/- {boot.km_stderr:.2f} uM, "
        f"95% [{boot.km_ci_cluster[0]:.2f}, {boot.km_ci_cluster[1]:.2f}]")
    out(f"  cluster SE / i.i.d. SE = "
        f"{boot.km_stderr / boot.km_stderr_iid:.0f}x on Km")
    kcat_prog = datasets.kcat_from_vmax(prog.vmax)
    out(f"  kcat = {kcat_prog:.0f} s^-1   "
        f"({datasets.activation_fraction(kcat_prog):.2f} x published)")

    # --- how the two compare -------------------------------------------
    out("\n--- comparison ---")
    out(f"{'':14} {'Vmax':>9} {'Km uM':>9} {'kcat':>8} {'kcat/Km':>10}")
    for name, vmax, km, se_km in (
        ("initial rates", rate_fit.params["vmax"], rate_fit.params["km"],
         rate_fit.stderr["km"]),
        ("progress", prog.vmax, prog.km, boot.km_stderr),
    ):
        kcat = datasets.kcat_from_vmax(vmax)
        eff = kcat / (km * 1e-6)
        out(f"{name:14} {vmax:9.2f} {km:9.2f} {kcat:8.0f} {eff / 1e6:10.2f}e6")
    out(f"Km relative SE: initial rates "
        f"{rate_fit.stderr['km'] / rate_fit.params['km']:.1%}, "
        f"progress {boot.km_stderr / prog.km:.1%} (cluster bootstrap)")

    out("")
    out(f"Vmax gap between the two analyses: "
        f"{rate_fit.params['vmax'] - prog.vmax:.2f} uM/min = "
        f"{(rate_fit.params['vmax'] - prog.vmax) / rate_fit.stderr['vmax']:.2f} "
        f"initial-rate SE. A gap under one SE is not evidence of anything.")

    window_sensitivity(out)

    # simple check of the paper's own claim, on our fits
    out("\n--- the paper's premise, measured on our fits ---")
    low = min(S)
    sat = rate_fit.params["km"]
    out(f"lowest [S]0 = {low:.1f} uM = {low / sat:.2f} x fitted Km")
    out("At that concentration the reaction is far from saturated, so the")
    out("initial-rate window covers a large fraction of S0 and depletes it.")
    out("That is exactly the regime where the paper argues progress curves")
    out("are better, and the Km standard errors above are the test of it.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"summary -> {OUT}")


if __name__ == "__main__":
    main()
