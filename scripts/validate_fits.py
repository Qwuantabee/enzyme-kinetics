"""Simulation study: does the fitting code actually work, and how much worse
is the double reciprocal plot?

Run:  python scripts/validate_fits.py

Writes figures to figures/ and prints the numbers quoted in the README.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enzkin import fitting, linearize, plotting, simulate  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

N_TRIALS = 500
TRUE_VMAX, TRUE_KM = 100.0, 25.0
NOISE = 0.05


def recovery_study(noise_type="relative"):
    """Fit the same simulated experiment 500 times with different noise and
    compare the nonlinear fit against the three linearisations.

    Run under both error models on purpose. The usual argument against
    Lineweaver-Burk assumes the error on v is constant in absolute terms,
    which is when inverting v really does blow up the low-[S] points. If the
    assay error is proportional instead, the transforms behave far better
    than their reputation. Which one applies is a property of the assay, not
    of the maths, so it is worth knowing which one you have before choosing
    how to fit.
    """
    print(f"\n--- parameter recovery, {N_TRIALS} datasets, "
          f"{NOISE:.0%} {noise_type} noise, 8 points ---")

    errs = {k: {"vmax": [], "km": []} for k in
            ("nonlinear", "lineweaver_burk", "hanes_woolf", "eadie_hofstee")}
    covered = 0

    for seed in range(N_TRIALS):
        S, v, truth = simulate.substrate_series(
            vmax=TRUE_VMAX, km=TRUE_KM, noise=NOISE, noise_type=noise_type,
            seed=seed)

        nl = fitting.fit_michaelis_menten(S, v)
        ests = {
            "nonlinear": nl.params,
            "lineweaver_burk": linearize.params_from_lineweaver_burk(S, v),
            "hanes_woolf": linearize.params_from_hanes_woolf(S, v),
            "eadie_hofstee": linearize.params_from_eadie_hofstee(S, v),
        }
        for name, p in ests.items():
            for key in ("vmax", "km"):
                errs[name][key].append(p[key] / truth[key] - 1)

        lo, hi = nl.ci95("km")
        covered += lo <= truth["km"] <= hi

    print(f"{'method':<18} {'Km bias':>9} {'Km SD':>8} {'Vmax bias':>10} {'Vmax SD':>8}")
    for name, d in errs.items():
        km, vm = np.array(d["km"]), np.array(d["vmax"])
        print(f"{name:<18} {100*km.mean():>8.1f}% {100*km.std():>7.1f}% "
              f"{100*vm.mean():>9.1f}% {100*vm.std():>7.1f}%")

    print(f"\n95% CI coverage for Km (nonlinear): {100*covered/N_TRIALS:.1f}% "
          f"(nominal 95%)")

    fig = plotting.plot_recovery(errs["nonlinear"]["km"],
                                 errs["lineweaver_burk"]["km"],
                                 errs["hanes_woolf"]["km"])
    fig.savefig(FIG / f"km_recovery_{noise_type}.png", dpi=150)
    plt.close(fig)
    return errs


def design_study():
    """How far up the substrate range do you need to go?

    This is the practical question when substrate is expensive or barely
    soluble, and the answer is usually 'further than you went'.
    """
    print("\n--- effect of substrate range on parameter precision ---")
    print(f"{'max [S]':>10} {'Vmax SE':>10} {'Km SE':>8}  {'% of Vmax reached':>18}")

    for mult in (1, 2, 5, 10, 20):
        v_se, km_se = [], []
        for seed in range(200):
            S, v, _ = simulate.substrate_series(
                vmax=TRUE_VMAX, km=TRUE_KM, s_max_multiple=mult,
                noise=NOISE, seed=seed)
            fit = fitting.fit_michaelis_menten(S, v)
            v_se.append(fit.stderr["vmax"] / fit.params["vmax"])
            km_se.append(fit.stderr["km"] / fit.params["km"])
        reached = 100 * mult / (1 + mult)
        print(f"{mult:>8}xKm {100*np.median(v_se):>9.1f}% "
              f"{100*np.median(km_se):>7.1f}% {reached:>17.0f}%")


def mechanism_study():
    """Can the AICc comparison tell the four inhibition mechanisms apart,
    and at what noise level does it stop being able to?"""
    print("\n--- mechanism identification ---")
    print(f"{'true mechanism':<16} {'noise':>6} {'picked correctly':>17}")

    for mech in ("competitive", "uncompetitive", "noncompetitive", "mixed"):
        for noise in (0.02, 0.05, 0.10):
            hits = 0
            trials = 100
            for seed in range(trials):
                S, I, v, _ = simulate.inhibition_grid(
                    mechanism=mech, noise=noise, seed=seed)
                res = fitting.compare_models(S, I, v)
                best = res[0].model
                # mixed nests noncompetitive, so count that as correct
                ok = best == mech or (mech == "noncompetitive" and best == "mixed")
                hits += ok
            print(f"{mech:<16} {noise:>5.0%} {100*hits/trials:>16.0f}%")


def example_figures():
    """One worked example of each figure, for the README."""
    S, v, truth = simulate.substrate_series(vmax=TRUE_VMAX, km=TRUE_KM,
                                            noise=NOISE, n=8, seed=42)
    fit = fitting.fit_michaelis_menten(S, v)
    print("\n--- worked example (seed 42) ---")
    print(f"true values: vmax={truth['vmax']}, km={truth['km']}")
    print(fit.summary())

    lb = linearize.params_from_lineweaver_burk(S, v)
    print(f"  Lineweaver-Burk on the same data: "
          f"vmax={lb['vmax']:.4g}, km={lb['km']:.4g}")

    boot = fitting.bootstrap_ci(S, v, n_boot=2000, seed=42)
    for k, (lo, hi) in boot.items():
        print(f"  bootstrap 95% CI {k}: [{lo:.4g}, {hi:.4g}]")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    plotting.plot_fit(S, v, fit, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "mm_fit.png", dpi=150)
    plt.close(fig)

    fig = plotting.plot_linearizations(S, v, fit)
    fig.savefig(FIG / "linearizations.png", dpi=150)
    plt.close(fig)

    S, I, v, truth = simulate.inhibition_grid(mechanism="mixed", ki=10.0,
                                              kis=40.0, noise=0.03, seed=3)
    results = fitting.compare_models(S, I, v)
    print("\n--- inhibition example, truth is mixed (Ki=10, Kis=40) ---")
    best_aicc = results[0].aicc
    for r in results:
        print(f"  {r.model:<15} AICc={r.aicc:8.2f}  "
              f"dAICc={r.aicc - best_aicc:6.2f}")
    print(results[0].summary())

    fig, ax = plt.subplots(figsize=(6, 4.2))
    plotting.plot_inhibition(S, I, v, results[0], ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "inhibition_fit.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    recovery_study("relative")
    recovery_study("absolute")
    design_study()
    mechanism_study()
    example_figures()
    print(f"\nfigures written to {FIG}")
