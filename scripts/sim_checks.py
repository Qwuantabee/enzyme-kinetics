"""Two checks on the simulation study in validate_fits.py.

Run:  python scripts/sim_checks.py      (writes results_sim_checks.txt)

(a) What the Lineweaver-Burk Km SD under constant absolute error is made of.
    Lineweaver-Burk's Km is a slope divided by an intercept that crosses zero,
    so its sampling distribution has no finite variance and the sample SD is
    set by a handful of fits. This prints how much of the README's figure one
    fit carries, a spread that does not depend on the tail (IQR / 1.349, which
    equals the SD for a normal distribution), how many fits return a negative
    Km, and the same quantities on eight further blocks of 500 seeds.

(b) Which model wins in the mechanism study, not only whether the true one
    does: the full pick counts under AICc and under plain AIC, the strict
    score for a noncompetitive truth (validate_fits.py counts mixed as
    correct there), and whether a wider substrate range changes how often
    mixed beats a competitive truth.

Same simulators, seeds and fitting calls as validate_fits.py. Nothing here
changes a number that script prints.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import fitting, linearize, models, simulate  # noqa: E402

OUT = ROOT / "results_sim_checks.txt"
VMAX, KM, NOISE = 100.0, 25.0, 0.05
METHODS = {
    "nonlinear": lambda S, v: fitting.fit_michaelis_menten(S, v).params,
    "lineweaver_burk": linearize.params_from_lineweaver_burk,
    "hanes_woolf": linearize.params_from_hanes_woolf,
    "eadie_hofstee": linearize.params_from_eadie_hofstee,
}
MECHS = ("competitive", "uncompetitive", "noncompetitive", "mixed")


def km_errors(seeds):
    """Relative Km error per method under constant absolute error."""
    errs = {m: [] for m in METHODS}
    for seed in seeds:
        S, v, _ = simulate.substrate_series(
            vmax=VMAX, km=KM, noise=NOISE, noise_type="absolute", seed=seed)
        for m, fn in METHODS.items():
            errs[m].append(fn(S, v)["km"] / KM - 1)
    return {m: np.array(e) for m, e in errs.items()}


def robust_sd(x):
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349


def aic_plain(r):
    return r.n * np.log(r.rss / r.n) + 2 * len(r.params)


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("Simulation checks (scripts/sim_checks.py)")
    out("=" * 74)
    S, v, _ = simulate.substrate_series(vmax=VMAX, km=KM, noise=0, seed=0)
    out(f"design: {len(S)} substrate concentrations, {S[0]/KM:.1f} to {S[-1]/KM:.0f} x Km, geometric")
    out(f"        true Km = {KM:g}, Vmax = {VMAX:g}; absolute noise SD = {NOISE*VMAX:g} "
        f"= {100*NOISE*VMAX/v[0]:.0f}% of the lowest true rate ({v[0]:.2f}), "
        f"{100*NOISE*VMAX/v[-1]:.1f}% of the highest ({v[-1]:.2f})")

    out()
    out("--- (a) Lineweaver-Burk Km error under constant absolute error ---")
    errs = km_errors(range(500))
    lb = errs["lineweaver_burk"]
    order = np.argsort(-np.abs(lb))
    out("seeds 0-499 (the README table):")
    out(f"  sample SD, all 500 fits                     {100*lb.std():9.1f}%")
    for drop in (1, 5, 10):
        keep = np.ones(len(lb), bool)
        keep[order[:drop]] = False
        out(f"  sample SD without the {drop:>2} most extreme fit(s)  {100*lb[keep].std():9.1f}%")
    out(f"  most extreme single fit: seed {int(order[0])}, Km error {100*lb[order[0]]:.0f}%")
    out(f"  fits returning a negative Km: {int((lb < -1).sum())} of {len(lb)}")
    out()
    out(f"  {'method':<16} {'median error':>13} {'robust SD':>10} {'sample SD':>10}")
    for m, e in errs.items():
        out(f"  {m:<16} {100*np.median(e):>+12.1f}% {100*robust_sd(e):>9.1f}% {100*e.std():>9.1f}%")
    out("  robust SD = IQR / 1.349 (equals the SD for a normal distribution)")
    out(f"  Lineweaver-Burk / nonlinear, robust SD: {robust_sd(lb)/robust_sd(errs['nonlinear']):.1f}x")
    out()
    out("eight further blocks of 500 seeds, same design:")
    out(f"  {'seeds':<10} {'LB sample SD':>13} {'LB robust SD':>13} {'LB negative Km':>15} {'nonlinear SD':>13}")
    sds, rsds = [100 * lb.std()], [100 * robust_sd(lb)]
    for b in range(1, 9):
        e = km_errors(range(500 * b, 500 * (b + 1)))
        x = e["lineweaver_burk"]
        sds.append(100 * x.std())
        rsds.append(100 * robust_sd(x))
        out(f"  {500*b:>4}-{500*(b+1)-1:<5} {100*x.std():>12.1f}% {100*robust_sd(x):>12.1f}% "
            f"{int((x < -1).sum()):>11} /500 {100*e['nonlinear'].std():>12.1f}%")
    out(f"  over all nine blocks: LB sample SD {min(sds):.0f}% to {max(sds):.0f}%, "
        f"LB robust SD {min(rsds):.1f}% to {max(rsds):.1f}%")

    out()
    out("--- (b) mechanism identification: which model ranked first ---")
    out("100 simulated experiments per row, seeds 0-99, the validate_fits.py grid")
    out(f"candidates: {', '.join(f'{k} ({len(v[1])} parameters)' for k, v in models.INHIBITION_MODELS.items())}")
    out()
    out(f"  {'truth':<15} {'noise':>5}  {'strict hit':>10}  ranked first by AICc | by plain AIC")
    for mech in MECHS:
        for noise in (0.02, 0.05, 0.10):
            aicc, aic = Counter(), Counter()
            for seed in range(100):
                S, I, v, _ = simulate.inhibition_grid(mechanism=mech, noise=noise, seed=seed)
                res = fitting.compare_models(S, I, v)
                aicc[res[0].model] += 1
                aic[min(res, key=aic_plain).model] += 1
            fmt = lambda c: ", ".join(f"{k} {c[k]}" for k in MECHS if c[k])  # noqa: E731
            out(f"  {mech:<15} {noise:>5.0%}  {aicc[mech]:>9}%  {fmt(aicc)} | {fmt(aic)}")
    out("  strict hit = the true model itself ranked first (validate_fits.py also")
    out("  counts mixed as correct when the truth is noncompetitive)")

    out()
    out("competitive truth, 5% noise, 100 experiments, 7 x 4 grid from Km/8 up to:")
    for top in (8, 32, 128, 512):
        picks = Counter()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            s_vals = np.geomspace(KM / 8, KM * top, 7)
            S, I = np.meshgrid(s_vals, np.array([0.0, 5.0, 15.0, 45.0]))
            S, I = S.ravel(), I.ravel()
            v = models.competitive(S, I, VMAX, KM, 10.0) * (1 + rng.normal(0, 0.05, S.shape))
            picks[fitting.compare_models(S, I, np.clip(v, 1e-12, None))[0].model] += 1
        out(f"  {top:>4} x Km: competitive {picks['competitive']}, mixed {picks['mixed']}")

    n = 28
    gap_aicc = (2 * 4 + 2 * 4 * 5 / (n - 5)) - (2 * 3 + 2 * 3 * 4 / (n - 4))
    out()
    out("what a fixed penalty predicts for a 3-parameter truth nested in mixed, n = 28:")
    for name, gap in (("AICc", gap_aicc), ("plain AIC", 2.0)):
        f_crit = (np.exp(gap / n) - 1) * (n - 4)
        out(f"  {name:<9} penalty gap {gap:.2f} -> mixed wins when F(1,{n-4}) > {f_crit:.2f}, "
            f"probability {100*stats.f.sf(f_crit, 1, n - 4):.1f}% (linear-model approximation)")
    lo, hi = stats.binomtest(15, 100).proportion_ci(method="wilson")
    out(f"  observed 15 of 100: Wilson 95% interval {100*lo:.1f}% to {100*hi:.1f}%")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary -> {OUT.name}")


if __name__ == "__main__":
    main()
