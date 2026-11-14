"""Progress-curve fitting: use the whole time course, not just the first slope.

The initial-rate approach throws away most of the data. You run a reaction for
two minutes, fit a line to the first twenty seconds, and do that at five
substrate concentrations -- then fit Michaelis-Menten to five numbers. Every
point after the linear window is discarded, even though the reason you stopped
measuring was that the substrate ran out, which is itself information about
Km.

The alternative is to fit the integrated rate equation to the entire trace.
For a single-substrate reaction with no product inhibition,

    dP/dt = Vmax * (S0 - P) / (Km + S0 - P)

which integrates to the implicit form

    Vmax * t = P + Km * ln( S0 / (S0 - P) )

That equation has an exact closed-form inverse, so P can be evaluated
directly at any t rather than by iteration:

    P(t) = S0 - Km * W( (S0/Km) * exp( (S0 - Vmax*t) / Km ) )

with W the principal branch of the Lambert W function (scipy.special.lambertw
with k=0). Everything below uses this, which makes the forward model exact,
vectorised and about as fast as evaluating the Michaelis-Menten equation.

Two things matter for the fit to behave:

**Residuals go in product space, not time space.** It is tempting to write the
implicit form as "predicted time minus observed time", but near complete
conversion dt/dP diverges, so a 0.1 uM error at 90% conversion counts
thousands of times more than the same error at 10%. That geometry lets the
last few points of a trace dictate the answer and, on simulated data with
known parameters, it recovers Km to within an order of magnitude at best.
Product space matches what the spectrophotometer actually measures and is the
right choice.

**A baseline term is needed.** A real trace starts at whatever the instrument
read before the reaction began, and both the mixing dead time and drift in the
first seconds put a constant offset into the product curve. Leaving it out
biases Vmax low; at low S0 the bias is large enough to matter.

The assumption that is now *testable* rather than assumed: steady state, no
product inhibition, no enzyme inactivation over the window. Those were assumed
by the initial-rate method too, but only over the first 15% of the curve where
they are most likely to hold. Fitting the whole trace tests them where they
are most likely to break, which is the point.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import lambertw


@dataclass
class ProgressFit:
    vmax: float
    km: float
    baselines: np.ndarray
    vmax_stderr_iid: float
    km_stderr_iid: float
    n_points: int
    n_traces: int
    rss: float
    sigma: float
    s0_used: np.ndarray
    success: bool = True
    message: str = ""
    # Trace-level (cluster) bootstrap, when it was run. These are the numbers
    # to quote: the i.i.d. values treat 2376 points from 18 traces as 2376
    # independent observations, and the residuals of a real trace are not
    # independent from point to point.
    vmax_stderr_cluster: float = np.nan
    km_stderr_cluster: float = np.nan
    km_ci_cluster: tuple = ()
    n_boot: int = 0
    # Per-trace lag-1 residual autocorrelation, diagnostics not a correction.
    lag1: tuple = ()

    @property
    def lag1_median(self):
        return float(np.median(self.lag1)) if len(self.lag1) else np.nan

    @property
    def km_stderr(self):
        """The honest Km standard error: cluster bootstrap if it was run.

        Falls back to the i.i.d. value only when no bootstrap is available,
        which is exactly the number that is wrong by an order of magnitude on
        autocorrelated traces -- so callers that care should check
        `n_boot > 0`.
        """
        return (self.km_stderr_cluster if np.isfinite(self.km_stderr_cluster)
                else self.km_stderr_iid)

    @property
    def vmax_stderr(self):
        return (self.vmax_stderr_cluster
                if np.isfinite(self.vmax_stderr_cluster)
                else self.vmax_stderr_iid)

    @property
    def p0(self):
        """Baseline of the first trace, for convenience."""
        return float(self.baselines[0]) if len(self.baselines) else 0.0

    def summary(self):
        lines = [f"progress-curve fit: {self.n_traces} trace(s), "
                 f"{self.n_points} points, s={self.sigma:.4g}",
                 f"  vmax = {self.vmax:.5g} +/- {self.vmax_stderr:.3g}",
                 f"   km  = {self.km:.5g} +/- {self.km_stderr:.3g}"]
        if self.n_boot:
            lines.append(
                f"         ({self.n_boot}-trace cluster bootstrap; "
                f"i.i.d. SE was km {self.km_stderr_iid:.3g}, "
                f"vmax {self.vmax_stderr_iid:.3g})")
        else:
            lines.append(
                "         SE is i.i.d. over points -- understated when the "
                "residuals are autocorrelated; run cluster_se()")
        if len(self.lag1):
            lo, hi = min(self.lag1), max(self.lag1)
            lines.append(
                f"  residual lag-1 autocorrelation: {lo:.2f} to {hi:.2f} "
                f"across {len(self.lag1)} traces (median "
                f"{self.lag1_median:.2f})")
        return "\n".join(lines)


def product_from_time(t, vmax, km, s0, baseline=0.0):
    """Exact product curve implied by (Vmax, Km, S0).

    The Lambert W form is used as the starting point and then refined by
    Newton iterations on the implicit equation. The refinement is not
    cosmetic: `lambertw` of a large argument inherits the relative error of
    the `exp` that produced it, which for the parameters of a typical
    kinetics run is only about 1e-4 -- visible as a failure of

        Vmax*t == P + Km*ln(S0/(S0-P))

    when it should hold to machine precision. Two Newton steps recover that.

    Parameters are validated rather than clamped: a Km of zero or a negative
    Vmax means the caller has a bug, and silently returning something
    plausible would hide it.
    """
    if vmax <= 0 or km <= 0 or s0 <= 0:
        raise ValueError("vmax, km and s0 must all be positive")
    t = np.asarray(t, dtype=float)
    arg = (s0 / km) * np.exp((s0 - vmax * t) / km)
    p = s0 - km * np.real(lambertw(arg, k=0))

    # Newton polish on f(p) = p + km*ln(s0/(s0-p)) - vmax*t, derivative
    # f'(p) = 1 + km/(s0-p). Seeded from the Lambert W value so it converges
    # in two or three steps. Points already pinned at the boundaries are left
    # alone. NaN from an inf seed is not possible here because lambertw is
    # finite for finite arguments and exp overflow is caught by the clip.
    p = np.clip(p, 0.0, s0 * (1 - 1e-15))
    for _ in range(4):
        gap = np.maximum(s0 - p, 1e-300)
        f = p + km * np.log(s0 / gap) - vmax * t
        df = 1.0 + km / gap
        step = f / df
        p = np.clip(p - step, 0.0, s0 * (1 - 1e-15))

    return p + baseline


def fit_progress_single(t, p_um, s0_um, fit_baseline=True):
    """Fit Vmax and Km to one progress curve.

    t in minutes, p_um product in uM, s0_um starting substrate in uM.

    The curve must reach a decent fraction of S0 for Km to be identifiable:
    if the reaction stops at 2% conversion the term carrying Km is invisible
    and the standard error will say so. That is correct behaviour.
    """
    t, p, s0 = _validate(t, p_um, s0_um)
    return _fit([t], [s0], [p], fit_baseline=fit_baseline)


def fit_progress_global(traces, fit_baseline=True):
    """Fit one Vmax/Km pair to several traces that differ only in S0.

    traces: iterable of (t, p_um, s0_um) tuples.

    Km is then determined by the *relative* shapes of curves at different S0,
    which is the strongest constraint available without knowing the active
    enzyme concentration.
    """
    if not traces:
        raise ValueError("no traces given")
    ts, ps, s0s = [], [], []
    for item in traces:
        t, p, s0 = item
        t, p, s0 = _validate(t, p, s0)
        ts.append(t)
        ps.append(p)
        s0s.append(s0)
    return _fit(ts, s0s, ps, fit_baseline=fit_baseline)


def _validate(t, p, s0):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.shape != p.shape:
        raise ValueError("t and p must be the same length")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
        raise ValueError("t and p must be finite")
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if p.max() > s0 * 1.05:
        raise ValueError("product formed exceeds starting substrate")
    # A point at or past complete conversion implies infinite time and cannot
    # be represented, so drop the tail rather than let it dominate.
    keep = p < s0 * 0.999
    if keep.sum() < 5:
        raise ValueError("fewer than 5 usable points below 99.9% conversion")
    return t[keep], p[keep], float(s0)


def _starts(ts, ps, s0s):
    """Initial guesses, cheap but not stupid."""
    v0 = 0.0
    for t, p in zip(ts, ps):
        if len(t) > 1 and t[-1] > t[0]:
            v0 = max(v0, (p[-1] - p[0]) / (t[-1] - t[0]))
    # the mean observed rate is a lower bound on Vmax, and the fraction of
    # substrate consumed tells you how far below saturation you are
    conv = max(float(np.max(p)) / s0 for p, s0 in zip(ps, s0s))
    vmax0 = max(v0 * (1.5 + 2.0 * conv), 1e-9)
    km0 = float(np.median(s0s))
    return vmax0, km0


def _fit(ts, s0s, ps, fit_baseline=True):
    n_traces = len(ts)
    n_points = int(sum(len(t) for t in ts))
    n_params = 2 + (n_traces if fit_baseline else 0)

    vmax0, km0 = _starts(ts, ps, s0s)

    def unpack(theta):
        vmax = np.exp(theta[0])
        km = np.exp(theta[1])
        offs = theta[2:] if fit_baseline else np.zeros(n_traces)
        return vmax, km, offs

    def resid(theta):
        vmax, km, offs = unpack(theta)
        return np.concatenate([
            product_from_time(t, vmax, km, s0, offs[i]) - p
            for i, (t, p, s0) in enumerate(zip(ts, ps, s0s))
        ])

    theta0 = np.concatenate([[np.log(vmax0), np.log(km0)],
                             np.zeros(n_traces if fit_baseline else 0)])
    out = least_squares(resid, theta0, method="trf", max_nfev=20000)

    vmax, km, offs = unpack(out.x)
    rss = float(2 * out.cost)
    dof = n_points - n_params
    sigma2 = rss / dof if dof > 0 else np.nan

    se_vmax = se_km = np.nan
    try:
        J = out.jac
        cov = np.linalg.inv(J.T @ J) * sigma2
        # delta method: parameters 0 and 1 are log-scale
        se_vmax = float(np.sqrt(max(cov[0, 0], 0.0)) * vmax)
        se_km = float(np.sqrt(max(cov[1, 1], 0.0)) * km)
    except np.linalg.LinAlgError:
        pass

    # Per-trace lag-1 autocorrelation of the residuals. Reported, never used
    # to rescale the i.i.d. SE: a single lag-overlap factor is not a valid
    # correction when a trace's residuals are almost perfectly correlated
    # (here 0.94-1.00), so the correction is the cluster bootstrap instead.
    lag1 = []
    for i, (t, p, s0) in enumerate(zip(ts, ps, s0s)):
        res = product_from_time(t, vmax, km, s0, offs[i]) - p
        lag1.append(_lag1(res))

    return ProgressFit(
        vmax=float(vmax), km=float(km), baselines=np.asarray(offs, dtype=float),
        vmax_stderr_iid=se_vmax, km_stderr_iid=se_km,
        n_points=n_points, n_traces=n_traces,
        rss=rss, sigma=float(np.sqrt(sigma2)) if dof > 0 else np.nan,
        s0_used=np.asarray(s0s, dtype=float),
        success=bool(out.success), message=out.message,
        lag1=tuple(lag1),
    )


def _lag1(res):
    """Lag-1 autocorrelation of one trace's residuals, NaN if undefined."""
    res = np.asarray(res, dtype=float)
    if len(res) < 3:
        return np.nan
    a, b = res[:-1], res[1:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cluster_se(traces, n_boot=400, seed=0, fit_baseline=True, alpha=0.05):
    """Trace-level (cluster) bootstrap of the global progress-curve fit.

    The parameter of interest, Km, is shared across 18 traces, and the
    residuals inside a trace are strongly autocorrelated (lag-1 0.94-1.00 on
    the P99 data). Treating every point as an independent observation
    understates the Km standard error by more than an order of magnitude.

    Resampling whole traces treats each trace as the independent unit, which
    is the level at which the data really replicate: the same enzyme
    preparation, pipetting and drift are shared by every point in a trace.

    Resampling is stratified by S0 by default -- each distinct substrate
    concentration contributes the same number of traces as it originally
    did -- because Km is identified by the *relative* shapes at different S0,
    and an unstratified draw can leave the fit with almost no leverage.

    Returns the point fit with `vmax_stderr_cluster`, `km_stderr_cluster`,
    `km_ci_cluster` (percentile interval) and `n_boot` filled in.
    """
    traces = list(traces)
    if not traces:
        raise ValueError("no traces given")
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20 for a usable SE")

    point = _fit([np.asarray(t, float) for t, _p, _s in traces],
                 [float(s) for _t, _p, s in traces],
                 [np.asarray(p, float) for _t, p, _s in traces],
                 fit_baseline=fit_baseline)

    groups = {}
    for tr in traces:
        groups.setdefault(tr[2], []).append(tr)
    keys = sorted(groups)

    rng = np.random.default_rng(seed)
    kms, vmaxs = [], []
    for _ in range(n_boot):
        sample = []
        for k in keys:
            members = groups[k]
            pick = rng.integers(0, len(members), len(members))
            sample.extend(members[i] for i in pick)
        try:
            f = _fit([np.asarray(t, float) for t, _p, _s in sample],
                     [float(s) for _t, _p, s in sample],
                     [np.asarray(p, float) for _t, p, _s in sample],
                     fit_baseline=fit_baseline)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if f.success and np.isfinite(f.km):
            kms.append(f.km)
            vmaxs.append(f.vmax)

    if len(kms) < 20:
        raise ValueError(f"only {len(kms)} of {n_boot} bootstrap fits converged")
    kms = np.asarray(kms)
    vmaxs = np.asarray(vmaxs)
    lo, hi = np.percentile(kms, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    point.km_stderr_cluster = float(kms.std(ddof=1))
    point.vmax_stderr_cluster = float(vmaxs.std(ddof=1))
    point.km_ci_cluster = (float(lo), float(hi))
    point.n_boot = len(kms)
    return point
