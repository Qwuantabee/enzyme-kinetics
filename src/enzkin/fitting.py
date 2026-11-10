"""Nonlinear fitting of the rate equations, with uncertainties.

The point of doing it this way rather than with a linearised plot is that
least squares on the untransformed data assumes the error is on v, which is
what the assay actually measures. Lineweaver-Burk inverts v and drags the
error along with it, so the low [S] points -- the noisiest ones -- end up
dominating the fit. See linearize.py for the demonstration.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from . import models


@dataclass
class FitResult:
    model: str
    params: dict
    stderr: dict
    n: int
    rss: float
    sigma: float          # residual standard deviation
    r_squared: float
    aicc: float
    cov: np.ndarray = field(repr=False, default=None)
    success: bool = True
    message: str = ""

    @property
    def dof(self):
        return self.n - len(self.params)

    def ci95(self, name):
        """95% interval from the standard error, using the t distribution.

        Using 1.96 here is a mistake that is easy to make and hard to notice.
        A kinetics run is typically 8 points and 2 parameters, so 6 degrees
        of freedom, where the right critical value is 2.45 rather than 1.96.
        Measured over 500 simulated datasets, the normal version covered the
        true Km only 87% of the time while claiming 95%; with the t value it
        is close to nominal. See scripts/validate_fits.py.

        Still an approximation: it assumes the sampling distribution is
        symmetric, and these parameters are bounded below at zero. When the
        SE is a large fraction of the estimate, use bootstrap_ci instead.
        """
        p, se = self.params[name], self.stderr[name]
        crit = stats.t.ppf(0.975, self.dof) if self.dof > 0 else np.inf
        return p - crit * se, p + crit * se

    def summary(self):
        lines = [f"model: {self.model}   n={self.n}   s={self.sigma:.4g}   "
                 f"R2={self.r_squared:.4f}   AICc={self.aicc:.2f}"]
        for k in self.params:
            se = self.stderr[k]
            pct = 100 * se / self.params[k] if self.params[k] else float("nan")
            lines.append(f"  {k:>6} = {self.params[k]:.4g} +/- {se:.3g}  ({pct:.1f}%)")
        return "\n".join(lines)


def _aicc(n, k, rss):
    """Small sample corrected AIC. Kinetics datasets are small, so the
    correction matters -- plain AIC under-penalises the 4 parameter mixed
    model on a 20 point dataset."""
    aic = n * np.log(rss / n) + 2 * k
    denom = n - k - 1
    if denom <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / denom


def _run_fit(resid_fn, p0, n_points, weights_used):
    """Shared least squares driver.

    Parameters are fitted in log space so they stay positive without hard
    bounds -- bounded fits park parameters on the boundary and then the
    covariance matrix is meaningless.
    """
    log_p0 = np.log(p0)

    def resid_log(log_p):
        return resid_fn(np.exp(log_p))

    out = least_squares(resid_log, log_p0, method="lm", max_nfev=20000)

    k = len(p0)
    dof = n_points - k
    rss = float(2 * out.cost)
    sigma2 = rss / dof if dof > 0 else np.nan

    # covariance in log space, then delta method back to linear space:
    # var(p) = (dp/dlogp)^2 * var(logp) = p^2 * var(logp)
    J = out.jac
    try:
        cov_log = np.linalg.inv(J.T @ J) * sigma2
    except np.linalg.LinAlgError:
        cov_log = np.full((k, k), np.nan)

    p = np.exp(out.x)
    scale = np.outer(p, p)
    cov = cov_log * scale
    se = np.sqrt(np.diag(cov))

    return p, se, cov, rss, sigma2, dof, out


def fit_michaelis_menten(S, v, weights="none", p0=None):
    """Fit v = vmax*S/(km+S).

    weights:
        "none"     ordinary least squares, assumes constant absolute error
        "relative" weights by 1/v, assumes constant percentage error

    Which one is right depends on the assay. Continuous spectrophotometric
    assays are usually closer to constant absolute error; anything involving
    a dilution series or counting tends to be proportional.
    """
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    if S.shape != v.shape:
        raise ValueError("S and v must be the same length")
    if np.any(S < 0) or np.any(v < 0):
        raise ValueError("negative concentrations or rates")

    if p0 is None:
        # vmax a bit above the largest observed rate, km at the substrate
        # concentration where v first passes half of that
        vmax0 = max(v.max() * 1.2, 1e-9)
        half = vmax0 / 2
        idx = np.argmin(np.abs(v - half))
        km0 = max(S[idx], S[S > 0].min() if np.any(S > 0) else 1.0)
        p0 = np.array([vmax0, km0])

    w = _weights(v, weights)

    def resid(p):
        return (models.michaelis_menten(S, *p) - v) * w

    p, se, cov, rss, sigma2, dof, out = _run_fit(resid, p0, len(v), weights)

    pred = models.michaelis_menten(S, *p)
    return FitResult(
        model="michaelis_menten",
        params=dict(zip(("vmax", "km"), p)),
        stderr=dict(zip(("vmax", "km"), se)),
        n=len(v),
        rss=rss,
        sigma=float(np.sqrt(sigma2)) if dof > 0 else np.nan,
        r_squared=_r2(v, pred),
        aicc=_aicc(len(v), 2, rss),
        cov=cov,
        success=bool(out.success),
        message=out.message,
    )


def fit_inhibition(S, I, v, model="mixed", weights="none", p0=None):
    """Fit one of the inhibition models to a full [S] x [I] matrix of rates.

    All the inhibitor concentrations are fitted together in one global fit,
    which is the whole point -- fitting each [I] separately and then eyeballing
    how Km and Vmax move is where most of the misassignment of mechanism comes
    from.
    """
    if model not in models.INHIBITION_MODELS:
        raise ValueError(f"unknown model {model!r}, "
                         f"choose from {sorted(models.INHIBITION_MODELS)}")

    fn, names = models.INHIBITION_MODELS[model]
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (S.shape == I.shape == v.shape):
        raise ValueError("S, I and v must all be the same length")

    if p0 is None:
        # start from the uninhibited points if there are any
        base = I == I.min()
        mm0 = fit_michaelis_menten(S[base], v[base]) if base.sum() >= 3 else None
        vmax0 = mm0.params["vmax"] if mm0 else v.max() * 1.2
        km0 = mm0.params["km"] if mm0 else np.median(S[S > 0])
        ki0 = np.median(I[I > 0]) if np.any(I > 0) else 1.0
        p0 = [vmax0, km0] + [ki0] * (len(names) - 2)
    p0 = np.asarray(p0, dtype=float)

    w = _weights(v, weights)

    def resid(p):
        return (fn(S, I, *p) - v) * w

    p, se, cov, rss, sigma2, dof, out = _run_fit(resid, p0, len(v), weights)

    pred = fn(S, I, *p)
    return FitResult(
        model=model,
        params=dict(zip(names, p)),
        stderr=dict(zip(names, se)),
        n=len(v),
        rss=rss,
        sigma=float(np.sqrt(sigma2)) if dof > 0 else np.nan,
        r_squared=_r2(v, pred),
        aicc=_aicc(len(v), len(names), rss),
        cov=cov,
        success=bool(out.success),
        message=out.message,
    )


def compare_models(S, I, v, candidates=None, weights="none"):
    """Fit every inhibition model and rank by AICc.

    Returns a list of FitResult sorted best first. A difference in AICc of
    less than about 2 is not evidence for one mechanism over another, and
    the function will not pretend otherwise -- check delta_aicc yourself
    before writing a mechanism into a figure legend.
    """
    if candidates is None:
        candidates = list(models.INHIBITION_MODELS)

    results = []
    for name in candidates:
        try:
            results.append(fit_inhibition(S, I, v, model=name, weights=weights))
        except Exception as exc:  # a model that can't fit is informative too
            print(f"  {name}: fit failed ({exc})")

    results.sort(key=lambda r: r.aicc)
    return results


def bootstrap_ci(S, v, n_boot=2000, alpha=0.05, seed=None, I=None, model=None):
    """Percentile bootstrap on the residuals.

    Resamples residuals rather than data points so the design (the chosen
    substrate concentrations) is held fixed, which is how the experiment was
    actually run.
    """
    rng = np.random.default_rng(seed)
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)

    if I is None:
        base = fit_michaelis_menten(S, v)
        pred = models.michaelis_menten(S, **base.params)
        refit = lambda y: fit_michaelis_menten(S, y)
    else:
        I = np.asarray(I, dtype=float)
        base = fit_inhibition(S, I, v, model=model)
        fn, _ = models.INHIBITION_MODELS[base.model]
        pred = fn(S, I, **base.params)
        refit = lambda y: fit_inhibition(S, I, y, model=base.model)

    resid = v - pred
    draws = {k: [] for k in base.params}

    for _ in range(n_boot):
        y = pred + rng.choice(resid, size=len(resid), replace=True)
        if np.any(y <= 0):
            y = np.clip(y, 1e-12, None)
        try:
            r = refit(y)
        except Exception:
            continue
        for k, val in r.params.items():
            draws[k].append(val)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(d, lo)), float(np.percentile(d, hi)))
            for k, d in draws.items() if d}


def _weights(v, scheme):
    if scheme == "none":
        return np.ones_like(v)
    if scheme == "relative":
        return 1.0 / np.clip(np.abs(v), 1e-12, None)
    raise ValueError(f"unknown weighting {scheme!r}")


def _r2(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
