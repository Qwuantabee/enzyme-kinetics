"""IC50 to Ki: the Cheng-Prusoff relations, and what they assume.

An IC50 is an assay-specific number. Change the substrate concentration and it
changes; change the enzyme concentration and it can change too. Ki is a
property of the inhibitor-enzyme pair. Converting between them is routine, and
routine enough that the assumptions behind the conversion are usually dropped
by the time anyone quotes the result.

For a competitive inhibitor in a steady-state assay (Cheng & Prusoff, 1973):

    Ki = IC50 / (1 + [S]/Km)

For an uncompetitive inhibitor:

    Ki = IC50 / (1 + Km/[S])

For a noncompetitive inhibitor (Ki = Kis) neither term applies and the IC50 is
independent of [S]:

    Ki = IC50

For a mixed inhibitor the two constants differ and one IC50 cannot give you
both; it gives a combination. That is a real limitation and this module
refuses to pretend otherwise -- see `ki_from_ic50` with model="mixed".

What that means in practice, and why the assumptions have to be stated:

1. **The mechanism must already be known.** Cheng-Prusoff is not a way to
   discover mechanism, it is a way to convert a number once you know it. Using
   the competitive form on an uncompetitive inhibitor biases Ki in the wrong
   direction by roughly (1+[S]/Km)(1+Km/[S]) -- at [S] = Km that is a factor
   of 4.

2. **[S] and Km must be the assay's, not a literature value.** Ki computed
   with the wrong Km is wrong by the ratio of the two correction factors. A
   competitive conversion at [S]=10*Km is a factor-of-11 correction, so a Km
   that is off by 30% moves Ki by about 25%.

3. **[S] must be the free concentration, and close to what you added.** If the
   assay consumes a large fraction of substrate -- which is exactly what the
   published progress-curve dataset in this repo does -- the mean free [S]
   during the measurement is lower than [S]0 and the correction is
   overestimated.

4. **The inhibitor must not deplete.** Ki is a dissociation constant; IC50 in
   a tight-binding assay reflects [E] as much as affinity. Cheng-Prusoff
   assumes [I] >> [E]. When IC50 approaches the enzyme concentration, the
   apparent IC50 is a stoichiometric titration and the conversion is invalid.

5. **Steady state, no slow-binding, no substrate inhibition.**

`check_assumptions` reports which of these are at risk for a given call, and
`tight_binding_warning` flags the case that silently flips an inhibitor's rank
order.

References
----------
Cheng Y, Prusoff WH (1973) Biochem Pharmacol 22:3099-3108.
  doi:10.1016/0006-2952(73)90196-7
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares

#: Mechanisms for which a single IC50 maps onto a single Ki.
SINGLE_KI_MODELS = ("competitive", "uncompetitive", "noncompetitive")


def _require_positive(**kw):
    for name, val in kw.items():
        if val is None or not np.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be positive and finite, got {val!r}")


def correction_factor(s_conc, km, model):
    """The (1 + ...) term for a competitive or uncompetitive inhibitor.

    Competitive:     1 + [S]/Km       (substrate competes, raises IC50)
    Uncompetitive:   1 + Km/[S]       (inhibitor needs the ES complex)
    Noncompetitive:  1                (IC50 independent of [S])
    """
    _require_positive(s_conc=s_conc, km=km)
    if model == "competitive":
        return 1.0 + s_conc / km
    if model == "uncompetitive":
        return 1.0 + km / s_conc
    if model == "noncompetitive":
        return 1.0
    raise ValueError(
        f"no single correction factor for model {model!r}; a mixed inhibitor "
        f"has two Ki values and one IC50 cannot separate them. Use "
        f"fit_inhibition on a full [S] x [I] matrix instead."
    )


def ki_from_ic50(ic50, s_conc, km, model="competitive"):
    """Convert an IC50 to a Ki using the Cheng-Prusoff relation.

    All concentrations in the same units; the units cancel through the ratio.

    Returns the Ki. It does NOT return an uncertainty: the conversion is exact
    given (IC50, [S], Km, mechanism), so the uncertainty in Ki comes from
    whatever you already have on those. Propagate it yourself, or use
    `ki_from_ic50_with_error`.
    """
    _require_positive(ic50=ic50)
    return ic50 / correction_factor(s_conc, km, model)


def ic50_from_ki(ki, s_conc, km, model="competitive"):
    """The forward direction: what IC50 would this Ki produce in that assay?"""
    _require_positive(ki=ki)
    return ki * correction_factor(s_conc, km, model)


def ki_from_ic50_with_error(ic50, ic50_se, s_conc, s_conc_se, km, km_se,
                            model="competitive"):
    """Propagate IC50, [S] and Km uncertainties through Cheng-Prusoff.

    Uses first-order (delta method) error propagation on

        Ki = IC50 / f,   f = 1 + [S]/Km

    which is exact to first order and adequate whenever the relative errors
    are modest. The result is a standard error, not a confidence interval: Ki
    is bounded below at zero, so a symmetric interval has the same defect
    described in fitting.py for Km. Use `bootstrap` for interval estimates if
    that matters.
    """
    _require_positive(ic50=ic50, km=km, s_conc=s_conc)
    f = correction_factor(s_conc, km, model)
    ki = ic50 / f

    df_ds = (1.0 / km) if model == "competitive" else (-km / s_conc ** 2)
    df_dkm = (-s_conc / km ** 2) if model == "competitive" else (1.0 / s_conc)
    if model == "noncompetitive":
        df_ds = df_dkm = 0.0

    # Ki = IC50 / f  ->  partials
    dki_dic50 = 1.0 / f
    dki_df = -ic50 / f ** 2

    var = (dki_dic50 * ic50_se) ** 2 \
        + (dki_df * df_ds * s_conc_se) ** 2 \
        + (dki_df * df_dkm * km_se) ** 2
    return ki, float(np.sqrt(max(var, 0.0)))


# --- IC50 curve fitting ------------------------------------------------------

def hill4(x, bottom, top, log_ic50, hill):
    """Four-parameter logistic, the standard dose-response form.

    Parameterised on log10(IC50) so the fit is well conditioned across six
    orders of magnitude, which is the normal span of an IC50 series.
    `hill` is the slope factor: 1 for a simple Michaelis-like response,
    >1 for a steep curve, <1 for a shallow one.
    """
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (1.0 + 10 ** ((log_ic50 - np.log10(x)) * hill))


def fit_ic50(conc, response, bottom=None, top=None, p0=None, maxfev=20000):
    """Fit a 4PL dose-response curve and return IC50 with a standard error.

    `conc` is inhibitor concentration (same units as you want the IC50 in),
    `response` is whatever you measured -- rate, signal, % inhibition. Both
    directions work (increasing or decreasing), since `bottom` and `top` are
    both free.

    Returns a dict with ic50, ic50_stderr, hill, top, bottom and r_squared.

    Note on what the standard error means: it comes from the covariance of the
    nonlinear fit, so it is a precision on this curve, not a statement about
    reproducibility across experiments. Those are different things and the
    second is usually larger.
    """
    x = np.asarray(conc, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.shape != y.shape:
        raise ValueError("conc and response must be the same length")
    if len(x) < 4:
        raise ValueError("need at least 4 points for a 4-parameter fit")
    if np.any(x <= 0):
        raise ValueError("concentrations must be positive (log scale)")
    if np.any(~np.isfinite(y)):
        raise ValueError("response must be finite")

    b0 = float(np.min(y)) if bottom is None else bottom
    t0 = float(np.max(y)) if top is None else top
    # initial IC50 guess: the concentration closest to the midpoint response
    mid = 0.5 * (b0 + t0)
    log_ic0 = float(np.log10(x[np.argmin(np.abs(y - mid))]))

    p_start = [b0, t0, log_ic0, 1.0] if p0 is None else list(p0)

    lo = [-np.inf, -np.inf, np.log10(x.min()) - 3, 0.05]
    hi = [np.inf, np.inf, np.log10(x.max()) + 3, 20.0]

    popt, pcov = curve_fit(hill4, x, y, p0=p_start, bounds=(lo, hi),
                           maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))

    pred = hill4(x, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "ic50": float(10 ** popt[2]),
        # log-scale SE -> relative SE in linear space, first order
        "ic50_stderr": float(10 ** popt[2] * np.log(10) * perr[2]),
        "log_ic50": float(popt[2]),
        "log_ic50_stderr": float(perr[2]),
        "hill": float(popt[3]),
        "bottom": float(popt[0]),
        "top": float(popt[1]),
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": len(x),
    }


# --- assumption checks -------------------------------------------------------

@dataclass
class AssumptionReport:
    model: str
    correction_factor: float
    s_over_km: float
    ic50: float
    enzyme_conc: float = None
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.warnings

    def summary(self):
        lines = [f"model={self.model}  [S]/Km={self.s_over_km:.2f}  "
                 f"correction={self.correction_factor:.2f}x"]
        if self.enzyme_conc:
            lines.append(f"  IC50/[E] = {self.ic50 / self.enzyme_conc:.1f}")
        if self.warnings:
            lines += [f"  WARNING: {w}" for w in self.warnings]
        else:
            lines.append("  no assumption violations detected")
        return "\n".join(lines)


def check_assumptions(ic50, s_conc, km, model="competitive",
                      enzyme_conc=None, total_conversion=None):
    """Report which Cheng-Prusoff assumptions are at risk.

    This does not decide for you. It flags:

      - a correction factor large enough that a Km error dominates the answer
      - IC50 within 10x of the enzyme concentration (tight binding)
      - substrate depletion during the measurement (total_conversion > 0.1)

    Pass enzyme_conc in the same units as IC50. Pass total_conversion as the
    fraction of substrate consumed by the end of the rate measurement, if you
    know it.
    """
    _require_positive(ic50=ic50, s_conc=s_conc, km=km)
    f = correction_factor(s_conc, km, model)
    s_over_km = s_conc / km
    warnings = []

    if f > 10:
        warnings.append(
            f"correction factor is {f:.1f}x: [S]/Km = {s_over_km:.1f} means "
            f"a 30% error in Km moves Ki by about 25%. Measure Km in the "
            f"same buffer and at the same temperature as the IC50."
        )

    if enzyme_conc is not None:
        _require_positive(enzyme_conc=enzyme_conc)
        if ic50 / enzyme_conc < 10:
            warnings.append(
                f"IC50 is only {ic50 / enzyme_conc:.1f}x the enzyme "
                f"concentration: this is tight binding, so the IC50 is partly "
                f"a titration of the enzyme and Cheng-Prusoff (which assumes "
                f"[I] >> [E]) underestimates Ki. Use a Morrison equation fit."
            )

    if total_conversion is not None and total_conversion > 0.10:
        warnings.append(
            f"substrate depletion is {total_conversion:.0%} over the rate "
            f"window: the free [S] is below [S]0, so the correction is "
            f"overestimated and Ki is pushed low. Use the mean free [S], or "
            f"fit progress curves."
        )

    return AssumptionReport(
        model=model, correction_factor=f, s_over_km=s_over_km, ic50=ic50,
        enzyme_conc=enzyme_conc, warnings=warnings,
    )


def tight_binding_threshold(enzyme_conc, factor=10.0):
    """The IC50 below which Cheng-Prusoff is invalid for a given [E].

    A convenient number to check an inhibitor series against: any compound
    whose IC50 lands below this cannot be converted without a Morrison fit,
    and its apparent rank among the others is unreliable.
    """
    _require_positive(enzyme_conc=enzyme_conc)
    return factor * enzyme_conc
