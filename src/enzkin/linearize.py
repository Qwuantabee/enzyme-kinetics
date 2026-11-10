"""The classic linear transforms, kept for comparison rather than for fitting.

These are still the plots everyone recognises, and they are genuinely useful
for *seeing* a mechanism (competitive inhibitors share a y-intercept,
uncompetitive ones give parallel lines). They are a bad way to get numbers
out, and the simulation in validate_fits.py shows how bad.

The reason is error propagation. If v has roughly constant error s, then
1/v has error s/v^2, so a point measured at low [S] -- where v is small and
the relative error is worst -- gets an enormous weight in the regression.
"""

import numpy as np


def lineweaver_burk(S, v):
    """1/v vs 1/S.  slope = km/vmax, intercept = 1/vmax."""
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = (S > 0) & (v > 0)
    return 1.0 / S[ok], 1.0 / v[ok]


def eadie_hofstee(S, v):
    """v vs v/S.  slope = -km, intercept = vmax.

    Better behaved than Lineweaver-Burk but v appears on both axes, so the
    errors in the two coordinates are correlated and ordinary regression
    is still not valid.
    """
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = S > 0
    return v[ok] / S[ok], v[ok]


def hanes_woolf(S, v):
    """S/v vs S.  slope = 1/vmax, intercept = km/vmax.

    Of the three this one distorts the error structure least, which is why
    it is the one to use if you are stuck doing this by hand.
    """
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = v > 0
    return S[ok], S[ok] / v[ok]


def params_from_lineweaver_burk(S, v):
    """Unweighted straight line through the double reciprocal plot.

    This is the "textbook" estimate. It is here to be compared against the
    nonlinear fit, not because it should be used.
    """
    x, y = lineweaver_burk(S, v)
    slope, intercept = np.polyfit(x, y, 1)
    vmax = 1.0 / intercept
    km = slope * vmax
    return {"vmax": float(vmax), "km": float(km)}


def params_from_hanes_woolf(S, v):
    x, y = hanes_woolf(S, v)
    slope, intercept = np.polyfit(x, y, 1)
    vmax = 1.0 / slope
    km = intercept * vmax
    return {"vmax": float(vmax), "km": float(km)}


def params_from_eadie_hofstee(S, v):
    x, y = eadie_hofstee(S, v)
    slope, intercept = np.polyfit(x, y, 1)
    return {"vmax": float(intercept), "km": float(-slope)}
