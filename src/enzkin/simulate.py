"""Synthetic datasets with known parameters, for checking the fitting code.

Any fitting routine will happily return numbers. The only way to know whether
those numbers mean anything is to feed it data where the true answer is known
and see whether it comes back, and whether the stated error bars actually
cover the truth at the rate they claim to.
"""

import numpy as np

from . import models


def substrate_series(vmax=100.0, km=25.0, s_max_multiple=10, n=8,
                     noise=0.03, noise_type="relative", seed=None,
                     replicates=1):
    """A single substrate titration.

    s_max_multiple: highest [S] as a multiple of km. Going to 10x Km gets you
    to about 91% of Vmax. Stopping at 1-2x Km, which is common when substrate
    is expensive or poorly soluble, makes Vmax nearly unidentifiable and the
    fit will say so through a large standard error.

    noise_type: "relative" scales with v, "absolute" does not.
    """
    rng = np.random.default_rng(seed)
    S = np.geomspace(km / 10, km * s_max_multiple, n)
    S = np.repeat(S, replicates)

    true = models.michaelis_menten(S, vmax, km)
    if noise_type == "relative":
        v = true * (1 + rng.normal(0, noise, size=true.shape))
    elif noise_type == "absolute":
        v = true + rng.normal(0, noise * vmax, size=true.shape)
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}")

    return S, np.clip(v, 1e-12, None), {"vmax": vmax, "km": km}


def inhibition_grid(mechanism="competitive", vmax=100.0, km=25.0,
                    ki=10.0, kis=40.0, inhibitor_levels=(0, 5, 15, 45),
                    n_substrate=7, noise=0.03, seed=None):
    """A full [S] x [I] grid, the design you'd actually run on a plate.

    Returns flat arrays so it can go straight into fit_inhibition.
    """
    rng = np.random.default_rng(seed)
    s_vals = np.geomspace(km / 8, km * 8, n_substrate)
    S, I = np.meshgrid(s_vals, np.asarray(inhibitor_levels, dtype=float))
    S, I = S.ravel(), I.ravel()

    if mechanism == "competitive":
        true = models.competitive(S, I, vmax, km, ki)
        truth = {"vmax": vmax, "km": km, "ki": ki}
    elif mechanism == "uncompetitive":
        true = models.uncompetitive(S, I, vmax, km, kis)
        truth = {"vmax": vmax, "km": km, "kis": kis}
    elif mechanism == "noncompetitive":
        true = models.noncompetitive(S, I, vmax, km, ki)
        truth = {"vmax": vmax, "km": km, "ki": ki}
    elif mechanism == "mixed":
        true = models.mixed(S, I, vmax, km, ki, kis)
        truth = {"vmax": vmax, "km": km, "ki": ki, "kis": kis}
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    v = true * (1 + rng.normal(0, noise, size=true.shape))
    return S, I, np.clip(v, 1e-12, None), truth
