"""Steady-state rate equations for enzyme kinetics.

Everything here assumes the usual steady-state (Briggs-Haldane) treatment:
[S] >> [E], initial rates only, and no product inhibition because we're
looking at the early part of the progress curve.

Symbols:
    S     substrate concentration
    I     inhibitor concentration
    vmax  maximum velocity (same units as the measured rate)
    km    Michaelis constant (concentration at half vmax)
    ki    dissociation constant for inhibitor binding free enzyme (E + I)
    kis   dissociation constant for inhibitor binding the ES complex
"""

import numpy as np


def michaelis_menten(S, vmax, km):
    """v = vmax * S / (km + S)"""
    S = np.asarray(S, dtype=float)
    return vmax * S / (km + S)


def competitive(S, I, vmax, km, ki):
    """Inhibitor binds free enzyme only.

    Km goes up by (1 + I/Ki), Vmax is untouched -- high substrate
    outcompetes the inhibitor.
    """
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    return vmax * S / (km * (1.0 + I / ki) + S)


def uncompetitive(S, I, vmax, km, kis):
    """Inhibitor binds the ES complex only.

    Both Vmax and Km drop by the same factor, so Vmax/Km is unchanged.
    This is the one that looks like parallel lines on a Lineweaver-Burk.
    """
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    return vmax * S / (km + S * (1.0 + I / kis))


def noncompetitive(S, I, vmax, km, ki):
    """Pure noncompetitive: inhibitor binds E and ES with equal affinity.

    Special case of mixed with ki == kis. Vmax falls, Km is unchanged.
    """
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    return vmax * S / ((km + S) * (1.0 + I / ki))


def mixed(S, I, vmax, km, ki, kis):
    """Inhibitor binds both E and ES, with different affinities.

    ki  < kis -> mostly competitive in character
    ki  > kis -> mostly uncompetitive
    ki == kis -> pure noncompetitive
    """
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    return vmax * S / (km * (1.0 + I / ki) + S * (1.0 + I / kis))


def substrate_inhibition(S, vmax, km, ksi):
    """Rate falls off at high [S] because a second substrate binds ES.

    Worth checking whenever the v vs S curve turns over instead of
    plateauing -- it gets misread as a bad assay more often than it should.
    """
    S = np.asarray(S, dtype=float)
    return vmax * S / (km + S * (1.0 + S / ksi))


def hill(S, vmax, k_half, n):
    """Empirical Hill equation for cooperative enzymes.

    n > 1 positive cooperativity, n < 1 negative. k_half is the half
    saturating concentration, not a Km -- it is not a dissociation constant
    and shouldn't be reported as one.
    """
    S = np.asarray(S, dtype=float)
    return vmax * S**n / (k_half**n + S**n)


# parameter names in the order the fitting code passes them
INHIBITION_MODELS = {
    "competitive": (competitive, ("vmax", "km", "ki")),
    "uncompetitive": (uncompetitive, ("vmax", "km", "kis")),
    "noncompetitive": (noncompetitive, ("vmax", "km", "ki")),
    "mixed": (mixed, ("vmax", "km", "ki", "kis")),
}

SUBSTRATE_MODELS = {
    "michaelis_menten": (michaelis_menten, ("vmax", "km")),
    "substrate_inhibition": (substrate_inhibition, ("vmax", "km", "ksi")),
    "hill": (hill, ("vmax", "k_half", "n")),
}


def catalytic_efficiency(vmax, km, enzyme_conc):
    """kcat and kcat/Km.

    enzyme_conc has to be the active site concentration in the same units
    as whatever vmax is per. If you only have total protein this number is
    an underestimate and should be labelled as such.
    """
    kcat = vmax / enzyme_conc
    return kcat, kcat / km
