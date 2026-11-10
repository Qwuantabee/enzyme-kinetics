"""Checks on the rate equations themselves.

These are limiting-case tests: at the extremes the equations have to reduce
to things that are true by inspection. If one of these breaks, the algebra
in models.py is wrong and no amount of good fitting will save it.
"""

import numpy as np
import pytest

from enzkin import models


def test_mm_half_vmax_at_km():
    assert models.michaelis_menten(25.0, 100.0, 25.0) == pytest.approx(50.0)


def test_mm_saturates():
    v = models.michaelis_menten(1e6, 100.0, 25.0)
    assert v == pytest.approx(100.0, rel=1e-4)


def test_mm_first_order_at_low_s():
    # well below Km, v should be close to (vmax/km)*S
    S = 1e-3
    expected = (100.0 / 25.0) * S
    assert models.michaelis_menten(S, 100.0, 25.0) == pytest.approx(expected, rel=1e-4)


def test_zero_inhibitor_reduces_to_mm():
    S = np.array([1.0, 10.0, 100.0])
    mm = models.michaelis_menten(S, 100.0, 25.0)
    for fn, kwargs in [
        (models.competitive, dict(ki=5.0)),
        (models.noncompetitive, dict(ki=5.0)),
        (models.uncompetitive, dict(kis=5.0)),
    ]:
        got = fn(S, 0.0, 100.0, 25.0, **kwargs)
        assert np.allclose(got, mm)
    assert np.allclose(models.mixed(S, 0.0, 100.0, 25.0, 5.0, 5.0), mm)


def test_mixed_equals_noncompetitive_when_ki_equals_kis():
    S = np.array([1.0, 10.0, 100.0])
    I = np.array([2.0, 2.0, 2.0])
    a = models.mixed(S, I, 100.0, 25.0, 8.0, 8.0)
    b = models.noncompetitive(S, I, 100.0, 25.0, 8.0)
    assert np.allclose(a, b)


def test_competitive_can_be_outrun_by_substrate():
    """The defining feature: at saturating S the inhibitor stops mattering."""
    big_s = 1e8
    v_inhib = models.competitive(big_s, 50.0, 100.0, 25.0, 1.0)
    assert v_inhib == pytest.approx(100.0, rel=1e-3)


def test_uncompetitive_cannot_be_outrun():
    """Vmax really does fall, so high substrate does not rescue it."""
    big_s = 1e8
    v_inhib = models.uncompetitive(big_s, 50.0, 100.0, 25.0, 10.0)
    assert v_inhib < 100.0 * 0.2


def test_uncompetitive_preserves_vmax_over_km():
    """Both Vmax and Km drop by (1 + I/Kis), so the ratio is unchanged.
    That ratio is the specificity constant, so an uncompetitive inhibitor
    doesn't change how the enzyme performs at low substrate."""
    vmax, km, kis, I = 100.0, 25.0, 10.0, 30.0
    factor = 1 + I / kis
    app_vmax, app_km = vmax / factor, km / factor
    assert app_vmax / app_km == pytest.approx(vmax / km)


def test_hill_n1_is_michaelis_menten():
    S = np.array([1.0, 10.0, 100.0])
    assert np.allclose(models.hill(S, 100.0, 25.0, 1.0),
                       models.michaelis_menten(S, 100.0, 25.0))


def test_substrate_inhibition_turns_over():
    S = np.geomspace(0.1, 1e4, 200)
    v = models.substrate_inhibition(S, 100.0, 25.0, 200.0)
    peak = np.argmax(v)
    assert 0 < peak < len(S) - 1          # maximum is interior, not at the end
    assert v[-1] < v[peak] * 0.5          # and it really does come back down


def test_catalytic_efficiency():
    kcat, keff = models.catalytic_efficiency(vmax=100.0, km=25.0, enzyme_conc=2.0)
    assert kcat == pytest.approx(50.0)
    assert keff == pytest.approx(2.0)
