"""Direction and size of the Cheng-Prusoff error terms, pinned numerically.

The module docstring once said the wrong-mechanism bias is
(1 + S/Km)(1 + Km/S) (a factor 4 at S = Km) and the tight-binding warning said
Cheng-Prusoff "underestimates" Ki. Both were wrong: the wrong-mechanism bias is
exactly Km/S (no bias at S = Km), and under tight binding the plain conversion
OVERestimates Ki. These tests derive each number from the underlying kinetics,
not from the module's own formulas.
"""

import pytest
from scipy.optimize import brentq

from enzkin import cheng_prusoff as cp


@pytest.mark.parametrize("s_over_km, expected", [(0.1, 10.0), (1.0, 1.0), (10.0, 0.1)])
def test_competitive_form_on_an_uncompetitive_inhibitor_is_off_by_km_over_s(
        s_over_km, expected):
    km, ki_true = 25.0, 10.0
    s = s_over_km * km
    ic50 = cp.ic50_from_ki(ki_true, s, km, model="uncompetitive")
    wrong = cp.ki_from_ic50(ic50, s, km, model="competitive")
    assert wrong / ki_true == pytest.approx(km / s, rel=1e-12)
    assert wrong / ki_true == pytest.approx(expected, rel=1e-12)
    # and it is NOT the (1+S/Km)(1+Km/S) the docstring once claimed
    claimed = (1 + s / km) * (1 + km / s)
    assert abs(wrong / ki_true - claimed) > 0.5


@pytest.mark.parametrize("s_over_km", [0.1, 1.0, 10.0])
def test_uncompetitive_form_on_a_competitive_inhibitor_is_off_by_s_over_km(s_over_km):
    km, ki_true = 25.0, 10.0
    s = s_over_km * km
    ic50 = cp.ic50_from_ki(ki_true, s, km, model="competitive")
    wrong = cp.ki_from_ic50(ic50, s, km, model="uncompetitive")
    assert wrong / ki_true == pytest.approx(s / km, rel=1e-12)


def test_the_docstring_states_the_corrected_bias():
    doc = cp.__doc__
    assert "Km/[S] exactly" in doc
    assert "(1+[S]/Km)(1+Km/[S])" not in doc
    assert "NO bias" in doc


def _exact_tight_binding_ic50(ki, s, km, e_tot):
    """IC50 from the exact (Morrison) quadratic, found by root finding -- not
    from the closed form being tested."""
    k_app = ki * (1.0 + s / km)

    def frac_active(i):
        b = e_tot + i + k_app
        bound = (b - (b * b - 4.0 * e_tot * i) ** 0.5) / 2.0
        return 1.0 - bound / e_tot

    return brentq(lambda i: frac_active(i) - 0.5, 1e-9, 1e6)


def test_tight_binding_makes_cheng_prusoff_overestimate_ki():
    ki, s, km, e_tot = 1.0, 1.0, 1.0, 10.0
    ic50 = _exact_tight_binding_ic50(ki, s, km, e_tot)
    assert ic50 == pytest.approx(7.0, rel=1e-6)   # Ki*(1+S/Km) + E/2 = 2 + 5
    ki_cp = cp.ki_from_ic50(ic50, s, km, model="competitive")
    assert ki_cp == pytest.approx(3.5, rel=1e-6)
    assert ki_cp > ki, "Cheng-Prusoff OVERestimates Ki under tight binding"
    # closed form: Ki + E / (2 f)
    assert ki_cp == pytest.approx(ki + e_tot / (2.0 * (1.0 + s / km)), rel=1e-6)


def test_tight_binding_warning_names_the_right_direction():
    rep = cp.check_assumptions(ic50=7.0, s_conc=1.0, km=1.0, enzyme_conc=10.0 / 10.0 * 1.0)
    text = " ".join(rep.warnings)
    assert "tight binding" in text
    assert "OVERestimates" in text
    assert "underestimates" not in text.lower()


def test_km_error_numbers_in_the_docstring_are_what_the_arithmetic_gives():
    km, s, ic50 = 25.0, 250.0, 1000.0           # S = 10 Km, factor 11
    f_true = cp.correction_factor(s, km, "competitive")
    assert f_true == pytest.approx(11.0)
    ki_true = ic50 / f_true
    high = cp.ki_from_ic50(ic50, s, km * 1.3) / ki_true   # Km 30% too high
    low = cp.ki_from_ic50(ic50, s, km * 0.7) / ki_true    # Km 30% too low
    assert cp.correction_factor(s, km * 1.3, "competitive") == pytest.approx(8.692, abs=1e-3)
    assert cp.correction_factor(s, km * 0.7, "competitive") == pytest.approx(15.286, abs=1e-3)
    assert high - 1 == pytest.approx(0.27, abs=0.005)
    assert low - 1 == pytest.approx(-0.28, abs=0.005)
    assert "+27%" in cp.check_assumptions(ic50=ic50, s_conc=s, km=km).warnings[0]
    assert "about 25%" not in cp.__doc__


def test_every_chembl_ic50_round_trips_not_just_the_first_200():
    from enzkin import datasets
    table = datasets.load_chembl_dhfr_ic50()
    assert len(table) > 200
    for row in table:
        ki = cp.ki_from_ic50(row["ic50_nM"], 60e3, 25e3)
        assert cp.ic50_from_ki(ki, 60e3, 25e3) == pytest.approx(row["ic50_nM"], rel=1e-9)
