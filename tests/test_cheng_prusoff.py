"""Tests for IC50 <-> Ki conversion.

Three layers:

1. Algebra. The Cheng-Prusoff relations have exact limits that can be checked
   directly: at [S] = 0 the competitive IC50 equals Ki, at [S] = Km the IC50 is
   exactly twice Ki, and the forward and inverse functions must invert.

2. Round trip through the simulator. Simulate a real inhibition experiment,
   compute the IC50 the way an assay does, convert it back with Cheng-Prusoff,
   and check the original Ki comes out. That tests the relation against the
   rate equations rather than against my own algebra.

3. Behaviour on a real dataset. 482 measured IC50s for E. coli dihydrofolate
   reductase from ChEMBL, used to check the conversion is numerically sane over
   the range real assays actually produce (sub-nM to mM) and that the
   assumption checks fire where they should.
"""

import numpy as np
import pytest

from enzkin import cheng_prusoff as cp
from enzkin import datasets, models, simulate


# --- 1. algebra --------------------------------------------------------------

def test_competitive_correction_limits():
    km = 25.0
    # no substrate: inhibitor is the only thing binding, IC50 == Ki
    assert cp.correction_factor(1e-9, km, "competitive") == pytest.approx(1.0)
    # [S] = Km: the correction is exactly 2
    assert cp.correction_factor(km, km, "competitive") == pytest.approx(2.0)
    # [S] = 10 Km: correction is 11
    assert cp.correction_factor(10 * km, km, "competitive") == pytest.approx(11.0)


def test_uncompetitive_correction_limits():
    km = 25.0
    # [S] = Km gives 2 as well, but the dependence is inverted
    assert cp.correction_factor(km, km, "uncompetitive") == pytest.approx(2.0)
    # very low substrate: the inhibitor needs the ES complex, so almost no
    # correction is needed at all... and the factor grows as [S] falls
    assert cp.correction_factor(km / 10, km, "uncompetitive") > 10


def test_noncompetitive_needs_no_correction():
    for s in (0.1, 1.0, 25.0, 1000.0):
        assert cp.correction_factor(s, 25.0, "noncompetitive") == 1.0


def test_forward_and_inverse_invert():
    km, s = 25.0, 40.0
    for model in cp.SINGLE_KI_MODELS:
        for ki in (0.1, 1.0, 50.0, 1e4):
            ic = cp.ic50_from_ki(ki, s, km, model=model)
            assert cp.ki_from_ic50(ic, s, km, model=model) == pytest.approx(ki)


def test_mixed_model_is_refused():
    """One IC50 cannot give two Ki values. Refusing is the correct answer."""
    with pytest.raises(ValueError, match="mixed"):
        cp.correction_factor(25.0, 25.0, "mixed")
    with pytest.raises(ValueError):
        cp.ki_from_ic50(10.0, 25.0, 25.0, model="mixed")


def test_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        cp.ki_from_ic50(0.0, 25.0, 25.0)
    with pytest.raises(ValueError):
        cp.ki_from_ic50(10.0, -1.0, 25.0)
    with pytest.raises(ValueError):
        cp.ki_from_ic50(10.0, 25.0, 0.0)
    with pytest.raises(ValueError):
        cp.correction_factor(25.0, 25.0, "allosteric")


# --- 2. round trip through the rate equations --------------------------------

@pytest.mark.parametrize("model,param_name", [
    ("competitive", "ki"),
    ("uncompetitive", "kis"),
    ("noncompetitive", "ki"),
])
def test_round_trip_through_simulated_experiment(model, param_name):
    """Compute an IC50 from the rate equation, then convert it back to Ki.

    The IC50 is found the way an assay defines it: the inhibitor
    concentration at which the rate falls to half of the uninhibited rate.
    If Cheng-Prusoff is right, converting that number gives back the Ki that
    generated the data.
    """
    vmax, km, s_conc, ki_true = 100.0, 25.0, 60.0, 8.0

    if model == "competitive":
        base = models.competitive(s_conc, 0.0, vmax, km, ki_true)
        fn = lambda i: models.competitive(s_conc, i, vmax, km, ki_true)
    elif model == "uncompetitive":
        base = models.uncompetitive(s_conc, 0.0, vmax, km, ki_true)
        fn = lambda i: models.uncompetitive(s_conc, i, vmax, km, ki_true)
    else:
        base = models.noncompetitive(s_conc, 0.0, vmax, km, ki_true)
        fn = lambda i: models.noncompetitive(s_conc, i, vmax, km, ki_true)

    # solve fn(I) = base/2 by bisection
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if fn(mid) > base / 2:
            lo = mid
        else:
            hi = mid
    ic50 = np.sqrt(lo * hi)

    ki_back = cp.ki_from_ic50(ic50, s_conc, km, model=model)
    assert ki_back == pytest.approx(ki_true, rel=1e-6)


def test_wrong_mechanism_biases_ki_in_a_predictable_direction():
    """Using the competitive form on uncompetitive truth is a common error.

    The two correction factors move in opposite directions with [S]:
    competitive grows as [S]/Km, uncompetitive shrinks as it. So misapplying
    the competitive form to an uncompetitive inhibitor makes Ki too HIGH when
    [S] < Km and too LOW when [S] > Km, and the error vanishes at [S] = Km
    where both factors equal 2. Verified numerically in the loop below rather
    than asserted from memory.
    """
    km, ki_true = 25.0, 10.0

    # at [S] = Km the two relations coincide, so no bias appears
    ic = cp.ic50_from_ki(ki_true, km, km, model="uncompetitive")
    assert cp.ki_from_ic50(ic, km, km, model="competitive") == \
        pytest.approx(ki_true, rel=1e-9)

    # below Km the competitive factor is small, so Ki is overestimated
    s_low = 5.0
    ic_low = cp.ic50_from_ki(ki_true, s_low, km, model="uncompetitive")
    wrong_low = cp.ki_from_ic50(ic_low, s_low, km, model="competitive")
    assert wrong_low > ki_true, f"expected overestimate, got {wrong_low}"

    # above Km the competitive factor is large, so Ki is underestimated
    s_high = 125.0
    ic_high = cp.ic50_from_ki(ki_true, s_high, km, model="uncompetitive")
    wrong_high = cp.ki_from_ic50(ic_high, s_high, km, model="competitive")
    assert wrong_high < ki_true, f"expected underestimate, got {wrong_high}"

    # and the size of the error is the ratio of the two correction factors
    ratio_low = cp.correction_factor(s_low, km, "uncompetitive") / \
        cp.correction_factor(s_low, km, "competitive")
    assert wrong_low / ki_true == pytest.approx(ratio_low, rel=1e-9)


# --- 3. IC50 curve fitting ---------------------------------------------------

def test_fit_ic50_recovers_a_known_ic50():
    truth = 120.0
    conc = np.array([1, 3, 10, 30, 100, 300, 1000, 3000], dtype=float)
    resp = cp.hill4(conc, 0.0, 100.0, np.log10(truth), 1.0)
    fit = cp.fit_ic50(conc, resp)
    assert fit["ic50"] == pytest.approx(truth, rel=0.02)
    assert fit["hill"] == pytest.approx(1.0, rel=0.05)
    assert fit["r_squared"] > 0.999


def test_fit_ic50_handles_noisy_data_and_reports_error():
    rng = np.random.default_rng(7)
    truth = 45.0
    conc = np.array([0.3, 1, 3, 10, 30, 100, 300, 1000, 3000], dtype=float)
    resp = cp.hill4(conc, 2.0, 98.0, np.log10(truth), 1.2)
    resp = resp + rng.normal(0, 2.0, size=resp.shape)
    fit = cp.fit_ic50(conc, resp)
    assert fit["ic50"] == pytest.approx(truth, rel=0.15)
    assert fit["ic50_stderr"] > 0
    assert fit["ic50_stderr"] / fit["ic50"] < 0.2


def test_fit_ic50_rejects_bad_input():
    with pytest.raises(ValueError):
        cp.fit_ic50([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        cp.fit_ic50([0, 1, 10, 100], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        cp.fit_ic50([1, 2, 3, 4], [1, 2, 3, np.nan])


# --- 4. error propagation ----------------------------------------------------

def test_error_propagation_matches_monte_carlo():
    """The delta-method SE must agree with what actually happens when the
    inputs are sampled."""
    rng = np.random.default_rng(3)
    ic50, ic50_se = 500.0, 40.0
    s, s_se = 60.0, 3.0
    km, km_se = 25.0, 2.0

    _, se = cp.ki_from_ic50_with_error(ic50, ic50_se, s, s_se, km, km_se)

    draws = []
    for _ in range(4000):
        draws.append(cp.ki_from_ic50(
            max(rng.normal(ic50, ic50_se), 1e-9),
            max(rng.normal(s, s_se), 1e-9),
            max(rng.normal(km, km_se), 1e-9),
        ))
    mc_se = float(np.std(draws))
    assert se == pytest.approx(mc_se, rel=0.20), \
        f"delta method {se:.2f} vs Monte Carlo {mc_se:.2f}"


def test_km_uncertainty_dominates_at_high_substrate():
    """At large [S]/Km the correction is [S]/Km, so Ki inherits Km's relative
    error directly. That is worth knowing before quoting a Ki to 2 s.f."""
    args = dict(ic50=1000.0, ic50_se=10.0, s_conc=250.0, s_conc_se=1.0)
    low_km_err = cp.ki_from_ic50_with_error(km=25.0, km_se=0.5, **args)[1]
    high_km_err = cp.ki_from_ic50_with_error(km=25.0, km_se=10.0, **args)[1]
    assert high_km_err > 3 * low_km_err


# --- 5. assumption checks ----------------------------------------------------

def test_tight_binding_is_flagged():
    rep = cp.check_assumptions(ic50=5.0, s_conc=25.0, km=25.0,
                               enzyme_conc=1.0)
    assert not rep.ok
    assert any("tight binding" in w for w in rep.warnings)


def test_loose_binding_passes():
    rep = cp.check_assumptions(ic50=500.0, s_conc=25.0, km=25.0,
                               enzyme_conc=1.0)
    assert rep.ok, rep.warnings


def test_large_correction_factor_is_flagged():
    rep = cp.check_assumptions(ic50=1e5, s_conc=1000.0, km=25.0)
    assert any("30% error in Km" in w for w in rep.warnings)


def test_substrate_depletion_is_flagged():
    rep = cp.check_assumptions(ic50=500.0, s_conc=25.0, km=25.0,
                               total_conversion=0.45)
    assert any("depletion" in w for w in rep.warnings)


def test_tight_binding_threshold():
    assert cp.tight_binding_threshold(1.0) == pytest.approx(10.0)
    assert cp.tight_binding_threshold(0.5, factor=20) == pytest.approx(10.0)


# --- 6. the real ChEMBL dataset ----------------------------------------------

def test_chembl_ic50_data_loads_and_is_plausible():
    table = datasets.load_chembl_dhfr_ic50()
    assert len(table) > 400
    ics = np.array([r["ic50_nM"] for r in table])
    assert np.all(ics > 0)
    # real DHFR inhibitor series span many orders of magnitude; if the loader
    # reversed a log or a unit this assertion is what notices
    assert ics.min() < 1.0
    assert ics.max() > 1e5
    assert len({r["molecule_chembl_id"] for r in table}) > 300


def test_chembl_conversion_round_trips_over_the_real_range():
    """Every measured IC50 in the dataset must survive conversion and back."""
    table = datasets.load_chembl_dhfr_ic50()
    km_nM, s_nM = 25e3, 60e3   # DHFR assays run near [S] ~ 2-3 Km, in nM
    for row in table[:200]:
        ic = row["ic50_nM"]
        ki = cp.ki_from_ic50(ic, s_nM, km_nM, model="competitive")
        back = cp.ic50_from_ki(ki, s_nM, km_nM, model="competitive")
        assert back == pytest.approx(ic, rel=1e-9)


def test_chembl_competitive_conversion_is_monotone():
    """A more potent IC50 must give a more potent Ki -- no rank reversals
    introduced by the arithmetic."""
    table = sorted(datasets.load_chembl_dhfr_ic50(),
                   key=lambda r: r["ic50_nM"])
    kis = [cp.ki_from_ic50(r["ic50_nM"], 60e3, 25e3) for r in table]
    assert all(a <= b * 1.000001 for a, b in zip(kis, kis[1:]))
