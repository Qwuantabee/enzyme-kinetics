"""Recovery tests: simulate data from known parameters, fit it, check we get
the parameters back and that the mechanism is identified correctly.

Tolerances are loose on purpose. With 3% noise on 8 points you are not going
to recover Km to three decimal places, and a test that demanded it would just
be fragile. What matters is that the estimate is unbiased and the stated
error is honest -- test_se_covers_truth is the one that checks that.
"""

import numpy as np
import pytest

from enzkin import fitting, linearize, simulate


def test_recovers_mm_parameters():
    S, v, truth = simulate.substrate_series(vmax=120.0, km=40.0, noise=0.02, seed=1)
    fit = fitting.fit_michaelis_menten(S, v)
    assert fit.success
    assert fit.params["vmax"] == pytest.approx(truth["vmax"], rel=0.10)
    assert fit.params["km"] == pytest.approx(truth["km"], rel=0.15)
    assert fit.r_squared > 0.98


def test_noise_free_fit_is_near_exact():
    S, v, truth = simulate.substrate_series(vmax=75.0, km=12.0, noise=0.0, seed=0)
    fit = fitting.fit_michaelis_menten(S, v)
    assert fit.params["vmax"] == pytest.approx(truth["vmax"], rel=1e-6)
    assert fit.params["km"] == pytest.approx(truth["km"], rel=1e-6)


def test_se_covers_truth():
    """The 95% interval should contain the true value about 95% of the time.

    This is the test that catches standard errors which are technically
    computed but meaningless. It originally failed at 87% coverage, because
    ci95 was using 1.96 instead of the t critical value for 6 degrees of
    freedom. Keep the band tight enough that regressing to the normal
    approximation would fail it again.
    """
    t_hits = normal_hits = 0
    trials = 300
    for seed in range(trials):
        S, v, truth = simulate.substrate_series(vmax=100.0, km=25.0,
                                                noise=0.05, seed=seed)
        fit = fitting.fit_michaelis_menten(S, v)
        lo, hi = fit.ci95("km")
        t_hits += lo <= truth["km"] <= hi

        p, se = fit.params["km"], fit.stderr["km"]
        normal_hits += (p - 1.96 * se) <= truth["km"] <= (p + 1.96 * se)

    coverage = t_hits / trials
    # the t interval must beat the naive normal one, and clear a floor well
    # above the ~87% the normal version manages
    assert coverage > normal_hits / trials
    assert 0.88 <= coverage <= 0.99, f"coverage was {coverage:.2f}"


def test_truncated_substrate_range_inflates_vmax_error():
    """If you never get near saturation, Vmax is poorly determined and the
    fit should report that rather than quietly returning a confident number."""
    _, _, _ = simulate.substrate_series(seed=3)
    S_wide, v_wide, _ = simulate.substrate_series(s_max_multiple=20, noise=0.03, seed=3)
    S_narrow, v_narrow, _ = simulate.substrate_series(s_max_multiple=1, noise=0.03, seed=3)

    wide = fitting.fit_michaelis_menten(S_wide, v_wide)
    narrow = fitting.fit_michaelis_menten(S_narrow, v_narrow)

    rel_wide = wide.stderr["vmax"] / wide.params["vmax"]
    rel_narrow = narrow.stderr["vmax"] / narrow.params["vmax"]
    assert rel_narrow > rel_wide * 3


@pytest.mark.parametrize("mechanism", ["competitive", "uncompetitive",
                                       "noncompetitive", "mixed"])
def test_identifies_mechanism(mechanism):
    S, I, v, truth = simulate.inhibition_grid(mechanism=mechanism, noise=0.02,
                                              seed=7)
    results = fitting.compare_models(S, I, v)
    best = results[0]

    # mixed contains noncompetitive as a special case, so when the truth is
    # noncompetitive the mixed model can tie or very slightly win. Accept
    # either, but require the competitive/uncompetitive pair to be beaten.
    if mechanism == "noncompetitive":
        assert best.model in ("noncompetitive", "mixed")
    else:
        assert best.model == mechanism

    for name, true_val in truth.items():
        assert best.params[name] == pytest.approx(true_val, rel=0.25)


def test_wrong_mechanism_fits_worse():
    S, I, v, _ = simulate.inhibition_grid(mechanism="competitive", noise=0.01,
                                          seed=11)
    comp = fitting.fit_inhibition(S, I, v, model="competitive")
    uncomp = fitting.fit_inhibition(S, I, v, model="uncompetitive")
    assert comp.aicc < uncomp.aicc - 2


def test_lineweaver_burk_is_worse_than_nonlinear():
    """The practical argument against double reciprocal plots.

    Over many noisy datasets the LB estimate of Km should be both more biased
    and more variable than the nonlinear fit.
    """
    nl_err, lb_err = [], []
    for seed in range(150):
        S, v, truth = simulate.substrate_series(vmax=100.0, km=25.0,
                                                noise=0.05, seed=seed)
        nl = fitting.fit_michaelis_menten(S, v)
        lb = linearize.params_from_lineweaver_burk(S, v)
        nl_err.append(nl.params["km"] / truth["km"] - 1)
        lb_err.append(lb["km"] / truth["km"] - 1)

    assert np.std(lb_err) > np.std(nl_err)


def test_rejects_mismatched_input():
    with pytest.raises(ValueError):
        fitting.fit_michaelis_menten([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        fitting.fit_michaelis_menten([1, -2, 3], [1, 2, 3])


def test_rejects_unknown_model():
    S, I, v, _ = simulate.inhibition_grid(seed=2)
    with pytest.raises(ValueError):
        fitting.fit_inhibition(S, I, v, model="allosteric")


def test_bootstrap_interval_is_sane():
    S, v, truth = simulate.substrate_series(vmax=100.0, km=25.0, noise=0.05,
                                            n=10, seed=5)
    ci = fitting.bootstrap_ci(S, v, n_boot=400, seed=5)
    lo, hi = ci["km"]
    assert lo < truth["km"] < hi
    assert lo > 0
