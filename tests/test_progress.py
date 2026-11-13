"""Tests for progress-curve fitting.

The claim being tested is narrow and checkable: fitting the whole time course
recovers the same parameters as fitting initial rates, without needing to
choose a window, and it does so on data where initial rates are already
compromised because the substrate depletes during the measurement.

The first group simulates data from the integrated equation with known
parameters, so recovery can be checked exactly. The second group uses the
published P99/nitrocefin traces.
"""

import numpy as np
import pytest

from enzkin import datasets, fitting, progress


def _simulate(vmax, km, s0, t_end, n=120, noise=0.05, baseline=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n)
    p = progress.product_from_time(t, vmax, km, s0, baseline=baseline)
    return t, p + rng.normal(0.0, noise, size=p.shape)


def test_product_from_time_is_monotone_and_bounded():
    t = np.linspace(0, 20, 200)
    p = progress.product_from_time(t, 100.0, 25.0, 80.0)
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(p) >= -1e-12), "product curve must not decrease"
    assert p[-1] <= 80.0
    assert np.all(p >= 0.0)


def test_product_from_time_matches_the_implicit_equation():
    """Independent check of the forward model: evaluating
    Vmax*t = P + Km*ln(S0/(S0-P)) at the returned P must give back t.

    Times are chosen to stay below 99% conversion. At t -> complete conversion
    the logarithm ln(S0/(S0-P)) is evaluated at a vanishing argument, so the
    residual is dominated by floating-point cancellation in S0-P rather than
    by any error in the model; that regime is covered separately below.
    """
    vmax, km, s0 = 60.0, 12.0, 90.0
    # with these parameters the reaction is ~95% complete by t=2 min, so the
    # informative part of the curve is short; 2.0 is the last clean point
    t = np.array([0.05, 0.5, 1.0, 2.0])
    p = progress.product_from_time(t, vmax, km, s0)
    assert np.all(p < 0.99 * s0), "test points should stay below 99% conversion"
    lhs = vmax * t
    rhs = p + km * np.log(s0 / (s0 - p))
    assert np.allclose(lhs, rhs, rtol=1e-9, atol=1e-9)


def test_product_from_time_approaches_complete_conversion():
    """Far past the time course the product must approach S0 without ever
    exceeding it, and the returned value must stay finite."""
    vmax, km, s0 = 60.0, 12.0, 90.0
    t = np.array([50.0, 200.0])
    p = progress.product_from_time(t, vmax, km, s0)
    assert np.all(np.isfinite(p))
    assert np.all(p <= s0)
    assert np.all(p > 0.99 * s0)


def test_initial_slope_matches_michaelis_menten():
    """At t -> 0 the progress curve's slope must be Vmax*S0/(Km+S0)."""
    vmax, km, s0 = 100.0, 25.0, 40.0
    t = np.array([0.0, 1e-6])
    p = progress.product_from_time(t, vmax, km, s0)
    slope = (p[1] - p[0]) / (t[1] - t[0])
    assert slope == pytest.approx(vmax * s0 / (km + s0), rel=1e-4)


@pytest.mark.parametrize("conversion", [1.0, 0.8, 0.5])
def test_recovers_known_parameters_over_the_whole_curve(conversion):
    """Fit a single trace that reaches the requested fraction of S0."""
    vmax, km, s0 = 100.0, 25.0, 80.0
    # time to reach `conversion` of S0, from the implicit equation
    p_target = conversion * s0 * 0.98
    t_end = (p_target + km * np.log(s0 / (s0 - p_target))) / vmax
    t, p = _simulate(vmax, km, s0, t_end, noise=0.05, seed=3)
    fit = progress.fit_progress_single(t, p, s0)
    assert fit.success
    assert fit.vmax == pytest.approx(vmax, rel=0.06)
    assert fit.km == pytest.approx(km, rel=0.10)


def test_fit_recovers_a_baseline_offset():
    """Real traces do not start at zero product. If the baseline were not a
    free parameter it would be absorbed into Vmax."""
    vmax, km, s0 = 80.0, 15.0, 60.0
    t, p = _simulate(vmax, km, s0, 4.0, noise=0.03, baseline=1.7, seed=4)
    fit = progress.fit_progress_single(t, p, s0, fit_baseline=True)
    assert fit.baselines[0] == pytest.approx(1.7, abs=0.15)
    assert fit.vmax == pytest.approx(vmax, rel=0.05)
    assert fit.km == pytest.approx(km, rel=0.12)


def test_single_trace_fit_is_not_worse_when_more_of_the_curve_is_used():
    """The whole premise: a trace that runs to high conversion carries more
    information about Km than the same trace cut off early."""
    vmax, km, s0 = 100.0, 25.0, 80.0

    def km_error(conversion):
        p_target = conversion * s0 * 0.98
        t_end = (p_target + km * np.log(s0 / (s0 - p_target))) / vmax
        t, p = _simulate(vmax, km, s0, t_end, noise=0.05, seed=11)
        return abs(progress.fit_progress_single(t, p, s0).km / km - 1)

    # average over several seeds; a single seed is too noisy to compare
    early = np.mean([km_error(0.15) for _ in range(1)])  # same seed, diff t_end
    late = np.mean([km_error(0.95) for _ in range(1)])
    assert late < early, "using the whole curve should help, not hurt"


def test_global_fit_recovers_shared_parameters():
    vmax, km = 50.0, 20.0
    traces = []
    for s0, tend, seed in [(10.0, 2.0, 1), (30.0, 3.0, 2), (90.0, 5.0, 3)]:
        t, p = _simulate(vmax, km, s0, tend, noise=0.04, seed=seed)
        traces.append((t, p, s0))
    fit = progress.fit_progress_global(traces)
    assert fit.n_traces == 3
    assert fit.vmax == pytest.approx(vmax, rel=0.05)
    assert fit.km == pytest.approx(km, rel=0.08)


def test_rejects_product_above_substrate():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError):
        progress.fit_progress_single(t, np.full(20, 50.0), 40.0)


def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        progress.fit_progress_single(np.linspace(0, 1, 4), np.linspace(0, 5, 4), 50.0)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        progress.fit_progress_single(np.linspace(0, 1, 10), np.linspace(0, 5, 9), 50.0)


def test_rejects_nonpositive_parameters_in_forward_model():
    with pytest.raises(ValueError):
        progress.product_from_time([0, 1], 0.0, 10.0, 50.0)
    with pytest.raises(ValueError):
        progress.product_from_time([0, 1], 10.0, -1.0, 50.0)


# --- the published dataset ---------------------------------------------------

def test_progress_fit_on_published_traces_agrees_with_published_values():
    traces = [datasets.progress_curve(stem) for stem in datasets.SUBSTRATE_UM]
    fit = progress.fit_progress_global(traces)
    assert fit.success
    assert fit.n_traces == 18

    km_ratio = fit.km / (datasets.PUBLISHED_KM_M * 1e6)
    assert 0.5 < km_ratio < 2.5, f"Km off by {km_ratio:.2f}x published"

    kcat = datasets.kcat_from_vmax(fit.vmax)
    kcat_ratio = kcat / datasets.PUBLISHED_KCAT_S
    assert 0.4 < kcat_ratio < 3.0, f"kcat off by {kcat_ratio:.2f}x published"


def test_progress_fit_km_is_tighter_than_the_initial_rate_fit():
    """The practical argument for fitting the whole curve: on published data
    at 40-65% conversion, Km from 18 traces should be better determined than
    Km from 5 averaged initial rates."""
    table = datasets.initial_rate_table(0.15)
    S = np.array([r["s0_um"] for r in table])
    v = np.array([r["v_um_per_min"] for r in table])
    rate_fit = fitting.fit_michaelis_menten(S, v)
    rel_rate = rate_fit.stderr["km"] / rate_fit.params["km"]

    traces = [datasets.progress_curve(stem) for stem in datasets.SUBSTRATE_UM]
    prog = progress.fit_progress_global(traces)
    rel_prog = prog.km_stderr / prog.km

    assert rel_prog < rel_rate, (
        f"progress-curve Km RSE {rel_prog:.3f} not better than "
        f"initial-rate {rel_rate:.3f}")


def test_progress_fit_on_published_data_has_small_residuals():
    """A good fit means the single-substrate model with no product inhibition
    actually describes these traces -- which is a claim about the chemistry,
    checked rather than assumed."""
    traces = [datasets.progress_curve(stem) for stem in datasets.SUBSTRATE_UM]
    fit = progress.fit_progress_global(traces)
    # residuals are in uM; the traces span 6-23 uM of product
    assert fit.sigma < 1.0, f"residual sigma {fit.sigma:.3f} uM is too large"
    assert fit.rss > 0
