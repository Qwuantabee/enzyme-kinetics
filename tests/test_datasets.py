"""Tests against a real published dataset.

These are different in kind from the recovery tests in test_fitting.py. There
the truth was known because it was invented. Here the numbers are measurements
of a real enzyme, and the only external benchmark is what the authors
published for this system.

Data: Frere, Verlaine & Matagne (2023) Scientific Reports 13:14994,
doi:10.1038/s41598-023-41805-y; archive doi:10.5281/zenodo.10059024 (CC-BY).
System: Enterobacter cloacae P99 class C beta-lactamase + nitrocefin, 462 nm.

What these tests are for: the loader must read the archive's awkward number
format correctly, the derived quantities must be dimensionally right, and the
fits must land in the neighbourhood of the published values. The last is a
loose check by design -- a real enzyme preparation is not the literature
preparation, and a test that demanded the published Km to 5% would be
testing the protein, not the code.
"""

import numpy as np
import pytest

from enzkin import datasets, fitting, progress


def test_parses_instrument_thousands_dot_format():
    """The archive writes '1.008.333' for 1.008333 min -- a dot used as a
    thousands separator on top of the decimal point. `float()` raises
    ValueError on that string, so the failure is loud rather than silent; the
    silent trap is mis-handling it as 1008.333, which stretches the trace by
    1000x and still fits a Michaelis-Menten curve perfectly while returning a
    Vmax three orders of magnitude too small. This is the format guard."""
    t, a = datasets.load_trace("vo1001")
    assert t[0] == 0.0
    assert t[-1] < 5.0, "time axis was not parsed correctly"
    assert np.all(np.diff(t) > 0), "time must increase"
    assert np.allclose(np.diff(t), 1 / 120.0, rtol=1e-3), \
        "the instrument sampled every 1/120 min (0.008333 min, i.e. 2 Hz)"


def test_loaded_trace_shapes_and_ranges():
    for stem in datasets.SUBSTRATE_UM:
        t, p_um, s0 = datasets.progress_curve(stem)
        assert len(t) == len(p_um) > 50
        assert np.all(np.isfinite(p_um))
        assert p_um[0] == pytest.approx(0.0, abs=1e-9)
        # the paper ran to 40-65% conversion at these concentrations
        assert 0.2 < p_um[-1] / s0 < 0.9, f"{stem}: implausible conversion"


def test_absorbsivity_conversion_is_dimensionally_right():
    """A 1 AU change at eps=15000 through 1 cm is 66.7 uM of product."""
    assert datasets.absorbance_to_product_um(1.0) == pytest.approx(66.667, rel=1e-4)
    assert datasets.absorbance_to_product_um(0.0) == 0.0
    with pytest.raises(ValueError):
        datasets.absorbance_to_product_um(1.0, epsilon=0.0)


def test_initial_rate_is_positive_and_monotone_in_substrate():
    table = datasets.initial_rate_table(0.15)
    v = [row["v_um_per_min"] for row in table]
    assert all(x > 0 for x in v)
    # five substrate concentrations from 11 to 64 uM, all below the published
    # Km of 25 uM except the top two: the rate must still be rising
    assert v == sorted(v), f"initial rates not monotone in [S]: {v}"


def test_initial_rate_depends_on_the_window_chosen():
    """This is the motivating problem for progress-curve fitting: the answer
    depends on where you cut the trace, and the cut is arbitrary."""
    narrow = datasets.initial_rate_table(0.05)
    wide = datasets.initial_rate_table(0.40)
    v_narrow = {r["s0_um"]: r["v_um_per_min"] for r in narrow}
    v_wide = {r["s0_um"]: r["v_um_per_min"] for r in wide}
    # averaging over a window that includes substrate depletion must give a
    # lower apparent rate at every concentration
    for s0 in v_narrow:
        assert v_wide[s0] < v_narrow[s0], f"{s0} uM: no window dependence"


def test_fit_to_published_data_lands_near_the_published_values():
    """The paper quotes kcat = 780 s^-1 and Km = 25 uM for this system.

    Tolerances are wide: Km within 0.5-2.5x and kcat within 0.4-3.0x of the
    published values. A real preparation is not the
    literature preparation -- see test_activation_fraction_is_reported. What
    would be alarming is an order-of-magnitude disagreement, and that is what
    this catches: e.g. a units error anywhere in the absorbance conversion.
    """
    table = datasets.initial_rate_table(0.15)
    S = np.array([r["s0_um"] for r in table])
    v = np.array([r["v_um_per_min"] for r in table])
    fit = fitting.fit_michaelis_menten(S, v)

    km_ratio = fit.params["km"] / (datasets.PUBLISHED_KM_M * 1e6)
    assert 0.5 < km_ratio < 2.5, f"Km off by {km_ratio:.2f}x the published value"

    kcat = datasets.kcat_from_vmax(fit.params["vmax"])
    kcat_ratio = kcat / datasets.PUBLISHED_KCAT_S
    assert 0.4 < kcat_ratio < 3.0, f"kcat off by {kcat_ratio:.2f}x published"


def test_activation_fraction_is_reported_not_hidden():
    """A kcat below the literature value is a statement about the enzyme
    preparation, not about the fitting code. The helper must express it as a
    fraction and must not silently rescale anything."""
    assert datasets.activation_fraction(780.0) == pytest.approx(1.0)
    assert datasets.activation_fraction(390.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        datasets.activation_fraction(100.0, published=0.0)


def test_kcat_conversion_matches_hand_calculation():
    """1 uM/min through 0.6 nM enzyme is (1e-6/60)/0.6e-9 = 27.8 s^-1."""
    assert datasets.kcat_from_vmax(1.0) == pytest.approx(27.7778, rel=1e-4)


def test_duplicate_files_are_declared_not_hidden():
    """The archive contains '-1' copies of three traces. They must be named in
    DUPLICATE_FILES so that anyone counting 21 files against 18 conditions can
    see why, rather than assuming the loader dropped data."""
    assert set(datasets.DUPLICATE_FILES.values()) <= set(datasets.SUBSTRATE_UM)
    assert not (set(datasets.DUPLICATE_FILES) & set(datasets.SUBSTRATE_UM))
    for dup in datasets.DUPLICATE_FILES:
        t, _, _ = datasets.progress_curve(dup)
        assert len(t) > 50


def test_rejects_a_nonexistent_trace():
    with pytest.raises(OSError):
        datasets.load_trace("not_a_trace")
