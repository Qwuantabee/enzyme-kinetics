"""results_sim_checks.txt says what scripts/sim_checks.py computes, and the
README quotes it without drifting.

The Lineweaver-Burk Km SD under absolute error is a sample moment of a
distribution with no finite variance, so the README quotes a tail-insensitive
spread next to it. These tests recompute the seeds 0-499 block and one row of
the mechanism table and compare them with the committed artefact, then check
that each figure the README takes from the artefact is still in it.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from enzkin import fitting, simulate
from scripts import sim_checks

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORT = (ROOT / "results_sim_checks.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def errs():
    return sim_checks.km_errors(range(500))


def test_lineweaver_burk_tail_numbers_match_the_artefact(errs):
    lb, nl = errs["lineweaver_burk"], errs["nonlinear"]
    order = np.argsort(-np.abs(lb))
    keep = np.ones(len(lb), bool)
    keep[order[0]] = False
    assert f"{100 * lb.std():9.1f}%" in REPORT
    assert f"most extreme fit(s)  {100 * lb[keep].std():9.1f}%" in REPORT
    assert f"fits returning a negative Km: {int((lb < -1).sum())} of 500" in REPORT
    assert (f"lineweaver_burk  {100 * np.median(lb):>+12.1f}% "
            f"{100 * sim_checks.robust_sd(lb):>9.1f}%") in REPORT
    assert (f"nonlinear        {100 * np.median(nl):>+12.1f}% "
            f"{100 * sim_checks.robust_sd(nl):>9.1f}%") in REPORT
    # the property the README leans on: the tail-insensitive spread is finite,
    # far below the sample SD, and still several times the nonlinear fit's
    assert sim_checks.robust_sd(lb) < lb.std() / 50
    assert sim_checks.robust_sd(lb) > 3 * sim_checks.robust_sd(nl)


def test_competitive_misses_are_all_mixed_and_match_the_artefact():
    picks = Counter()
    for seed in range(100):
        S, I, v, _ = simulate.inhibition_grid(mechanism="competitive", noise=0.05, seed=seed)
        picks[fitting.compare_models(S, I, v)[0].model] += 1
    assert set(picks) == {"competitive", "mixed"}
    assert (f"competitive        5%  {picks['competitive']:>9}%  "
            f"competitive {picks['competitive']}, mixed {picks['mixed']} |") in REPORT


@pytest.mark.parametrize("readme_text,report_text", [
    ("1845%", "most extreme fit(s)     1845.2%"),
    ("84 of the 500 fits return a negative Km", "fits returning a negative Km: 84 of 500"),
    ("59% for Lineweaver-Burk against 16% for the nonlinear fit", "lineweaver_burk         -37.5%      59.1%"),
    ("59% for Lineweaver-Burk against 16% for the nonlinear fit", "nonlinear                -1.0%      16.3%"),
    ("3.6× worse", "robust SD: 3.6x"),
    ("1088% to 11507%", "LB sample SD 1088% to 11507%"),
    ("52.5–72.7%", "LB robust SD 52.5% to 72.7%"),
    ("about 13% at n = 28", "probability 12.9%"),
    ("21, 16 and 15 of 100", "32 x Km: competitive 79, mixed 21"),
    ("21, 16 and 15 of 100", "128 x Km: competitive 84, mixed 16"),
    ("21, 16 and 15 of 100", "512 x Km: competitive 85, mixed 15"),
    ("84%, 84% and 85%, not 100%", "noncompetitive     2%         84%"),
    ("84%, 84% and 85%, not 100%", "noncompetitive    10%         85%"),
    ("20 times in 100 where AICc picks it 15", "competitive 85, mixed 15 | competitive 80, mixed 20"),
    ("84% under AICc, 90% under plain AIC", "noncompetitive 16, mixed 84 | noncompetitive 10, mixed 90"),
])
def test_readme_figures_from_the_sim_checks_are_in_the_artefact(readme_text, report_text):
    assert readme_text in README, f"README no longer says: {readme_text}"
    assert report_text in REPORT, f"results_sim_checks.txt no longer says: {report_text}"
