"""Two README headline numbers that no test tied to an artefact.

Changing "12× too small" to "13× too small", or the competitive row of the
mechanism table from 85% to 95%, left the whole suite green. These tests read
the committed artefacts and fail when the README stops saying what they say.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PUBLISHED = (ROOT / "results_published.txt").read_text(encoding="utf-8")
SIM = (ROOT / "results_sim_checks.txt").read_text(encoding="utf-8")


def test_readme_se_ratio_and_both_standard_errors_match_results_published():
    ratio = re.search(r"cluster SE / i\.i\.d\. SE = (\d+)x on Km", PUBLISHED).group(1)
    iid = re.search(r"km\s+=\s+[\d.]+ \+/- ([\d.]+)\n\s+SE is i\.i\.d\.", PUBLISHED).group(1)
    boot = re.search(r"cluster bootstrap: km = [\d.]+ \+/- ([\d.]+) uM, 95% \[([\d.]+), ([\d.]+)\]",
                     PUBLISHED)
    se, lo, hi = boot.groups()
    traces, points = re.search(r"(\d+) trace\(s\), (\d+) points", PUBLISHED).groups()
    assert f"**A standard error {ratio}× too small.**" in README
    assert f"**The i.i.d. Km standard error is {ratio}× too small on this dataset.**" in README
    assert f"±{iid} µM" in README and f"±{se} µM" in README
    assert f"({lo}–{hi})" in README
    assert f"{traces} progress curves, {points} points" in README
    # the ratio the artefact prints is the ratio of the two SEs it prints
    assert round(float(se) / float(iid)) == int(ratio)


def test_readme_mechanism_table_matches_the_pick_counts():
    """README cell = how often the true model ranked first; the noncompetitive
    row also counts mixed, as validate_fits.py does."""
    cells = {}
    for mech, noise, picks in re.findall(
            r"^  (\w+)\s+(\d+)%\s+\d+%  (.*?) \|", SIM, re.M):
        counts = dict((k, int(n)) for k, n in re.findall(r"(\w+) (\d+)", picks))
        hit = counts.get(mech, 0)
        if mech == "noncompetitive":
            hit += counts.get("mixed", 0)
        cells.setdefault(mech, {})[int(noise)] = hit
    assert set(cells) == {"competitive", "uncompetitive", "noncompetitive", "mixed"}
    for mech, by_noise in cells.items():
        row = f"| {mech} | {by_noise[2]}% | {by_noise[5]}% | {by_noise[10]}% |"
        assert row in README, f"README mechanism row is not: {row}"
