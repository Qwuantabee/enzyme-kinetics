"""README and docs/ statements added on 2026-11-17, tied to their artefacts.

Each test reads a committed result file and fails when the README (or the
moved docs/planned_items.md) stops saying what the file says.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOCS = (ROOT / "docs" / "planned_items.md").read_text(encoding="utf-8")
PUBLISHED = (ROOT / "results_published.txt").read_text(encoding="utf-8")
SIM = (ROOT / "results_sim_checks.txt").read_text(encoding="utf-8")


def test_window_sweep_in_readme_matches_results_published():
    block = PUBLISHED.split("--- initial rates: dependence on the window ---", 1)[1]
    rows = re.findall(r"^\s+(\d+)%\s+([\d.]+)\s+[\d.]+\s+\d+\s+\d+$", block.split("---", 1)[0], re.M)
    assert [w for w, _ in rows] == ["5", "10", "15", "25", "40"]
    vmax = [v for _, v in rows]
    assert f"moves Vmax from {vmax[0]} to {vmax[-1]}" in README
    assert f"({', '.join(vmax)} at 5, 10, 15, 25 and 40%" in README


def test_aicc_k_plus_one_statement_matches_section_c():
    m = re.search(r"winner changed in (\d+) of (\d+) experiments", SIM)
    assert m, "section (c) summary line missing from results_sim_checks.txt"
    changed, total = m.groups()
    assert f"in {changed} of the {total} experiments below" in README
    moved = [int(x) for x in re.findall(r"; changed (\d+) of 100$", SIM, re.M)]
    assert len(moved) == 12 and max(moved) <= 3
    assert "no cell by more than 3 points" in README
    # gap between a 3- and a 4-parameter model at n = 28, both conventions
    n = 28
    corr = lambda k: 2 * k * (k + 1) / (n - k - 1)  # noqa: E731
    assert f"{2 + corr(4) - corr(3):.2f}" == "2.74" and "from 2.00 to 2.74" in README
    assert f"{2 + corr(5) - corr(4):.2f}" == "2.99" and "makes the gap 2.99" in README


def test_lead_bullet_spread_matches_results_sim_checks():
    lb = re.search(r"^\s+lineweaver_burk\s+[+-][\d.]+%\s+([\d.]+)%", SIM, re.M).group(1)
    nl = re.search(r"^\s+nonlinear\s+[+-][\d.]+%\s+([\d.]+)%", SIM, re.M).group(1)
    ratio = re.search(r"Lineweaver-Burk / nonlinear, robust SD: ([\d.]+)x", SIM).group(1)
    lb, nl = round(float(lb)), round(float(nl))
    assert f"is {lb}% against {nl}% for the nonlinear fit over 500 simulated experiments" in README
    assert f"Km spread {ratio}× the nonlinear fit's" in README
    assert (f"Km spread is {ratio}× worse than the nonlinear fit's ({lb}% for Lineweaver-Burk "
            f"against {nl}% for the nonlinear fit)") in README


def test_moved_cheng_prusoff_numbers_still_match_the_stats_csv():
    with open(ROOT / "results" / "cheng_prusoff_ratio_stats.csv", newline="") as fh:
        strict = next(r for r in csv.DictReader(fh) if r["subset"] == "same_document_all")
    assert f"{float(strict['within_group_log10_sd']):.3f}" in DOCS
    assert f"F(4,41) = {float(strict['anova_F']):.2f}" in DOCS


def test_readme_points_to_the_moved_block_and_notice():
    assert "[`docs/planned_items.md`](docs/planned_items.md)" in README
    assert (ROOT / "NOTICE").is_file() and "[`NOTICE`](NOTICE)" in README
    assert "## Limits, stated plainly" in README and "## Reproducing" in README
