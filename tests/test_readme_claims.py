"""The README states what the artefacts measured, and nothing stale.

Every string below was once a README claim that the data contradicted or that
no file supported; each check makes that sentence fail loudly if it returns.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def test_qsar_direction_is_stated_the_right_way_round():
    """fe2ed35: the README had 'bind worse' for a negative rho against
    log10 Ki. A negative rho means the descriptor raises affinity."""
    assert "bind *better* here" in README
    assert "bind *worse*" not in README
    assert "heavier compounds binding *worse*" not in README


def test_stale_claims_do_not_return():
    for stale in ("five-order-of-magnitude", "354 compounds", "non_enzyme",
                  "It is not tail-driven", "threshold sits",
                  "genuinely describes these traces", "has_assay_substrate",
                  "No record carries [S]", "four assays, four implied",
                  "the rest weaker", "so their R² is carried by compounds no"):
        assert stale not in README, stale


def test_readme_cheng_prusoff_numbers_match_the_committed_artefact():
    rows = {r["subset"]: r for r in csv.DictReader(
        (ROOT / "results" / "cheng_prusoff_ratio_stats.csv").open())}
    strict = rows["same_document_all"]
    assert f"{float(strict['median_ratio']):.2f}" in README
    assert f"| {strict['n']} | {float(strict['median_ratio']):.2f} |" in README
    lo, hi = float(strict["median_ci_cluster_low"]), float(strict["median_ci_cluster_high"])
    assert f"**{lo:.2f}–{hi:.2f}**" in README
    assert f"{float(strict['within_group_log10_sd']):.3f}" in README
    assert f"F(4,41) = {float(strict['anova_F']):.2f}" in README
    assert strict["verdict"] == "partially supported"
    assert rows["same_molecule_all"]["verdict"] == "not testable"
    assert "partially supported" in README and "not testable" in README


def test_readme_tail_numbers_match_the_sweep_csv():
    rows = list(csv.DictReader((ROOT / "results" / "qsar_tail_sweep.csv").open()))

    def get(doc, subset, cutoff):
        return next(r for r in rows if (r["document"], r["subset"], r["cutoff_nM"])
                    == (doc, subset, cutoff))
    s96 = get("CHEMBL1129627", "whole series", "100.0")
    s88 = get("CHEMBL1124156", "whole series", "100.0")
    assert f"**{float(s96['loocv_r2']):.3f}**".replace("-", "−") in README
    assert f"n = {s96['n_kept']}" in README
    assert f"**{float(s88['loocv_r2']):.3f}**" in README
    assert f"n = {s88['n_kept']}" in README


def test_readme_test_count_is_the_collected_count():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p",
         "no:cacheprovider", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    m = re.search(r"(\d+) tests? collected", out.stdout + out.stderr)
    assert m, out.stdout[-300:]
    n = int(m.group(1))
    assert f"{n} tests:" in README, f"README says another number than {n}"
