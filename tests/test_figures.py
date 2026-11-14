"""Every figure is reproducible, and says what its artefact says.

scripts/make_figures.py is the only thing that writes figures/. These tests fail
when a PNG exists that nothing regenerates (or the reverse), when a figure's
inputs drift from the numbers the results files and the README state, or when
the simulated figures stop reproducing the README tables. They never edit a
figure and never compare PNG bytes (those differ between matplotlib builds);
the committed PNG's pixel size is compared, which catches a figure that was
changed in the script and not re-committed.
"""

import importlib.util
import math
import re
import struct
from pathlib import Path

import pytest

from enzkin import fitting, simulate

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _load_make_figures():
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "scripts" / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mf = _load_make_figures()


def _png_size(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", data[16:24])


# --- the directory is reproducible -----------------------------------------

def test_figures_directory_and_generator_agree():
    on_disk = {p.name for p in (ROOT / "figures").glob("*.png")}
    registered = set(mf.FIGURES)
    assert on_disk == registered, (
        f"on disk only: {sorted(on_disk - registered)}; "
        f"generated only: {sorted(registered - on_disk)}")


def test_every_figure_is_embedded_in_the_readme():
    missing = [n for n in mf.FIGURES if f"figures/{n}" not in README]
    assert not missing, f"figures not shown in README.md: {missing}"


def test_only_make_figures_writes_figures():
    offenders = [p.name for p in (ROOT / "scripts").glob("*.py")
                 if p.name != "make_figures.py"
                 and "savefig" in p.read_text(encoding="utf-8")]
    assert not offenders, f"{offenders} write figures; only make_figures.py may"


def test_all_figures_regenerate_at_the_committed_size(tmp_path):
    assert mf.main(["--out", str(tmp_path)]) == 0
    for name in mf.FIGURES:
        width, height = _png_size(tmp_path / name)
        assert 800 <= width <= 1700 and 600 <= height <= 950, (name, width, height)
        assert (width, height) == _png_size(ROOT / "figures" / name), (
            f"figures/{name} was not re-committed after make_figures.py changed")


def test_unknown_figure_name_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        mf.main(["--only", "no_such_figure", "--out", str(tmp_path)])


# --- figures read from artefacts show what the artefacts say ---------------

def test_two_substrate_inputs_match_the_results_text():
    d = mf.read_two_substrate()
    assert len(d["ratio"]) == 11 and all(d["inside"])
    assert round(d["median"], 2) == 2.18
    assert (round(min(d["ratio"]), 2), round(max(d["ratio"]), 2)) == (1.45, 2.77)
    assert round(d["upper"], 2) == 3.33
    assert (round(d["pred_low"], 2), round(d["pred_high"], 2)) == (2.84, 2.69)
    text = (ROOT / "results_cheng_prusoff.txt").read_text(encoding="utf-8")
    assert "median 2.18, range 1.45 to 2.77; 11 of 11 inside the bounds" in text
    assert "2.84 (30-based Km) or 2.69 (100-based Km)" in text


def test_ratio_groups_reproduce_the_per_group_table():
    text = (ROOT / "results_cheng_prusoff.txt").read_text(encoding="utf-8")
    rows = re.findall(
        r"^(CHEMBL\d+(?:@\d+uM)?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)x\s+[\d.]+\s*$",
        text, re.M)
    assert len(rows) == 5
    d = mf.read_pairs()
    by_name = dict(d["groups"])
    for name, n, median, spread in rows:
        s = mf.group_stats(by_name[name])
        assert (s["n"], round(s["median"], 2), round(s["spread"], 2)) == (
            int(n), float(median), float(spread)), name
    # the ranges quoted in the figure title and the README caption
    assert "Per-group median ratios run from 3.0 to 20.9" in text
    assert "per-group max/min ratio spreads run\nfrom 2.0x to 26.9x" in text
    medians = [float(m) for _, _, m, _ in rows]
    assert (min(medians), max(medians)) == (3.04, 20.95)
    assert sum(len(v) for _, v in d["groups"]) == d["n"] == 47
    # the pooled within-group SD recomputed from the pairs is the stored one
    big = [v for _, v in d["groups"] if len(v) >= 3]
    ss = sum((len(v) - 1) * mf.group_stats(v)["sd"] ** 2 for v in big)
    pooled = math.sqrt(ss / sum(len(v) - 1 for v in big))
    assert abs(pooled - d["pooled_sd"]) < 5e-4
    assert round(d["pooled_median"], 2) == 9.38


def test_tail_sweep_inputs_match_the_readme_numbers():
    sw = mf.read_tail_sweep()
    cases = {
        ("CHEMBL1129627", "whole series"): (33, -0.070),
        ("CHEMBL1124156", "whole series"): (25, 0.556),
        ("CHEMBL1129627", "largest scaffold"): (29, -0.143),
        ("CHEMBL1124156", "largest scaffold"): (21, 0.873),
    }
    for key, (n, r2) in cases.items():
        row = sw[key][100.0]
        assert (row["n"], round(row["r2"], 3)) == (n, r2), key


def test_mw_gain_inputs_agree_with_the_sweep_baseline():
    rows = mf.read_mw_gain()
    got = [(r["series"], r["subset"], r["n"], r["descriptor"], r["gain"]) for r in rows]
    assert got == [("1996", "whole series", 73, "MW", 0.104),
                   ("1996", "largest scaffold", 66, "MW", 0.109),
                   ("1988", "whole series", 72, "MW", 0.166),
                   ("1988", "largest scaffold", 52, "MW", 0.009)]
    sweep = mf.read_tail_sweep()
    doc = {"1996": "CHEMBL1129627", "1988": "CHEMBL1124156"}
    for r in rows:
        base = sweep[(doc[r["series"]], r["subset"])][None]
        assert abs(base["r2"] - r["ten"]) < 6e-4
        assert abs(base["single"] - r["single"]) < 6e-4
        assert abs(r["ten"] - r["single"] - r["gain"]) < 1.6e-3


def test_published_km_inputs_and_the_claim_in_the_title():
    d = mf.read_published()
    assert (d["rate_km"], d["rate_se"], d["n_rates"]) == (33.98, 5.03, 5)
    assert (d["iid_km"], d["iid_se"]) == (29.572, 0.185)
    assert (d["n_traces"], d["n_points"], d["n_boot"]) == (18, 2376, 400)
    assert (d["cl_km"], d["cl_se"], d["cl_lo"], d["cl_hi"]) == (29.57, 2.21, 25.74, 33.44)
    assert d["se_ratio"] == 12 and d["published_km"] == 25.0
    # the figure says the published Km lies outside the cluster interval
    assert d["published_km"] < d["cl_lo"]


# --- simulated figures reproduce the README tables -------------------------

@pytest.fixture(scope="module")
def recovery():
    return {nt: mf.km_recovery_errors(nt) for nt in ("relative", "absolute")}


ROWS = [
    # error model, method, README row prefix, Km bias %, Km SD %, tolerance
    ("relative", "nonlinear", "| nonlinear fit | +0.8% | 10.3% |", 0.8, 10.3, 0.06),
    ("relative", "lineweaver_burk", "| Lineweaver-Burk | +0.1% | 11.1% |", 0.1, 11.1, 0.06),
    ("relative", "hanes_woolf", "| Hanes-Woolf | +0.8% | 11.0% |", 0.8, 11.0, 0.06),
    ("relative", "eadie_hofstee", "| Eadie-Hofstee | -1.1% | **6.2%** |", -1.1, 6.2, 0.06),
    ("absolute", "nonlinear", "| nonlinear fit | +1.0% | 18.2% |", 1.0, 18.2, 0.06),
    ("absolute", "lineweaver_burk", "| Lineweaver-Burk | -318% | **6086%** |", -318.5, 6085.9, 0.6),
    ("absolute", "hanes_woolf", "| Hanes-Woolf | +17.6% | 228% |", 17.6, 228.2, 0.6),
    ("absolute", "eadie_hofstee", "| Eadie-Hofstee | -23.8% | 25.2% |", -23.8, 25.2, 0.06),
]


@pytest.mark.parametrize("noise,method,prefix,bias,sd,tol", ROWS,
                         ids=[f"{r[0]}-{r[1]}" for r in ROWS])
def test_recovery_figure_matches_the_readme_table(recovery, noise, method, prefix, bias, sd, tol):
    errors = recovery[noise][method]
    assert abs(errors.mean() - bias) <= tol
    assert abs(errors.std() - sd) <= tol
    assert prefix in README, f"README table row missing: {prefix}"


def test_lineweaver_burk_fails_only_under_absolute_error(recovery):
    assert recovery["absolute"]["lineweaver_burk"].std() > 100 * recovery["absolute"]["nonlinear"].std()
    assert recovery["relative"]["lineweaver_burk"].std() < 1.2 * recovery["relative"]["nonlinear"].std()
    for noise in ("relative", "absolute"):
        sds = {k: v.std() for k, v in recovery[noise].items()}
        assert sds["nonlinear"] < max(sds.values()), "the nonlinear fit was the worst"


def test_worked_example_numbers_match_the_readme():
    _, _, truth, fit = mf._example_fit()
    assert (truth["vmax"], truth["km"]) == (100.0, 25.0)
    assert (round(fit.params["vmax"], 1), round(fit.stderr["vmax"], 2)) == (98.0, 2.91)
    assert (round(fit.params["km"], 1), round(fit.stderr["km"], 2)) == (25.5, 2.48)
    assert "vmax = 98.0 +/- 2.91" in README and "km = 25.5 +/- 2.48" in README


def test_inhibition_example_matches_the_readme_table():
    S, I, v, _ = simulate.inhibition_grid(mechanism="mixed", ki=10.0, kis=40.0,
                                          noise=0.03, seed=3)
    results = fitting.compare_models(S, I, v)
    best = results[0]
    assert best.model == "mixed"
    assert (round(best.params["ki"], 1), round(best.params["kis"], 1)) == (10.0, 45.1)
    daicc = {r.model: round(r.aicc - best.aicc, 2) for r in results}
    assert daicc == {"mixed": 0.0, "noncompetitive": 53.08, "competitive": 69.61,
                     "uncompetitive": 92.42}
    for model in ("noncompetitive", "competitive", "uncompetitive"):
        assert f"| {model} |" in README and f"| {daicc[model]:.2f} |" in README
    assert "Recovered Ki = 10.0 ± 0.7 and Kis = 45.1 ± 3.1" in README
