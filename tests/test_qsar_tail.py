"""The tail sweep: the signal is NOT tail-independent in the 1996 series.

Pinned from results/qsar_tail_sweep.csv (regenerate with
scripts/qsar_tail_sweep.py). The earlier README sentence 'It is not tail-driven'
rested on dropping only the 1-2 compounds at Ki >= 100 uM; at a 100 nM cutoff the
1996 series loses its fit and the 1988 series keeps it.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    return list(csv.DictReader((ROOT / "results" / "qsar_tail_sweep.csv").open()))


def _get(doc, subset, cutoff):
    for r in _rows():
        if (r["document"], r["subset"], r["cutoff_nM"]) == (doc, subset, cutoff):
            return r
    raise AssertionError((doc, subset, cutoff))


def test_1996_whole_series_loses_the_signal_at_a_100_nM_cutoff():
    full = _get("CHEMBL1129627", "whole series", "")
    mild = _get("CHEMBL1129627", "whole series", "100000.0")
    strict = _get("CHEMBL1129627", "whole series", "100.0")
    assert float(full["loocv_r2"]) == pytest.approx(0.326, abs=0.001)
    assert float(mild["loocv_r2"]) == pytest.approx(0.582, abs=0.001)
    assert int(strict["n_kept"]) == 33 and int(strict["n_dropped"]) == 40
    assert float(strict["loocv_r2"]) < 0, "R2 is negative at the 100 nM cutoff"
    assert float(strict["best_single_r2"]) == pytest.approx(0.036, abs=0.002)


def test_1996_largest_scaffold_also_goes_negative():
    strict = _get("CHEMBL1129627", "largest scaffold", "100.0")
    assert int(strict["n_kept"]) == 29
    assert float(strict["loocv_r2"]) == pytest.approx(-0.143, abs=0.002)


def test_1988_keeps_a_positive_r2_but_one_descriptor_matches_it_at_100_nM():
    strict = _get("CHEMBL1124156", "whole series", "100.0")
    assert int(strict["n_kept"]) == 25
    assert float(strict["loocv_r2"]) == pytest.approx(0.556, abs=0.002)
    assert float(strict["p_perm"]) <= 0.002
    # the ten-descriptor model is no better than the best single column
    assert float(strict["best_single_r2"]) >= float(strict["loocv_r2"])


def test_the_two_series_do_not_agree_so_no_blanket_tail_claim_is_allowed():
    s96 = float(_get("CHEMBL1129627", "whole series", "100.0")["loocv_r2"])
    s88 = float(_get("CHEMBL1124156", "whole series", "100.0")["loocv_r2"])
    assert s96 < 0 < s88


def test_affinity_bands_count_every_compound_once():
    spec = importlib.util.spec_from_file_location(
        "qsar_tail_sweep", ROOT / "scripts" / "qsar_tail_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ki = [0.5, 1.0, 99.9, 100.0, 99999.0, 1e5, 2e6]
    b = mod.affinity_bands(ki)
    assert b == {"below_1nM": 1, "1_to_100nM": 2, "100nM_to_100uM": 2,
                 "100uM_and_above": 2}
    assert sum(b.values()) == len(ki)


def test_report_states_ki_bands_the_readme_relies_on():
    text = (ROOT / "results_qsar_tail.txt").read_text()
    assert "below_1nM 12, 1_to_100nM 13, 100nM_to_100uM 45, 100uM_and_above 2" in text
    assert "below_1nM 0, 1_to_100nM 33, 100nM_to_100uM 39, 100uM_and_above 1" in text
