"""The committed QSAR artefacts hold every series, and say what they measure."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results_qsar_holds_all_three_series_and_the_csv_too():
    """d72ca0b replaced the artefact with a two-series run and deleted the
    21-compound section the README quotes. A rerun must not do that again."""
    mod = _load("qsar_congeneric")
    text = (ROOT / "results_qsar.txt").read_text()
    for doc in mod.SERIES_LABELS:
        assert f"--- {doc}:" in text, f"{doc} missing from results_qsar.txt"
    docs = {r["document_chembl_id"] for r in csv.DictReader(
        (ROOT / "results" / "qsar_series.csv").open())}
    assert docs == set(mod.SERIES_LABELS)


def test_default_series_is_the_full_set_and_a_subset_run_cannot_overwrite_it(tmp_path):
    mod = _load("qsar_congeneric")
    ap_default = mod.SERIES_LABELS
    assert len(ap_default) == 3
    src = (ROOT / "scripts" / "qsar_congeneric.py").read_text()
    assert 'default=list(SERIES_LABELS)' in src
    assert "qsar_adhoc_" in src       # subset runs go elsewhere


def test_correlation_with_the_loo_prediction_is_not_called_in_sample():
    src = (ROOT / "scripts" / "qsar_congeneric.py").read_text()
    assert "pearson_in_sample" not in src
    assert "pearson_obs_vs_loo" in src


def test_descriptor_spread_adjusts_eta_squared_and_flags_constants():
    mod = _load("qsar_single_scaffold")
    from enzkin import qsar
    smi = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1C", "C1CCCCC1CC",
           "C1CCCCC1CCC", "c1ccccc1O", "C1CCCCC1O"]
    rows = [(f"M{i}", 10.0 + i, qsar.compute_descriptors(s))
            for i, s in enumerate(smi)]
    scaffolds = ["benz", "benz", "benz", "cyc", "cyc", "cyc", "benz", "cyc"]
    spread = mod.descriptor_spread(rows, scaffolds)
    by = {d["descriptor"]: d for d in spread}
    assert by["AromaticRings"]["constant_within"] is True
    assert by["AromaticRings"]["eta_squared"] == pytest.approx(1.0)
    assert all(0.0 <= d["eta_squared_adj"] <= d["eta_squared"] + 1e-12 for d in spread)
    assert any(d["eta_squared_adj"] < d["eta_squared"] - 1e-6 for d in spread)


def test_single_scaffold_report_flags_constant_descriptors():
    text = (ROOT / "results_single_scaffold.txt").read_text()
    assert "constant within every scaffold" in text
    assert "adj" in text
    assert "other nine descriptors" not in text
