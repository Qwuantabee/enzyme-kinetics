"""The ChEMBL fetch follows pagination, filters units and keeps validity flags.

No network: the HTTP getter is injected. The raw-record fixtures mirror the
fields the real records carry (checked against the fetched JSON: standard_units
is "nM" on every record, 23 records have no raw units and a standard_value of
10**(9 - raw value), and ChEMBL flags them in data_validity_comment).
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_chembl_dhfr", ROOT / "scripts" / "fetch_chembl_dhfr.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def _page(n, start, nxt, total):
    return {"page_meta": {"next": nxt, "total_count": total},
            "activities": [{"activity_id": start + i} for i in range(n)]}


def test_fetch_follows_every_page_not_just_the_first():
    pages = {
        0: _page(1000, 0, "/chembl/api/data/activity.json?offset=1000", 2300),
        1: _page(1000, 1000, "/chembl/api/data/activity.json?offset=2000", 2300),
        2: _page(300, 2000, None, 2300),
    }
    calls = []

    def get(url):
        calls.append(url)
        return pages[len(calls) - 1]

    acts = fetch.fetch_activities("IC50", get=get, sleep=0)
    assert len(acts) == 2300 and len(calls) == 3
    assert calls[1] == "https://www.ebi.ac.uk/chembl/api/data/activity.json?offset=1000"


def test_fetch_refuses_a_truncated_result():
    def get(url):
        return _page(1000, 0, None, 1500)   # says 1500, delivers 1000
    with pytest.raises(RuntimeError, match="truncated"):
        fetch.fetch_activities("Ki", get=get, sleep=0)


def _act(value, units="nM", **kw):
    d = {"standard_value": value, "standard_units": units,
         "molecule_chembl_id": "M", "assay_chembl_id": "A",
         "document_chembl_id": "D", "standard_relation": "="}
    d.update(kw)
    return d


def test_units_are_filtered_and_every_drop_is_counted():
    acts = [_act("10"), _act("5", units="uM"), _act("abc"), _act("0"),
            _act("-3"), _act(None), _act("7", units=None)]
    rows, dropped = fetch.activity_rows(acts, "ic50_nM", {})
    assert len(rows) == 1 and rows[0]["ic50_nM"] == "10"
    assert dropped == {"no_value": 2, "non_positive": 2, "wrong_units": 2}
    assert len(rows) + sum(dropped.values()) == len(acts)


def test_validity_comment_raw_units_and_pchembl_are_written():
    acts = [_act("158489319", units=None, standard_units="nM",
                 data_validity_comment="Outside typical range",
                 pchembl_value=None)]
    rows, _ = fetch.activity_rows(acts, "ic50_nM", {})
    r = rows[0]
    assert r["data_validity_comment"] == "Outside typical range"
    assert r["raw_units"] == "" and r["standard_units"] == "nM"
    assert "pchembl" in r
    raw = {**_act("12.5"), "units": "nM", "pchembl_value": "7.90"}
    ok = fetch.activity_rows([raw], "ki_nM", {})[0][0]
    assert ok["pchembl"] == "7.90" and ok["raw_units"] == "nM"


def test_the_vendored_loader_reads_the_column_the_fetch_now_writes():
    from enzkin import datasets
    src = (ROOT / "src" / "enzkin" / "datasets.py").read_text()
    assert 'r.get("pchembl")' in src
    assert "pchembl" in fetch.activity_rows([_act("1")], "ic50_nM", {})[0][0]
