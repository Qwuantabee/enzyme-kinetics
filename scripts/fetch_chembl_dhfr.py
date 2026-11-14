"""Fetch the E. coli DHFR inhibition data used in this repo from ChEMBL.

    python scripts/fetch_chembl_dhfr.py [--force]

Writes data/chembl_dhfr/ecoli_dhfr_ic50.csv and ecoli_dhfr_ki.csv, plus the
per-assay descriptions the mutant filter depends on.

Why this exists rather than just committing the CSVs: the assay descriptions
are what make the data interpretable. Methotrexate appears in the Ki table
five times with values spanning 0.001 to 170000 nM, and the only way to know
that three of those are Phe31/Tyr31/Val31 mutants is to read the descriptions.
Committing those alongside the numbers means the filter in
enzkin.datasets.load_chembl_dhfr_ki can be checked against its source.

What the fetch guards against
-----------------------------
* PAGINATION. ChEMBL returns at most "limit" records per request and gives
  the next page in page_meta.next. The previous version read one page and
  stopped, so a type with more than 1000 records would have been silently
  truncated. fetch_activities follows page_meta.next until it is empty and
  checks the number of records against page_meta.total_count; a mismatch
  aborts instead of writing a short table.
* UNITS. ChEMBL's standard_value is in standard_units, and a value is only a
  concentration in nM if standard_units == "nM". Records in any other unit are
  dropped here and COUNTED in the printed summary. This is not sufficient on
  its own: ChEMBL can standardise a unit-less raw number as if it were a
  -log10(M) quantity and still label the result "nM". Those records carry a
  data_validity_comment (e.g. "Outside typical range") and no raw units, so
  data_validity_comment and the raw units are written to the CSV, where the
  loader can flag them. Nothing is dropped for a validity comment here.
* pChEMBL. The loader reads a pchembl column. It was never written, so it
  was always empty; pchembl_value is now written.

Network: ChEMBL's REST API, no key. Roughly one request per assay, a few
hundred total, with a short sleep between the assay lookups so as not to lean
on a public service.
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "chembl_dhfr"
TARGET = "CHEMBL1809"
HOST = "https://www.ebi.ac.uk"
BASE = HOST + "/chembl"      # page_meta.next already starts with "/chembl/api/..."
PAGE_LIMIT = 1000

#: The only unit in which standard_value is a concentration in nM.
CONCENTRATION_UNIT = "nM"


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as fh:
                return json.load(fh)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_activities(standard_type, get=_get, sleep=0.15):
    """All activities of one type for the target, '=' relations only.

    Follows page_meta.next until it is null and verifies that the number of
    records collected equals page_meta.total_count; raises if it does not,
    rather than returning a silently truncated list. get is injectable so the
    pagination can be tested without a network.
    """
    url = (f"{BASE}/api/data/activity.json?target_chembl_id={TARGET}"
           f"&standard_type={standard_type}&standard_relation=%3D"
           f"&limit={PAGE_LIMIT}")
    out, total = [], None
    while url:
        data = get(url)
        meta = data.get("page_meta") or {}
        if total is None:
            total = meta.get("total_count")
        out.extend(data.get("activities", []))
        nxt = meta.get("next")
        url = (HOST + nxt) if nxt else None
        if url and sleep:
            time.sleep(sleep)
    if total is not None and len(out) != total:
        raise RuntimeError(
            f"{standard_type}: collected {len(out)} records but ChEMBL reports "
            f"total_count {total}; refusing to write a truncated table")
    return out


def fetch_assay_meta(assay_ids):
    """assay_chembl_id -> (description, organism). One request each."""
    out = {}
    for i, aid in enumerate(sorted(assay_ids), 1):
        try:
            d = _get(f"{BASE}/api/data/assay/{aid}.json", retries=2)
            out[aid] = {
                "description": d.get("description") or "",
                "organism": d.get("assay_organism") or "",
            }
        except Exception as exc:
            print(f"  {aid}: {exc}", file=sys.stderr)
            out[aid] = {"description": "", "organism": ""}
        if i % 10 == 0:
            print(f"  {i}/{len(assay_ids)} assays")
        time.sleep(0.15)
    return out


def activity_rows(acts, value_key, meta):
    """Turn raw ChEMBL activities into CSV rows. Returns (rows, dropped).

    dropped counts every record that did not become a row, by reason:
    no_value (standard_value missing or not a number), non_positive,
    wrong_units (standard_units != "nM"). Nothing else is dropped.
    """
    rows = []
    dropped = {"no_value": 0, "non_positive": 0, "wrong_units": 0}
    for a in acts:
        try:
            v = float(a["standard_value"])
        except (TypeError, ValueError, KeyError):
            dropped["no_value"] += 1
            continue
        if a.get("standard_units") != CONCENTRATION_UNIT:
            dropped["wrong_units"] += 1
            continue
        if v <= 0:
            dropped["non_positive"] += 1
            continue
        m = meta.get(a.get("assay_chembl_id", ""), {})
        rows.append({
            "molecule_chembl_id": a.get("molecule_chembl_id", ""),
            "assay_chembl_id": a.get("assay_chembl_id", ""),
            "document_chembl_id": a.get("document_chembl_id", ""),
            value_key: f"{v:.6g}",
            "relation": a.get("standard_relation", "="),
            "standard_units": a.get("standard_units", ""),
            "raw_units": a.get("units") or "",
            "data_validity_comment": a.get("data_validity_comment") or "",
            "pchembl": a.get("pchembl_value") or "",
            "assay_description": m.get("description", ""),
            "assay_organism": m.get("organism", ""),
        })
    return rows, dropped


def write_table(rows, path):
    if not rows:
        raise SystemExit(f"no rows to write for {path}")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if the CSVs already exist")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ic_path = OUT_DIR / "ecoli_dhfr_ic50.csv"
    ki_path = OUT_DIR / "ecoli_dhfr_ki.csv"
    if not args.force and ic_path.exists() and ki_path.exists():
        print("CSVs already present; pass --force to re-fetch")
        return 0

    print(f"target {TARGET}")
    ic = fetch_activities("IC50")
    ki = fetch_activities("Ki")

    assay_ids = {a["assay_chembl_id"] for a in ic + ki}
    print(f"{len(ic)} IC50 records, {len(ki)} Ki records, "
          f"{len(assay_ids)} distinct assays")
    meta = fetch_assay_meta(assay_ids)

    ic_rows, ic_drop = activity_rows(ic, "ic50_nM", meta)
    ki_rows, ki_drop = activity_rows(ki, "ki_nM", meta)
    print(f"IC50: kept {len(ic_rows)}, dropped {ic_drop}")
    print(f"Ki:   kept {len(ki_rows)}, dropped {ki_drop}")
    write_table(ic_rows, ic_path)
    write_table(ki_rows, ki_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
