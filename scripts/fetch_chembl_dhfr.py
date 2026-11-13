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
BASE = "https://www.ebi.ac.uk/chembl/api/data"


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


def fetch_activities(standard_type):
    """All activities of one type for the target, with '=' relations only."""
    url = (f"{BASE}/activity.json?target_chembl_id={TARGET}"
           f"&standard_type={standard_type}&standard_relation=%3D&limit=1000")
    data = _get(url)
    return data.get("activities", [])


def fetch_assay_meta(assay_ids):
    """assay_chembl_id -> (description, organism). One request each."""
    out = {}
    for i, aid in enumerate(sorted(assay_ids), 1):
        try:
            d = _get(f"{BASE}/assay/{aid}.json", retries=2)
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

    def rows_for(acts, value_key):
        out = []
        for a in acts:
            try:
                v = float(a["standard_value"])
            except (TypeError, ValueError, KeyError):
                continue
            if v <= 0:
                continue
            m = meta.get(a["assay_chembl_id"], {})
            out.append({
                "molecule_chembl_id": a.get("molecule_chembl_id", ""),
                "assay_chembl_id": a.get("assay_chembl_id", ""),
                "document_chembl_id": a.get("document_chembl_id", ""),
                value_key: f"{v:.6g}",
                "relation": a.get("standard_relation", "="),
                "assay_description": m.get("description", ""),
                "assay_organism": m.get("organism", ""),
            })
        return out

    write_table(rows_for(ic, "ic50_nM"), ic_path)
    write_table(rows_for(ki, "ki_nM"), ki_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
