"""Cache canonical SMILES for the compounds in the local Ki table.

    python scripts/fetch_chembl_smiles.py [--force]

Writes data/chembl_dhfr/ecoli_dhfr_smiles.csv (molecule_chembl_id,
canonical_smiles) covering every compound that appears in
data/chembl_dhfr/ecoli_dhfr_ki.csv.

Why a separate step rather than folding the lookups into the analysis: the
structure-affinity script (scripts/qsar_congeneric.py) has to be re-runnable
offline, and ChEMBL is a shared public service. Fetch once, cache to disk, and
let every later run read the cache.

The output is about 20 KB of SMILES, small enough to commit under the same
exception the Ki and IC50 tables already use, so the descriptor analysis runs
on a fresh clone with no network. The fetch is batched -- ChEMBL's molecule
endpoint accepts `molecule_chembl_id__in` and returns up to 1000 records per
request -- so the whole table is a handful of calls rather than one per
compound.
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "chembl_dhfr"
KI_CSV = DATA_DIR / "ecoli_dhfr_ki.csv"
OUT = DATA_DIR / "ecoli_dhfr_smiles.csv"
BASE = "https://www.ebi.ac.uk/chembl/api/data"
BATCH = 50


def _get_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                return json.load(fh)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_batch(ids, cache):
    """Look up one batch of ChEMBL ids, filling `cache` in place."""
    query = ",".join(ids)
    url = (f"{BASE}/molecule.json?molecule_chembl_id__in="
           f"{urllib.parse.quote(query)}&limit=1000")
    data = _get_json(url)
    for mol in data.get("molecules", []):
        cid = mol.get("molecule_chembl_id")
        structures = mol.get("molecule_structures") or {}
        cache[cid] = structures.get("canonical_smiles") or None
    for cid in ids:
        cache.setdefault(cid, None)   # ChEMBL did not return it at all
    return cache


def read_cache(path=None):
    """The on-disk SMILES map, empty if the cache has not been built."""
    path = Path(path) if path is not None else OUT
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["molecule_chembl_id"]: (r["canonical_smiles"] or None)
                for r in csv.DictReader(fh)}


def write_cache(cache, path=None):
    path = Path(path) if path is not None else OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_chembl_id", "canonical_smiles"])
        for cid in sorted(cache):
            w.writerow([cid, cache[cid] or ""])
    return path


def compound_ids(path=None):
    path = Path(path) if path is not None else KI_CSV
    with path.open(newline="") as fh:
        return sorted({r["molecule_chembl_id"] for r in csv.DictReader(fh)})


def load_or_fetch(ids=None, force=False, cache_path=None, batch=BATCH):
    """Return the SMILES map, reading the on-disk cache unless `force`."""
    cache = {} if force else read_cache(cache_path)
    if ids is None:
        ids = compound_ids()
    todo = [c for c in ids if c not in cache or cache[c] is None]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        fetch_batch(chunk, cache)
        print(f"  {min(i + batch, len(todo))}/{len(todo)}", file=sys.stderr)
        time.sleep(0.3)
    write_cache(cache, cache_path)
    return cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="ignore the cache and re-query ChEMBL")
    args = ap.parse_args(argv)
    cache = load_or_fetch(force=args.force)
    have = sum(1 for v in cache.values() if v)
    print(f"{have}/{len(cache)} compounds have a canonical SMILES")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
