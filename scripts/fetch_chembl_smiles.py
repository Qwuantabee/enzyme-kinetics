"""Cache canonical SMILES for the compounds in the local Ki tables.

    python scripts/fetch_chembl_smiles.py [--force]

Writes data/chembl_dhfr/ecoli_dhfr_smiles.csv (molecule_chembl_id,
canonical_smiles) covering every compound that appears in
data/chembl_dhfr/ecoli_dhfr_ki.csv.

Why a separate step rather than folding the lookups into the analysis:
the structure-affinity script (scripts/qsar_congeneric.py) has to be
re-runnable offline, and ChEMBL is a shared public service. Fetch once,
cache to disk, and let every later run read the cache.

The output is ~20 KB of SMILES, small enough to commit under the same
exception the Ki and IC50 tables already use, so the descriptor analysis runs
on a fresh clone with no network. Re-running this is cheap: 266 Ki records,
219 distinct compounds, one request each against ChEMBL's REST API (no key),
with a short sleep between requests.
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "chembl_dhfr"
OUT = DATA_DIR / "ecoli_dhfr_smiles.csv"
BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _get_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as fh:
                return json.load(fh)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_smiles(chembl_id, cache):
    """Canonical SMILES for one ChEMBL molecule, or None if unavailable."""
    if chembl_id in cache:
        return cache[chembl_id]
    try:
        data = _get_json(f"{BASE}/molecule/{chembl_id}.json")
    except Exception:
        cache[chembl_id] = None
        return None
    structures = data.get("molecule_structures") or {}
    cache[chembl_id] = structures.get("canonical_smiles")
    time.sleep(0.2)
    return cache[chembl_id]


def read_cache(path=None):
    """The on-disk SMILES map, empty if the cache has not been built."""
    path = Path(path) if path is not None else OUT
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["molecule_chembl_id"]: (r["canonical_smiles"] or None)
                for r in csv.DictReader(fh)}


def _write_cache(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_chembl_id", "canonical_smiles"])
        for cid in sorted(cache):
            w.writerow([cid, cache[cid] or ""])


def load_or_fetch(ids=None, force=False):
    """Return the SMILES map, reading the on-disk cache unless `force`."""
    cache = {} if force else read_cache()
    if ids is None:
        with (DATA_DIR / "ecoli_dhfr_ki.csv").open(newline="") as fh:
            ids = sorted({r["molecule_chembl_id"] for r in csv.DictReader(fh)})
    todo = [c for c in ids if c not in cache or cache[c] is None]
    for i, cid in enumerate(todo, 1):
        fetch_smiles(cid, cache)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", file=sys.stderr)
    _write_cache(cache, OUT)
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
