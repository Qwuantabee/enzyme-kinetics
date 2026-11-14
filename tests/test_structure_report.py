"""Review fix C17: the structure report states what its numbers are.

The README said methotrexate spans 'five orders of magnitude' (0.001-170000 nM
is 8.2), quoted 14 H-bonds and 9 salt bridges as if general (true for 1DDR
only) and counted 6 Ki/structure rows as if independent. These tests pin the
measured values and the counting rule.
"""
import math
from pathlib import Path

from enzkin import datasets, structure

ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "results_structure.txt").read_text()


def test_methotrexate_span_is_measured_not_five_orders():
    ki = [r["ki_nM"] for r in datasets.load_chembl_dhfr_ki(wild_type_only=False)
          if r["molecule_chembl_id"] == "CHEMBL34259"]
    assert math.log10(max(ki) / min(ki)) == __import__("pytest").approx(8.23, abs=0.01)
    assert math.log10(sorted(ki)[-2] / min(ki)) == __import__("pytest").approx(6.38, abs=0.01)
    assert "8.23-order-of-magnitude" in REPORT
    assert "5-order-of-magnitude" not in REPORT


def test_contact_counts_differ_by_structure_and_are_atom_pair_counts():
    counts = {}
    for pid in ("1DDR", "3DAU", "4DFR"):
        s = structure.load_structure(ROOT / "data" / "pdb" / f"{pid}.pdb")
        c = s.ligand_contacts("MTX")
        counts[pid] = (c["hbond"], c["salt_bridge"])
    assert counts == {"1DDR": (14, 9), "3DAU": (12, 11), "4DFR": (13, 11)}
    doc = structure.Structure.ligand_contacts.__doc__
    assert "PAIRS" in doc and "not 14 distinct hydrogen bonds" in doc
    assert "backbone atoms included" in doc


def test_report_states_the_six_rows_are_not_independent():
    assert "1 compound x 2 distinct Ki records x 3 structures" in REPORT
    assert "NOT independent observations" in REPORT
    assert "(protein atom, ligand atom) PAIRS" in REPORT


def test_site_residue_counts_per_structure():
    n = {}
    for pid in ("1DDR", "3DAU", "4DFR"):
        s = structure.load_structure(ROOT / "data" / "pdb" / f"{pid}.pdb")
        n[pid] = len(s.binding_site("MTX", chain="A"))
    assert n == {"1DDR": 18, "3DAU": 21, "4DFR": 18}
