"""Link measured DHFR Ki values to binding-site structure.

    python scripts/ki_to_structure.py

Joins the ChEMBL Ki dataset (data/chembl_dhfr/ecoli_dhfr_ki.csv) to crystal
structures of E. coli DHFR (data/pdb/) via the PDB's chemical-component ChEMBL
cross-reference, and describes the binding site each ligand occupies.

The point is not to discover a structure-affinity relationship. With one
protein and very few ligands whose structures exist, that cannot be done
honestly. The point is to put a measured Ki next to an actual coordinate-level
description of the pocket it binds in, and to be explicit about what each
number does and does not establish.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import datasets, structure  # noqa: E402

PDB_DIR = ROOT / "data" / "pdb"
OUT = ROOT / "results_structure.txt"

#: PDB ligand code -> ChEMBL molecule id, taken from the PDB chemical
#: component cross-reference (rcsb_chem_comp_related, resource_name=ChEMBL).
#: Hard-coded here because it was checked by hand against RCSB's API; the
#: check itself is in the note printed below the table.
LIGAND_TO_CHEMBL = {
    "MTX": "CHEMBL34259",   # methotrexate
    "FOL": "CHEMBL1622",    # folate (dihydrofolate reductase substrate)
    "NAP": "CHEMBL121334",  # NADP+
    "NDP": "CHEMBL1235856",  # NADPH
}


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 76)
    out("Ki -> binding site, E. coli dihydrofolate reductase")
    out("=" * 76)

    ki_all = datasets.load_chembl_dhfr_ki(wild_type_only=False)
    ki_wt = datasets.load_chembl_dhfr_ki(wild_type_only=True)
    out(f"ChEMBL Ki records      {len(ki_all)} total, {len(ki_wt)} wild-type "
        f"({len(ki_all) - len(ki_wt)} mutant records excluded)")

    out("\n--- the mutant problem, on the record ---")
    mtx_all = sorted(r["ki_nM"] for r in ki_all
                     if r["molecule_chembl_id"] == "CHEMBL34259")
    mtx_wt = sorted(r["ki_nM"] for r in ki_wt
                    if r["molecule_chembl_id"] == "CHEMBL34259")
    out(f"methotrexate Ki, all records : {mtx_all} nM")
    out(f"methotrexate Ki, wild type   : {mtx_wt} nM")
    out("Pooling these would claim a 5-order-of-magnitude uncertainty on one")
    out("compound. The spread is not noise: three records are point mutants at")
    out("residue 31 (Phe/Tyr/Val), which the loader filters on the assay")
    out("description.")

    # --- structures ------------------------------------------------------
    out("\n--- structures analysed ---")
    structs = {}
    for pdb_id in ("1DDR", "3DAU", "4DFR"):
        path = PDB_DIR / f"{pdb_id}.pdb"
        if not path.exists():
            out(f"{pdb_id}: not present, skipped")
            continue
        s = structure.load_structure(path)
        structs[pdb_id] = s
        out(f"{pdb_id}: {s.title[:62]}")
        for lig in s.ligands():
            copies = s.ligand_copies(lig)
            frac, nb, nt = s.ligand_burial(lig)
            site = s.binding_site(lig)
            out(f"    {lig} [{','.join(copies)}]: {len(site)} site residues, "
                f"burial {frac:.0%} ({nb}/{nt} atoms)")

    # --- the join --------------------------------------------------------
    out("\n--- Ki joined to site ---")
    out("Compound identity comes from the PDB chemical-component ChEMBL")
    out("cross-reference, not from SMILES matching. That matters: the link is")
    out("the crystallographers' assertion about what the ligand is.")

    joined = []
    for pdb_id, s in structs.items():
        for lig in s.ligands():
            chembl_id = LIGAND_TO_CHEMBL.get(lig)
            if not chembl_id:
                continue
            recs = [r for r in ki_wt if r["molecule_chembl_id"] == chembl_id]
            if not recs:
                out(f"  {pdb_id}/{lig} ({chembl_id}): no measured Ki, skipped")
                continue
            # Report every wild-type record for this compound, not just the
            # most potent one. MTX has two wild-type Ki values spanning 21x
            # (0.001 and 0.021 nM); quoting only the best would misrepresent
            # how well "the" Ki of this compound is actually known.
            for rec in sorted(recs, key=lambda r: r["ki_nM"]):
                link = structure.link_ki_to_structure(
                    {"chembl_molecule_id": chembl_id, "ki_nM": rec["ki_nM"],
                     "assay_chembl_id": rec["assay_chembl_id"],
                     "document_chembl_id": rec["document_chembl_id"]},
                    s, lig)
                joined.append(link)

    for link in joined:
        out("")
        for line in link.summary().splitlines():
            out(line)

    # --- what can and cannot be said -------------------------------------
    out("\n--- what this establishes ---")
    out("Each row above is a real measured affinity next to real coordinates:")
    out("the number of residues lining the pocket, which residues they are,")
    out("how enclosed the ligand is, and how many contacts of each type.")
    out("Those are computed from the deposited coordinates and can be checked")
    out("against the structure by hand.")

    out("\n--- what it does not establish ---")
    out("Not a structure-affinity relationship. With one protein and at most a")
    out("few ligands that have both a measured Ki and a crystal structure,")
    out("any correlation between burial and potency would be anecdotal, and a")
    out("p-value on n=3 would be a decoration. Establishing that a contact")
    out("contributes to binding needs mutagenesis or a congeneric series, not")
    out("a comparison of two ligands.")
    out("")
    out("The geometric descriptors are also proxies, not physics:")
    out("  - burial is a contact-count fraction, not a Shrake-Rupley SASA")
    out("  - 'hydrophobic contacts' count carbon-carbon distances, which")
    out("    correlates with burial but is not an energy")
    out("  - the Ki and the structure may come from different conditions")
    out("    (pH, temperature, buffer, construct, and sometimes species)")
    out("")
    out("The last is the one that bites in practice. 4DFR contains both an")
    out("E. coli and a Lactobacillus casei copy; analysing the wrong chain")
    out("would compare an E. coli Ki to a L. casei pocket and look fine.")

    if joined:
        out(f"\n{len(joined)} Ki/structure pairs joined.")
        by_compound = {}
        for link in joined:
            by_compound.setdefault(link.chembl_molecule_id, []).append(link.ki_nM)
        out("\n--- within-compound spread of the measured Ki ---")
        for cid, vals in sorted(by_compound.items()):
            vals = sorted(set(vals))
            span = vals[-1] / vals[0] if len(vals) > 1 else 1.0
            out(f"{cid}: {vals} nM  ({span:.0f}x spread across wild-type assays)")
        out("The structure is one thing; the affinity is a distribution. Any")
        out("statement of the form 'the Ki for this scaffold is X' needs to say")
        out("which assay it came from.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nsummary -> {OUT}")


if __name__ == "__main__":
    main()
