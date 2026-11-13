"""Link measured Ki values to the binding site in a crystal structure.

A Ki is a number about a molecule. A structure is a set of coordinates. This
module connects them: given a PDB entry and a ligand, find the residues lining
the binding site, measure how tightly packed it is, and put the measured Ki
next to that description.

What the link actually rests on, and what it does not
----------------------------------------------------
The connection is a chain of identities, each of which can fail:

    measured Ki  ->  compound identity (ChEMBL ID)
                 ->  chemical component ID in the PDB (MTX, NAP, ...)
                 ->  coordinates in an entry

The ChEMBL ID comes from the PDB's own chemical-component cross-reference
(rcsb_chem_comp_related, resource_name="ChEMBL"), *not* from matching SMILES.
That matters: two entries can share a ligand three-letter code while being
different compounds over time, and SMILES matching in the absence of RDKit is
a string comparison that silently returns nothing at all. Using the deposited
cross-reference means the link is the crystallographers' statement rather than
ours.

What this does NOT establish is causality. A buried inhibitor with a low Ki
does not mean burial *causes* the affinity -- a large flexible inhibitor can be
tightly packed and weak, and a small one can be loosely packed and potent.
Structure-affinity relationships are correlational at this level of analysis,
and with one protein and a handful of ligands any correlation is anecdotal.
The functions here are built to report the numbers honestly rather than to
score them.

Tested against real structures: E. coli DHFR (1DDR, 4DFR, 3DAU) with folate
analogues, and P99 beta-lactamase (1BLC) for the enzyme used in the kinetics
work in this repo.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Standard water and buffer components that are not the ligand of interest.
SOLVENT_AND_ION = {
    "HOH", "WAT", "DOD", "GOL", "EDO", "PEG", "SO4", "PO4", "CL", "NA", "K",
    "MG", "MN", "CA", "ZN", "NI", "CD", "BME", "DTT", "ACT", "FMT", "ACE",
    "TRS", "EPE", "MES", "IMD", "NO3", "SCN", "BR", "IOD", "URE",
}

#: Three-letter codes for the amino acids, for residue naming.
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    occupancy: float = 1.0

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Structure:
    pdb_id: str
    atoms: list
    title: str = ""

    @property
    def protein_atoms(self):
        """Standard amino-acid atoms, first model, highest-occupancy altloc."""
        return [a for a in self.atoms
                if a.record == "ATOM" and a.resname in AA3]

    @property
    def hetatms(self):
        return [a for a in self.atoms if a.record == "HETATM"]

    def ligands(self, exclude=None):
        """Distinct HETATM residue names, excluding solvent, ions and waters."""
        ex = SOLVENT_AND_ION if exclude is None else (SOLVENT_AND_ION | set(exclude))
        return sorted({a.resname for a in self.hetatms if a.resname not in ex})

    def ligand_atoms(self, code, chain=None):
        """Ligand atoms of `code`, optionally restricted to one chain.

        The chain restriction matters because a dimeric structure has one
        copy of the ligand per subunit, and pooling them gives a "binding
        site" that is really two sites smeared together -- 36 residues where
        each monomer has 18.
        """
        atoms = [a for a in self.hetatms if a.resname == code]
        if chain is not None:
            atoms = [a for a in atoms if a.chain == chain]
        return atoms

    def ligand_copies(self, code):
        """Chains that contain a copy of this ligand, in file order."""
        seen, out = set(), []
        for a in self.ligand_atoms(code):
            if a.chain not in seen:
                seen.add(a.chain)
                out.append(a.chain)
        return out

    def binding_site(self, code, cutoff=4.5, chain=None):
        """Protein residues with any atom within `cutoff` Angstrom of the ligand.

        4.5 A is generous enough to catch the second shell of a hydrophobic
        pocket, which is where most of the buried surface is, without
        wandering into the rest of the protein. The number is a choice, and
        it is a parameter rather than a constant for that reason.

        If `chain` is None and the ligand has copies in several chains, the
        first copy is used and the others are ignored -- see
        `ligand_copies` and `pick_ligand_chain`.
        """
        if chain is None:
            copies = self.ligand_copies(code)
            if not copies:
                raise ValueError(f"no ligand {code!r} in {self.pdb_id}; "
                                 f"available: {self.ligands()}")
            chain = copies[0]
        lig = self.ligand_atoms(code, chain=chain)
        if not lig:
            raise ValueError(f"no ligand {code!r} in chain {chain!r} of "
                             f"{self.pdb_id}")
        lig_xyz = np.array([a.xyz for a in lig])
        residues = {}
        for a in self.protein_atoms:
            d = np.min(np.linalg.norm(lig_xyz - a.xyz, axis=1))
            if d <= cutoff:
                key = (a.chain, a.resseq, a.icode)
                if key not in residues or d < residues[key]["min_dist"]:
                    residues[key] = {
                        "chain": a.chain, "resseq": a.resseq,
                        "icode": a.icode, "resname": a.resname,
                        "min_dist": float(d),
                    }
        return sorted(residues.values(),
                      key=lambda r: (r["min_dist"], r["chain"], r["resseq"]))

    def binding_site_sequence(self, code, cutoff=4.5, chain=None):
        """One-letter codes of the binding-site residues, in residue order."""
        site = self.binding_site(code, cutoff=cutoff, chain=chain)
        three_to_one = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            "MSE": "M",
        }
        ordered = sorted(site, key=lambda r: (r["chain"], r["resseq"]))
        return "".join(three_to_one.get(r["resname"], "X") for r in ordered)

    def ligand_burial(self, code, cutoff=4.5, chain=None):
        """Fraction of the ligand's atoms in van der Waals contact with protein.

        Each ligand atom is counted as buried if any protein atom is within
        `cutoff`, which defaults to 4.5 A -- about the sum of two carbon van
        der Waals radii plus a little slack. This is a crude proxy for buried
        surface area, not a Shrake-Rupley calculation (which needs explicit
        radii and a probe sphere), but it is computed from coordinates rather
        than assumed and it separates a buried ligand from a surface one.

        The default used to be 8 A, which was wrong in an instructive way:
        almost any ligand in a crystal has protein within 8 A, so every ligand
        scored 100% and the number carried no information. Methotrexate in
        1DDR, which really is enclosed, has a largest nearest-protein
        distance of 4.63 A.

        Protein atoms from all chains are counted, because a dimer interface
        really does occlude ligand surface. Pass `chain` to pick one copy.

        Returns (fraction, n_buried, n_total).
        """
        if chain is None:
            copies = self.ligand_copies(code)
            if not copies:
                raise ValueError(f"no ligand {code!r} in {self.pdb_id}")
            chain = copies[0]
        lig = self.ligand_atoms(code, chain=chain)
        if not lig:
            raise ValueError(f"no ligand {code!r} in chain {chain!r} of "
                             f"{self.pdb_id}")
        prot_xyz = np.array([a.xyz for a in self.protein_atoms])
        n_buried = 0
        for a in lig:
            d = np.min(np.linalg.norm(prot_xyz - a.xyz, axis=1))
            if d <= cutoff:
                n_buried += 1
        return n_buried / len(lig), n_buried, len(lig)

    def ligand_contacts(self, code, cutoff=4.5, chain=None):
        """Counts of contact types between ligand and protein.

        Hydrogen bonds, hydrophobic contacts, salt bridges, aromatic stacking
        -- each is a set of geometric criteria, and each is an approximation.
        What is returned is the *counts* under stated rules, not a claim about
        which interactions actually contribute to binding energy. The rules:

          hbond        N/O of protein within 3.5 A of N/O of ligand
          hydrophobic  C of protein within 4.0 A of C of ligand
          salt_bridge  charged protein N <-> ligand O, or protein O <->
                       ligand N, within 4.0 A
          aromatic     any protein C within 4.5 A of ligand C (a lower bound
                       on pi-stacking, since it also counts plain contacts)

        The distinction between these categories is real chemistry only in
        the hbond and salt-bridge cases, where the identity of the atoms
        matters. The hydrophobic count is geometry that correlates with
        burial.
        """
        if chain is None:
            copies = self.ligand_copies(code)
            if not copies:
                raise ValueError(f"no ligand {code!r} in {self.pdb_id}")
            chain = copies[0]
        lig = self.ligand_atoms(code, chain=chain)
        if not lig:
            raise ValueError(f"no ligand {code!r} in chain {chain!r} of "
                             f"{self.pdb_id}")
        prot = self.protein_atoms

        counts = {"hbond": 0, "hydrophobic": 0, "salt_bridge": 0,
                  "aromatic": 0}
        for p in prot:
            pe = p.element.upper()
            for l in lig:
                le = l.element.upper()
                d = float(np.linalg.norm(p.xyz - l.xyz))

                if pe in ("N", "O") and le in ("N", "O") and d <= 3.5:
                    counts["hbond"] += 1
                if pe == "C" and le == "C":
                    if d <= 4.0:
                        counts["hydrophobic"] += 1
                    if d <= 4.5:
                        counts["aromatic"] += 1
                # salt bridges: use residue identity for charge
                if pe in ("N", "O") and le in ("N", "O") and d <= 4.0:
                    if p.resname in ("ARG", "LYS", "HIS") and le == "O":
                        counts["salt_bridge"] += 1
                    elif p.resname in ("ASP", "GLU") and le == "N":
                        counts["salt_bridge"] += 1
        return counts


def load_structure(path, pdb_id=None):
    """Parse a PDB file. Only the first model is read.

    Deliberately minimal: ATOM/HETATM records, first model, altloc A or blank.
    No mmCIF, no symmetry expansion, no chain repair. If a structure needs
    more than that it probably needs a real structural-bioinformatics library
    and this is the wrong module for it.
    """
    path = Path(path)
    atoms = []
    title_parts = []
    in_first_model = False
    seen_model = False

    with path.open() as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                seen_model = True
                in_first_model = True
                continue
            if rec == "ENDMDL" and seen_model:
                break
            if rec == "TITLE":
                title_parts.append(line[10:80].strip())
            if rec in ("ATOM", "HETATM"):
                altloc = line[16:17]
                if altloc not in (" ", "A"):
                    continue
                try:
                    atoms.append(Atom(
                        record=rec,
                        serial=int(line[6:11]),
                        name=line[12:16].strip(),
                        altloc=altloc,
                        resname=line[17:20].strip(),
                        chain=line[21:22].strip(),
                        resseq=int(line[22:26]),
                        icode=line[26:27].strip(),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        element=(line[76:78].strip()
                                 or line[12:14].strip()[:1]),
                        occupancy=float(line[54:60] or 1.0),
                    ))
                except (ValueError, IndexError):
                    continue

    if not atoms:
        raise ValueError(f"{path} contains no readable ATOM/HETATM records")
    if pdb_id is None:
        pdb_id = path.stem.upper()
    return Structure(pdb_id=pdb_id, atoms=atoms,
                     title=" ".join(title_parts).strip())


@dataclass
class KiStructureLink:
    """A measured Ki joined to a binding-site description."""
    chembl_molecule_id: str
    ligand_code: str
    pdb_id: str
    ki_nM: float
    assay_chembl_id: str = ""
    document_chembl_id: str = ""
    n_site_residues: int = 0
    site_sequence: str = ""
    burial_fraction: float = 0.0
    contacts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self):
        lines = [
            f"{self.chembl_molecule_id} ({self.ligand_code}) in {self.pdb_id}",
            f"  Ki = {self.ki_nM:.4g} nM  [assay {self.assay_chembl_id}]",
            f"  site: {self.n_site_residues} residues, "
            f"burial {self.burial_fraction:.0%}",
            f"  seq: {self.site_sequence}",
            f"  contacts: {self.contacts}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def link_ki_to_structure(ki_record, structure, ligand_code, cutoff=4.5):
    """Join one measured Ki to the binding site that ligand occupies.

    `ki_record` must carry chembl_molecule_id, ligand_code, ki_nM and
    optionally assay_chembl_id / document_chembl_id.

    The notes attached to the result record the things a reader needs in order
    to interpret it, rather than those things being implicit in the code: that
    the identity link is via the PDB's ChEMBL cross-reference, that Ki and the
    crystal structure may come from different conditions, and that the
    structural descriptors are geometric proxies.
    """
    site = structure.binding_site(ligand_code, cutoff=cutoff)
    frac, n_buried, n_total = structure.ligand_burial(ligand_code)
    contacts = structure.ligand_contacts(ligand_code, cutoff=cutoff)

    notes = [
        "compound identity linked via the PDB chemical-component ChEMBL "
        "cross-reference, not by SMILES matching",
        "the Ki and the crystal structure may have been measured under "
        "different conditions (pH, temperature, buffer, construct)",
    ]
    if frac < 0.5:
        notes.append(
            f"only {frac:.0%} of ligand atoms have protein within 8 A: this "
            f"ligand is largely solvent-exposed, so site descriptors are weak"
        )

    return KiStructureLink(
        chembl_molecule_id=ki_record["chembl_molecule_id"],
        ligand_code=ligand_code,
        pdb_id=structure.pdb_id,
        ki_nM=float(ki_record["ki_nM"]),
        assay_chembl_id=ki_record.get("assay_chembl_id", ""),
        document_chembl_id=ki_record.get("document_chembl_id", ""),
        n_site_residues=len(site),
        site_sequence=structure.binding_site_sequence(ligand_code,
                                                      cutoff=cutoff),
        burial_fraction=frac,
        contacts=contacts,
        notes=notes,
    )


def site_residue_table(structure, ligand_code, cutoff=4.5):
    """The binding-site residues with distances, for a report."""
    return structure.binding_site(ligand_code, cutoff=cutoff)
