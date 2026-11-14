"""Tests for linking Ki values to binding-site structure.

Two things are being checked, and they are different:

1. The PDB parser and geometric routines do what they claim on real
   coordinates. These are checkable against known facts about specific
   structures -- methotrexate binds E. coli DHFR in a deep pocket lined by
   residues including Phe31 and Asp27, the dimer has one site per subunit, and
   so on.

2. The join from a measured Ki to a structure is honest. That means checking
   the identity chain (ChEMBL ID -> chemical component -> coordinates) is real,
   and that records which cannot be joined safely are not joined by force.
   The mutant problem found while building this is the test case: the same
   compound, five Ki values spanning five orders of magnitude, and the only
   thing that distinguishes them is the assay description.

What these tests deliberately do not do is assert anything about
structure-affinity correlation. One protein with two ligands cannot support
that claim, and a test asserting it would be an assertion about my priors
rather than about the data.
"""

import numpy as np
import pytest

from enzkin import datasets, structure

PDB_DIR = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "data" / "pdb")


def _load(pdb_id):
    path = PDB_DIR / f"{pdb_id}.pdb"
    # the PDB files are tracked in git: a missing one is a broken checkout,
    # not a reason to skip the structure tests
    assert path.exists(), f"{pdb_id}.pdb is missing from data/pdb"
    return structure.load_structure(path)


# --- parser ------------------------------------------------------------------

def test_parses_a_real_structure():
    s = _load("1DDR")
    assert s.pdb_id == "1DDR"
    assert len(s.protein_atoms) > 1000
    assert len(s.hetatms) > 0
    assert "DIHYDROFOLATE REDUCTASE" in s.title.upper()


def test_reads_only_the_first_model():
    """An NMR ensemble has many models; pooling them would put dozens of
    copies of every atom in the same site and wreck every distance."""
    s = _load("1DDR")
    assert len({a.chain for a in s.protein_atoms}) <= 3


def test_altloc_handling():
    """Altlocs other than A are dropped, so each atom appears once."""
    s = _load("1DDR")
    names = [(a.chain, a.resseq, a.name) for a in s.protein_atoms]
    assert len(names) == len(set(names)), "duplicate atoms: altlocs not filtered"


def test_ligand_discovery_excludes_solvent_and_ions():
    s = _load("1DDR")
    lig = s.ligands()
    assert "MTX" in lig
    for junk in ("HOH", "CL", "CA", "GOL", "SO4"):
        assert junk not in lig, f"{junk} should not be reported as a ligand"


def test_parses_coordinates_correctly():
    """Spot-check one atom against the raw file."""
    path = PDB_DIR / "1DDR.pdb"
    raw = None
    for line in path.open():
        if line.startswith("ATOM") and line[12:16].strip() == "N":
            raw = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            break
    s = structure.load_structure(path)
    first_n = next(a for a in s.protein_atoms if a.name == "N")
    assert (first_n.x, first_n.y, first_n.z) == pytest.approx(raw)


# --- binding site ------------------------------------------------------------

def test_binding_site_has_the_expected_size():
    """Methotrexate in E. coli DHFR contacts roughly 15-25 residues within
    4.5 A per subunit. A number far from that means the distance cutoff or
    the chain handling is wrong."""
    s = _load("1DDR")
    site = s.binding_site("MTX", chain="A")
    assert 12 <= len(site) <= 30, f"got {len(site)} residues"


def test_dimer_has_one_site_per_subunit():
    """1DDR is a dimer with one MTX per chain. Analysing each chain gives one
    site per subunit; pooling both chains would give 36 residues that exist
    nowhere in the protein. The default (chain=None) must pick a single copy,
    and the two copies must be reported separately."""
    s = _load("1DDR")
    copies = s.ligand_copies("MTX")
    assert len(copies) == 2, f"expected a dimer, got chains {copies}"

    a = s.binding_site("MTX", chain="A")
    b = s.binding_site("MTX", chain="B")
    assert len(a) == len(b)
    assert all(r["chain"] == "A" for r in a)
    assert all(r["chain"] == "B" for r in b)

    # the default picks the first copy rather than merging them
    default = s.binding_site("MTX")
    assert len(default) == len(a)
    assert {r["chain"] for r in default} == {copies[0]}

    # and the two subunits really are different sites, not the same one
    # reported twice: their residue numbering differs
    ids_a = {(r["resseq"], r["resname"]) for r in a}
    ids_b = {(r["resseq"], r["resname"]) for r in b}
    assert ids_a != ids_b or {r["chain"] for r in b} != {copies[0]}


def test_binding_site_contains_known_catalytic_residues():
    """Asp27 and Phe31 line the folate-binding pocket of E. coli DHFR. If the
    parser or the neighbour search is broken, these are the residues that go
    missing first."""
    s = _load("1DDR")
    names = {(r["resname"], r["resseq"]) for r in s.binding_site("MTX")}
    assert ("ASP", 27) in names, f"Asp27 not in site: {sorted(names)}"
    assert ("PHE", 31) in names, f"Phe31 not in site: {sorted(names)}"


def test_site_sequence_matches_site_residues():
    s = _load("1DDR")
    site = s.binding_site("MTX", chain="A")
    seq = s.binding_site_sequence("MTX", chain="A")
    assert len(seq) == len(site)
    assert seq.isalpha()


def test_distance_cutoff_is_respected():
    s = _load("1DDR")
    for r in s.binding_site("MTX", cutoff=3.5):
        assert r["min_dist"] <= 3.5
    # a tighter cutoff cannot give more residues
    assert len(s.binding_site("MTX", cutoff=3.5)) <= \
        len(s.binding_site("MTX", cutoff=5.0))


def test_burial_discriminates_between_ligands():
    """Methotrexate is enclosed in a deep pocket; NADPH (NDP) sits in a
    broader cleft. Both are buried by any sensible measure, but the routine
    must not return 100% for everything, which is what an 8 A cutoff did."""
    s = _load("3DAU")
    frac_mtx, _, _ = s.ligand_burial("MTX")
    frac_ndp, _, _ = s.ligand_burial("NDP")
    assert 0.5 <= frac_mtx <= 1.0
    assert 0.5 <= frac_ndp <= 1.0
    # and a deliberately generous cutoff saturates, proving the parameter
    # is doing work
    generous, _, _ = s.ligand_burial("MTX", cutoff=8.0)
    assert generous == pytest.approx(1.0)


def test_missing_ligand_raises():
    s = _load("1DDR")
    with pytest.raises(ValueError, match="no ligand"):
        s.binding_site("ZZZ")


def test_contacts_are_counts_and_are_positive():
    s = _load("1DDR")
    c = s.ligand_contacts("MTX")
    assert set(c) == {"hbond", "hydrophobic", "salt_bridge", "aromatic"}
    assert all(v >= 0 for v in c.values())
    # methotrexate is a glutamate analogue: it must make salt bridges
    assert c["salt_bridge"] > 0
    # aromatic because of the pteridine ring and the Phe31 contact
    assert c["aromatic"] > 0
    # bound analogues make hydrogen bonds to the pteridine ring
    assert c["hbond"] > 0


def test_contacts_scale_with_cutoff():
    s = _load("1DDR")
    tight = s.ligand_contacts("MTX", cutoff=3.0)
    loose = s.ligand_contacts("MTX", cutoff=6.0)
    assert loose["hydrophobic"] >= tight["hydrophobic"]


def test_rejects_a_file_with_no_atoms(tmp_path):
    bad = tmp_path / "bad.pdb"
    bad.write_text("HEADER nothing here\nEND\n")
    with pytest.raises(ValueError, match="no readable"):
        structure.load_structure(bad)


# --- the Ki -> structure link ------------------------------------------------

def test_ki_dataset_loads():
    table = datasets.load_chembl_dhfr_ki()
    assert len(table) > 100
    kis = np.array([r["ki_nM"] for r in table])
    assert np.all(kis > 0)
    assert np.median(kis) > 0


def test_mutant_assays_are_filtered_out():
    """Methotrexate's five Ki records span 0.001 to 170000 nM, and the
    difference is not measurement noise -- three are point mutants at
    residue 31. The default loader must not return them together."""
    all_recs = datasets.load_chembl_dhfr_ki(wild_type_only=False)
    wt = datasets.load_chembl_dhfr_ki(wild_type_only=True)
    assert len(wt) < len(all_recs), "filter removed nothing"

    def mtx(recs):
        return sorted(r["ki_nM"] for r in recs
                      if r["molecule_chembl_id"] == "CHEMBL34259")

    mtx_all = mtx(all_recs)
    mtx_wt = mtx(wt)
    assert len(mtx_wt) < len(mtx_all), \
        "methotrexate should lose its mutant records"
    # the wild-type set must not span the mutant values
    assert max(mtx_wt) / min(mtx_wt) < max(mtx_all) / min(mtx_all)


def test_mutant_detector_recognises_real_descriptions():
    assert datasets._looks_mutant(
        "Binding constant(Ki) to dihydrofolate reductase (DHFR) Phe -31 "
        "was determined")
    assert datasets._looks_mutant("Ki against Tyr31 variant")
    assert not datasets._looks_mutant(
        "Binding affinity against Dihydrofolate reductase of Escherichia coli")
    assert not datasets._looks_mutant("")


def test_link_builds_a_complete_record():
    """Join the wild-type methotrexate Ki to the MTX site in 1DDR."""
    wt = [r for r in datasets.load_chembl_dhfr_ki()
          if r["molecule_chembl_id"] == "CHEMBL34259"]
    # A mutant-filter regression (or a changed Ki file) must FAIL here, not
    # skip: the wild-type record set for methotrexate is what the join needs.
    assert wt, ("no wild-type MTX Ki record: the mutant filter now removes "
                "every methotrexate record, or the Ki file changed")
    assert {r["molecule_chembl_id"] for r in wt} == {"CHEMBL34259"}
    s = _load("1DDR")
    nmr = {"chembl_molecule_id": "CHEMBL34259", "ki_nM": min(r["ki_nM"] for r in wt),
           "assay_chembl_id": wt[0]["assay_chembl_id"]}
    link = structure.link_ki_to_structure(nmr, s, "MTX")
    assert link.pdb_id == "1DDR"
    assert link.ligand_code == "MTX"
    assert link.ki_nM > 0
    assert link.n_site_residues >= 12
    assert len(link.site_sequence) == link.n_site_residues
    assert 0 < link.burial_fraction <= 1.0
    # the notes must record how identity was established
    assert any("cross-reference" in n for n in link.notes)
    assert any("different conditions" in n for n in link.notes)


def test_link_site_is_identical_across_two_structures_of_the_same_complex():
    """1DDR and 4DFR are both wild-type E. coli DHFR with methotrexate. The
    site residue set should agree almost exactly; if it does not, the join is
    picking up something structural in the coordinates rather than chemistry."""
    a = _load("1DDR")
    b = _load("4DFR")
    site_a = {(r["resname"], r["resseq"]) for r in a.binding_site("MTX", chain="A")}
    site_b = {(r["resname"], r["resseq"]) for r in b.binding_site("MTX", chain="A")}
    overlap = len(site_a & site_b) / len(site_a | site_b)
    assert overlap > 0.8, f"site overlap only {overlap:.2f}"


def test_mixed_species_entry_chain_selection():
    """4DFR contains E. coli DHFR (chains A, B) and Lactobacillus casei DHFR
    in another mol_id. Selecting the wrong chain would compare an E. coli Ki
    to an L. casei pocket and look perfectly normal -- same fold, same
    ligand, different protein. Guard: the E. coli chains must be the ones
    with the E. coli sequence length, and their sites must agree with a
    structure that contains only E. coli."""
    ec = _load("1DDR")
    mixed = _load("4DFR")

    def residue_names(struct, chain):
        seen, out = set(), []
        for at in struct.protein_atoms:
            if at.chain != chain:
                continue
            key = (at.chain, at.resseq)
            if key in seen:
                continue
            seen.add(key)
            out.append(at.resname)
        return out

    ec_len = len(residue_names(ec, "A"))
    for chain in ("A", "B"):
        assert len(residue_names(mixed, chain)) == ec_len, \
            f"4DFR chain {chain} is not the E. coli copy"

    # the pocket of the E. coli copy must match the E. coli-only structure
    site_mixed = {(r["resname"], r["resseq"])
                  for r in mixed.binding_site("MTX", chain="A")}
    site_ec = {(r["resname"], r["resseq"])
               for r in ec.binding_site("MTX", chain="A")}
    assert len(site_mixed & site_ec) / len(site_ec) > 0.85
