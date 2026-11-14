"""Loading real published datasets, as opposed to simulated ones.

Everything else in this package was validated against data with known
answers. That is how you find out whether the fitting code is correct, but it
says nothing about whether the models fit real experiments -- which have
baseline drift, a mixing dead time, an enzyme that is only partly active, and
a substrate that depletes while you watch it.

This module loads the experimental time-courses published with:

    Frere J-M, Verlaine O, Matagne A (2023)
    "The measurement of true initial rates is not always absolutely necessary
     to estimate enzyme kinetic parameters"
    Scientific Reports 13:14994.  doi:10.1038/s41598-023-41805-y
    Data: doi:10.5281/zenodo.10059024  (CC-BY 4.0)

The system is the Enterobacter cloacae P99 class C beta-lactamase hydrolysing
nitrocefin, followed spectrophotometrically at 462 nm. That dataset is a good
fit for this package because it is a genuine [S] series with five substrate
concentrations, several replicate traces at each, and -- importantly -- the
reaction is followed to 40-64% substrate conversion, so the same traces can
be analysed either as initial rates or as full progress curves. The paper's
whole argument is that the second is not worse than the first, and having the
raw traces means that claim can be checked here rather than taken on trust.

Provenance note: `data/raw/frere2023.zip` is the Zenodo archive as downloaded
(136 MB, tracked out of the repo by .gitignore). `data/frere2023_nitrocefin/`
holds the 18 experimental CSVs extracted from it, with the instrument's
semicolon header stripped down to two numeric columns and the thousands-style
time values ("1.008.333" meaning 1.008333 min) normalised. No values were
altered. `scripts/prepare_frere2023.py` regenerates them from the archive.
"""

from pathlib import Path

import numpy as np

# Where the extracted CSVs live, relative to the repo root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "frere2023_nitrocefin"

# ---------------------------------------------------------------------------
# Conditions, transcribed from _Experimental.summary.pdf inside the archive.
# ---------------------------------------------------------------------------
#: Final enzyme concentration in the cuvette. Class C beta-lactamase, P99.
ENZYME_CONC_M = 0.6e-9          # mol/L
#: Molar absorptivity of the nitrocefin hydrolysis product at 462 nm.
DELTA_EPSILON = 15000.0         # M^-1 cm^-1
PATHLENGTH_CM = 1.0
TEMPERATURE_C = 30.0
BUFFER = "10 mM HEPES pH 8.2, 0.2 M NaCl, 50 ug/mL BSA"

#: Published literature values for this enzyme/substrate pair, quoted in the
#: paper as the reference the analysis is judged against. These are NOT
#: measured here -- they are the external benchmark, and are labelled as such
#: wherever they appear.
PUBLISHED_KCAT_S = 780.0        # s^-1  (ref 31 of the paper)
PUBLISHED_KM_M = 25.0e-6        # mol/L

#: file stem -> starting substrate concentration in uM.
#: Transcribed from the summary PDF. The "*-1" files in the archive are
#: duplicate injections of the same trace and the PDF table lists them once,
#: so they are deliberately excluded here (not silently dropped -- see
#: DUPLICATE_FILES).
SUBSTRATE_UM = {
    "vo201": 11.1, "vo202": 11.1, "vo203": 11.1, "vo204": 11.1, "vo205": 11.1,
    "vo401": 22.1, "vo402": 22.1, "vo403": 22.1, "vo404": 22.1,
    "vo601": 35.0, "vo602": 35.0, "vo603": 35.0, "vo604": 35.0,
    "vo801": 52.2, "vo802": 52.2,
    "vo1001": 64.0, "vo1002": 64.0, "vo1003": 64.0,
}

#: Extra traces present in the Zenodo archive that repeat a trace above.
DUPLICATE_FILES = {"vo203-1": "vo203", "vo403-1": "vo403", "vo404-1": "vo404"}


def load_trace(stem, data_dir=None):
    """Read one time-course: returns (time_min, absorbance) as float arrays.

    The instrument writes absorbance rounded to 4 decimals at 462 nm; that
    quantisation (~1e-4 AU, about 0.007 uM of product) is the real noise floor
    and is why the very first points of a trace look almost flat.
    """
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    path = d / f"{stem}.csv"
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path} is not a two-column time-course")
    ok = np.isfinite(arr).all(axis=1)
    t, a = arr[ok, 0], arr[ok, 1]
    order = np.argsort(t)
    return t[order], a[order]


def absorbance_to_product_um(delta_a, epsilon=None, pathlength=None):
    """Convert an absorbance change to product concentration in uM.

    Beer-Lambert: dA = eps * [P] * l. With eps in M^-1 cm^-1 and l in cm,
    [P] comes out in mol/L; multiply by 1e6 for uM.
    """
    eps = DELTA_EPSILON if epsilon is None else epsilon
    l = PATHLENGTH_CM if pathlength is None else pathlength
    if eps <= 0 or l <= 0:
        raise ValueError("epsilon and pathlength must be positive")
    return np.asarray(delta_a, dtype=float) / (eps * l) * 1e6


def substrate_for(stem):
    """Starting substrate concentration for any archive trace, including the
    '-1' duplicates, which carry the concentration of the trace they repeat."""
    if stem in SUBSTRATE_UM:
        return SUBSTRATE_UM[stem]
    if stem in DUPLICATE_FILES:
        return SUBSTRATE_UM[DUPLICATE_FILES[stem]]
    raise KeyError(f"unknown trace {stem!r}")


def progress_curve(stem, data_dir=None):
    """One trace as (time_min, product_um, s0_um).

    Product is referenced to the first reading, which is the only baseline
    available -- the archive does not contain separate blank injections.
    Anything that drifts in the first seconds (mixing, temperature
    equilibration) therefore lands in this baseline, which is one reason
    progress-curve fits carry a fitted baseline term.
    """
    t, a = load_trace(stem, data_dir=data_dir)
    p = absorbance_to_product_um(a - a[0])
    return t, p, substrate_for(stem)


def initial_rate(stem, fraction=0.15, data_dir=None):
    """Slope over the early part of a trace, in uM/min.

    `fraction` is the fraction of *total* product formed that the window
    covers, matching how initial rates are taken from a real trace: you
    cannot know in advance how long the linear phase lasts, so you take a
    fixed early slice.

    0.15 is deliberately conservative. The point of the exercise is that the
    answer depends on this choice, which is the problem progress-curve
    fitting removes -- see scripts/fit_published.py, which reports the
    spread over several windows rather than one.
    """
    t, p, s0 = progress_curve(stem, data_dir=data_dir)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    total = p[-1] - p[0]
    if total <= 0:
        raise ValueError(f"{stem}: no net product formation")
    window = p <= p[0] + fraction * total
    if window.sum() < 5:
        raise ValueError(f"{stem}: fewer than 5 points in the {fraction:g} window")
    slope, intercept, r, se_slope = _linfit(t[window], p[window])
    return {
        "stem": stem,
        "s0_um": s0,
        "v_um_per_min": slope,
        "v_stderr": se_slope,
        "window_fraction": fraction,
        "window_points": int(window.sum()),
        "window_span_min": float(t[window][-1] - t[window][0]),
        "conversion_at_end": float(total / s0),
        "r2": r ** 2,
        "intercept_um": intercept,
    }


def _linfit(x, y):
    """Least-squares line with the standard error on the slope."""
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 points")
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx <= 0:
        raise ValueError("all x identical")
    slope = np.sum((x - xm) * (y - ym)) / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = np.sum(resid ** 2) / dof if dof > 0 else np.nan
    se = np.sqrt(s2 / sxx) if dof > 0 else np.nan
    ss_tot = np.sum((y - ym) ** 2)
    r = np.sqrt(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r), float(se)


def initial_rate_table(fraction=0.15, data_dir=None, pooled=True):
    """Every trace at a given window, as a list of dicts.

    pooled=True averages the replicates at each [S]0 and propagates their
    spread as the standard error of the mean, which is the form the
    Michaelis-Menten fit wants.
    """
    rows = [initial_rate(s, fraction=fraction, data_dir=data_dir)
            for s in SUBSTRATE_UM]
    if not pooled:
        return rows

    out = []
    for s0 in sorted({r["s0_um"] for r in rows}):
        group = [r for r in rows if r["s0_um"] == s0]
        v = np.array([g["v_um_per_min"] for g in group])
        out.append({
            "s0_um": s0,
            "v_um_per_min": float(v.mean()),
            "v_sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1
                     else float(group[0]["v_stderr"]),
            "n_replicates": len(v),
            "v_min": float(v.min()),
            "v_max": float(v.max()),
        })
    return out


def activation_fraction(kcat_measured_s, published=PUBLISHED_KCAT_S):
    """What fraction of the published kcat the assay actually delivered.

    A real enzyme preparation is rarely 100% active, and the quoted protein
    concentration is rarely the active-site concentration. Rather than
    pretending the difference is a fitting error, report it: this ratio is
    the product of purity and specific activity, and it is the honest
    interpretation of "our kcat came out lower than the literature".
    """
    if published <= 0:
        raise ValueError("published kcat must be positive")
    return kcat_measured_s / published


def kcat_from_vmax(vmax_um_per_min, enzyme_conc_m=None):
    """Turn a fitted Vmax into kcat in s^-1.

    Vmax in uM/min -> mol/L/s by *1e-6/60. Divide by the enzyme
    concentration in mol/L. This assumes one active site per enzyme molecule
    and a known active-site concentration, both of which are assumptions, not
    measurements -- see activation_fraction.
    """
    e = ENZYME_CONC_M if enzyme_conc_m is None else enzyme_conc_m
    if e <= 0:
        raise ValueError("enzyme concentration must be positive")
    return (vmax_um_per_min * 1e-6 / 60.0) / e


# ---------------------------------------------------------------------------
# ChEMBL: E. coli dihydrofolate reductase inhibition data
# ---------------------------------------------------------------------------
#: Target used for the IC50 dataset. E. coli DHFR, single-protein format.
CHEMBL_DHFR_TARGET = "CHEMBL1809"
CHEMBL_DHFR_DIR = (Path(__file__).resolve().parents[2]
                   / "data" / "chembl_dhfr")


def load_chembl_dhfr_ic50(data_dir=None):
    """Measured IC50 values for E. coli DHFR, as a list of dicts.

    Fetched from ChEMBL (target CHEMBL1809, standard_type=IC50,
    standard_relation='=') and vendored to
    data/chembl_dhfr/ecoli_dhfr_ic50.csv. 482 records over 353 compounds and
    42 publications -- a real inhibitor series, not a simulation.

    Why this dataset: it is what an IC50-to-Ki conversion actually operates
    on. The values span sub-nanomolar to millimolar, which is the range in
    which the arithmetic has to stay sane, and the records carry the molecule
    and document identifiers so any individual value can be traced back.

    What it lacks, and this matters: ChEMBL's activity records here do not
    carry the assay's substrate concentration or the enzyme concentration.
    Those are exactly the numbers Cheng-Prusoff needs inputs for. So this
    dataset can validate the arithmetic across a realistic range and cannot
    validate the conversion against measured Ki values. That limitation is
    recorded in README.md rather than papered over -- see
    tests/test_cheng_prusoff.py, which checks what is checkable.

    Returns dicts with keys: molecule_chembl_id, assay_chembl_id,
    document_chembl_id, ic50_nM, relation, pchembl.
    """
    import csv

    d = Path(data_dir) if data_dir is not None else CHEMBL_DHFR_DIR
    path = d / "ecoli_dhfr_ic50.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            value = float(r["ic50_nM"])
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        out.append({
            "molecule_chembl_id": r["molecule_chembl_id"],
            "assay_chembl_id": r["assay_chembl_id"],
            "document_chembl_id": r["document_chembl_id"],
            "ic50_nM": value,
            "relation": r.get("relation", "="),
            "pchembl": r.get("pchembl") or None,
        })
    return out


#: Assay descriptions that indicate a mutant enzyme rather than wild type.
#: This is a text filter and it is a blunt instrument, but the alternative --
#: trusting that every ChEMBL record for "E. coli DHFR" refers to the same
#: protein -- is worse. It was caught the hard way: five methotrexate Ki
#: records spanned 0.001 to 170000 nM, and reading the assay descriptions
#: showed three of them were Phe31/Tyr31/Val31 point mutants.
MUTANT_MARKERS = ("mutant", "Phe -31", "Tyr -31", "Val -31", "Phe31",
                  "Tyr31", "Val31", "variant", "Cys -")


def _looks_mutant(description):
    d = (description or "").lower()
    return any(m.lower() in d for m in MUTANT_MARKERS)


def load_chembl_dhfr_ki(data_dir=None, wild_type_only=True):
    """Measured Ki values for E. coli DHFR, as a list of dicts.

    Source: ChEMBL target CHEMBL1809, standard_type=Ki, 266 records over 219
    compounds, vendored to data/chembl_dhfr/ecoli_dhfr_ki.csv.

    `wild_type_only` (default) drops records whose assay description names a
    point mutant. This is not cosmetic. Methotrexate appears five times in
    this dataset with Ki values of 0.001, 0.021, 330, 2400 and 170000 nM; the
    three middle-high ones are Phe31, Tyr31 and Val31 mutants of the same
    residue, and only two are wild-type. Pooling them would make "the Ki of
    methotrexate for E. coli DHFR" meaningless, and any structure comparison
    built on it would be comparing a mutant's affinity to a wild-type crystal.

    The filter is a keyword match on the assay description, which is a blunt
    instrument and will not be perfect. Records whose description is missing
    or generic are kept, because dropping unknowns would hide the problem
    rather than expose it. `wild_type_only=False` returns everything.

    Each record must also carry `assay_description` and `assay_organism` so a
    caller can apply a stricter filter of its own.
    """
    import csv

    d = Path(data_dir) if data_dir is not None else CHEMBL_DHFR_DIR
    path = d / "ecoli_dhfr_ki.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            value = float(r["ki_nM"])
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        desc = r.get("assay_description") or ""
        if wild_type_only and _looks_mutant(desc):
            continue
        out.append({
            "molecule_chembl_id": r["molecule_chembl_id"],
            "assay_chembl_id": r["assay_chembl_id"],
            "document_chembl_id": r["document_chembl_id"],
            "ki_nM": value,
            "relation": r.get("relation", "="),
            "assay_description": desc,
            "assay_organism": r.get("assay_organism") or "",
        })
    return out
