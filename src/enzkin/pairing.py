"""Join measured IC50 and measured Ki for the same compound, and test
Cheng-Prusoff against the ratios that come out.

The idea being tested
---------------------
For a competitive inhibitor, Cheng-Prusoff says

    IC50 = Ki * (1 + [S]/Km)

so the *ratio* IC50/Ki is not a free parameter: it is 1 + [S]/Km, a property
of the assay's substrate concentration relative to the enzyme's Km for that
substrate, and nothing else. If that is right, then two things must be true
of a real IC50/Ki dataset:

1. The ratio is a constant *within* one assay condition. A single paper runs
   one assay protocol, so one [S]/Km, so one ratio, regardless of how potent
   the compounds are. A ratio that tracked potency (tight compounds corrected
   more than weak ones) would falsify the relation.
2. The ratio may differ *between* papers, and that difference is
   information: it is the difference in their assay [S]/Km.

That pair of predictions is checkable without ever knowing [S] or Km, which
is the point of this module. It is the strongest test the ChEMBL records
actually support, and it is weaker than knowing [S]: it can show the ratios
are consistent with a per-assay substrate factor, but it cannot pin the
factor's value, because any [S]/Km can produce any ratio above 1.

What the records mostly lack
----------------------------
ChEMBL's activity records for this target carry neither the assay [S] nor
[E] as fields. For most assays the description does not state [S] either, so
a compound whose ratio is 20 cannot be checked against its own published
Cheng-Prusoff correction. Exceptions exist and are used: a few IC50 assay
descriptions state the substrate concentration in free text
("in presence of 30 uM ..."), parsed by parse_substrate_uM. One document
(CHEMBL1136127) measured the same compounds at two substrate concentrations,
which is a direct (1 + [S]/Km) test; see two_substrate_check.

Honesty rules used here
-----------------------
* Values are pooled per (molecule, document, IC50 substrate condition) by
  geometric mean before taking a ratio. IC50 and Ki are log-scale quantities;
  averaging them on the arithmetic scale would make a compound with a 1000 nM
  and a 1 nM measurement average to 500.
* IC50 records whose assay description states different [S] are NOT blended:
  a ratio pooled over two substrate concentrations belongs to neither.
* No value is dropped to make a statistic look better. Records that look
  wrong stay in and are flagged (assay_class), and the report gives the
  headline statistic with and without them. A flag is a label read off a
  description, not a verified fact about the measurement.
* Non-positive and non-finite values are refused rather than clamped.
"""

import math
import re
from collections import defaultdict

import numpy as np
from scipy import stats as _sstats

#: Ki assay descriptions that NAME a whole-cell endpoint. This is a keyword
#: match on ChEMBL's free-text description, nothing more. One document
#: (CHEMBL1124568) describes its Ki records as "Antibacterial activity against
#: Escherichia coli", but the values (0.4-1.7 nM; trimethoprim 1.3 nM, the
#: same number that appears as an enzyme Ki in other documents) look like
#: enzyme Ki that ChEMBL mis-described. The label is therefore UNVERIFIED: it
#: is used to run a sensitivity analysis and never to delete data.
NON_ENZYME_MARKERS = ("antibacterial", "antimicrobial", "whole cell", "mic ")

CLASS_ENZYME = "enzyme"
CLASS_ANTIBACTERIAL_DESCRIBED = "antibacterial_described"


def assay_class(description):
    """Flag a Ki assay description as naming a whole-cell endpoint, or not.

    Returns "enzyme" or "antibacterial_described". The second value means
    only that the free-text description contains a whole-cell/antibacterial
    keyword. It is an UNVERIFIED label: in the one document that triggers it
    here the numbers look like enzyme Ki, so the flag marks records for a
    sensitivity analysis and is not evidence that they are not enzyme
    measurements. Generic or empty descriptions return "enzyme" because
    dropping unknowns would hide the problem rather than expose it.
    """
    d = (description or "").lower()
    if any(m in d for m in NON_ENZYME_MARKERS):
        return CLASS_ANTIBACTERIAL_DESCRIBED
    return CLASS_ENZYME


_UNIT_TO_UM = {"nm": 1e-3, "um": 1.0, "\u00b5m": 1.0, "\u03bcm": 1.0, "mm": 1e3}
_SUBSTRATE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(nM|[u\u00b5\u03bc]M|mM)\s+(?:of\s+)?"
    r"(?:dihydrofolate|dihydrofolic|dhf\b)", re.IGNORECASE)


def parse_substrate_uM(description):
    """Assay substrate concentration (uM) stated in an assay description.

    Matches "<number> <nM|uM|mM> dihydrofolate / dihydrofolic acid / DHF"
    (case-insensitive), e.g. "in presence of 30 uM Dihydrofolate reductase"
    gives 30.0. One ChEMBL document writes the substrate as "Dihydrofolate
    reductase" (apparently a typo for dihydrofolate); the "in presence of
    <conc> <substrate>" context is why that is read as [S]. Returns None when
    no concentration is stated, which is the usual case.
    """
    m = _SUBSTRATE_RE.search(description or "")
    if not m:
        return None
    return float(m.group(1)) * _UNIT_TO_UM[m.group(2).lower()]


def geometric_mean(values):
    """Geometric mean of positive values. Raises on any non-positive input.

    Refusing rather than dropping is deliberate: a zero or negative
    concentration is a bad record, and quietly discarding it would change
    the statistic without anyone seeing it.
    """
    vals = list(values)
    if not vals:
        raise ValueError("geometric_mean of no values")
    logs = []
    for v in vals:
        v = float(v)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"geometric_mean needs positive finite values, got {v!r}")
        logs.append(math.log(v))
    return math.exp(sum(logs) / len(logs))


def _pool(rows, value_key):
    """Geometric-mean a record list down to one value per (molecule, document)."""
    buckets = defaultdict(list)
    for r in rows:
        key = (r["molecule_chembl_id"], r["document_chembl_id"])
        buckets[key].append(float(r[value_key]))
    return {k: geometric_mean(v) for k, v in buckets.items()}, \
           {k: len(v) for k, v in buckets.items()}


def join_ic50_ki_pairs(ic50_rows, ki_rows, strict_document=True,
                       split_substrate=True):
    """Pair IC50 and Ki measurements of the same compound.

    strict_document=True (the default, and the defensible comparison) keeps
    only pairs where the IC50 and the Ki come from the same publication. That
    holds everything constant but the fact that one assay measured 50%
    inhibition and the other measured a binding constant: same compounds,
    same lab, same protocol family, same paper.

    split_substrate=True (default) additionally keeps IC50 records apart when
    their assay descriptions state different substrate concentrations
    (parse_substrate_uM). A compound measured at 30 uM and at 100 uM
    substrate then yields two pairs, each against the same pooled Ki, rather
    than one pair whose IC50 is the geometric mean of two different assay
    conditions. split_substrate=False restores the old blending and exists to
    measure how many pairs the split changes.

    strict_document=False pairs on molecule identity alone, taking each
    compound's pooled IC50 against its pooled Ki. That is a looser comparison
    -- the two values may come from different labs, different enzyme
    preparations and different substrate concentrations -- and it is included
    only to show how much extra scatter that looseness adds. Its pairs belong
    to no paper, so group is None and they are excluded from every
    within-paper statistic.

    Returns a list of dicts, sorted by ratio, with keys:
    molecule_chembl_id, document_chembl_id, ic50_nM, ki_nM, ratio,
    n_ic50_records, n_ki_records, ki_assay_class, ki_assay_description,
    ic50_substrate_uM (None unless stated), ic50_assay_ids (sorted list) and
    group (the unit of "one assay condition" used by the within/between-paper
    statistics: the document, or document@S when that document ran IC50 at
    more than one stated [S]).

    New records must carry molecule_chembl_id, document_chembl_id and the
    value column; anything else is optional.
    """
    ki_pool, ki_n = _pool(ki_rows, "ki_nM")
    ki_desc = {}
    for r in ki_rows:
        ki_desc[(r["molecule_chembl_id"], r["document_chembl_id"])] = \
            r.get("assay_description", "")

    pairs = []
    if strict_document:
        buckets = defaultdict(list)
        assays = defaultdict(set)
        for r in ic50_rows:
            s = (parse_substrate_uM(r.get("assay_description"))
                 if split_substrate else None)
            key = (r["molecule_chembl_id"], r["document_chembl_id"], s)
            buckets[key].append(float(r["ic50_nM"]))
            if r.get("assay_chembl_id"):
                assays[key].add(r["assay_chembl_id"])
        for (mol, doc, s), vals in buckets.items():
            kk = (mol, doc)
            if kk not in ki_pool:
                continue
            pairs.append(_make_pair(mol, doc, geometric_mean(vals), ki_pool[kk],
                                    len(vals), ki_n[kk], ki_desc[kk], s,
                                    sorted(assays[(mol, doc, s)])))
        # one assay condition per document unless the document ran IC50 at
        # more than one stated [S]
        doc_s = defaultdict(set)
        for p in pairs:
            if p["ic50_substrate_uM"] is not None:
                doc_s[p["document_chembl_id"]].add(p["ic50_substrate_uM"])
        for p in pairs:
            d = p["document_chembl_id"]
            if len(doc_s[d]) > 1 and p["ic50_substrate_uM"] is not None:
                p["group"] = f"{d}@{p['ic50_substrate_uM']:g}uM"
    else:
        by_mol_ic = defaultdict(list)
        by_mol_ki = defaultdict(list)
        ic_assays = defaultdict(set)
        for r in ic50_rows:
            by_mol_ic[r["molecule_chembl_id"]].append(float(r["ic50_nM"]))
            if r.get("assay_chembl_id"):
                ic_assays[r["molecule_chembl_id"]].add(r["assay_chembl_id"])
        for r in ki_rows:
            by_mol_ki[r["molecule_chembl_id"]].append(float(r["ki_nM"]))
        for m in by_mol_ic:
            if m in by_mol_ki:
                p = _make_pair(m, None, geometric_mean(by_mol_ic[m]),
                               geometric_mean(by_mol_ki[m]),
                               len(by_mol_ic[m]), len(by_mol_ki[m]), "", None,
                               sorted(ic_assays[m]))
                p["group"] = None
                pairs.append(p)
    pairs = [p for p in pairs
             if math.isfinite(p["ic50_nM"]) and math.isfinite(p["ki_nM"])
             and p["ki_nM"] > 0]
    pairs.sort(key=lambda p: p["ratio"])
    return pairs


def _make_pair(mol, doc, ic50, ki, n_ic, n_ki, ki_description, substrate_uM,
               assay_ids):
    return {
        "molecule_chembl_id": mol,
        "document_chembl_id": doc,
        "ic50_nM": ic50,
        "ki_nM": ki,
        "ratio": ic50 / ki if ki > 0 else float("nan"),
        "n_ic50_records": n_ic,
        "n_ki_records": n_ki,
        "ki_assay_class": assay_class(ki_description),
        "ki_assay_description": ki_description,
        "ic50_substrate_uM": substrate_uM,
        "ic50_assay_ids": assay_ids,
        "group": doc,
    }


def _group(p):
    """The assay-condition group of a pair, or None if it belongs to no paper."""
    return p["group"] if "group" in p else p.get("document_chembl_id")


def _cluster_bootstrap_median_ci(logs, groups, n_boot=2000, seed=0):
    """Two-stage (paper, then pair-within-paper) percentile CI for the median.

    Resamples whole papers with replacement and then the pairs inside each
    drawn paper, so that between-paper variation enters the interval. Returns
    (nan, nan) when fewer than two papers exist. With only a handful of
    papers the interval is coarse and under-covers; it is reported as the
    honest alternative to the i.i.d. interval, not as an exact one.
    """
    clusters = defaultdict(list)
    for x, g in zip(logs, groups):
        clusters[g].append(x)
    keys = list(clusters)
    if len(keys) < 2 or any(k is None for k in keys):
        return float("nan"), float("nan")
    arrs = [np.asarray(clusters[k]) for k in keys]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, len(arrs), size=len(arrs))
        sample = np.concatenate(
            [rng.choice(arrs[j], size=arrs[j].size, replace=True) for j in picks])
        boots[i] = np.median(sample)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(10 ** lo), float(10 ** hi)


def summarize_ratios(ratios, groups=None):
    """Distribution summary of IC50/Ki ratios on the log scale.

    Returns dict with n, median, q1, q3, iqr, log10_sd, min, max, and two 95%
    bootstrap intervals for the median:

    * median_ci_low/high -- i.i.d. percentile bootstrap (2000 resamples, fixed
      seed). It treats every pair as an independent draw, which they are not:
      pairs from one paper share an assay protocol. Reported for comparison.
    * median_ci_cluster_low/high -- cluster (paper-level) bootstrap, only when
      "groups" (one label per ratio, e.g. each pair's group) is given and
      holds at least two papers; otherwise nan. This is the interval to
      quote. With very few papers it is coarse.

    The log10 standard deviation is the honest spread measure for a ratio: on
    the linear scale one 30-fold outlier dominates the standard deviation and
    makes an otherwise tight distribution look wild.
    """
    r = np.asarray([float(x) for x in ratios], dtype=float)
    if r.size == 0:
        raise ValueError("summarize_ratios needs at least one ratio")
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("ratios must be positive and finite")
    if groups is not None and len(groups) != r.size:
        raise ValueError("groups must have one label per ratio")
    logs = np.log10(r)
    med = float(10 ** np.median(logs))
    q1, q3 = (float(10 ** v) for v in np.percentile(logs, [25, 75]))

    rng = np.random.default_rng(0)
    boots = np.empty(2000)
    for i in range(2000):
        sample = rng.choice(logs, size=logs.size, replace=True)
        boots[i] = np.median(sample)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if groups is None:
        clo, chi = float("nan"), float("nan")
    else:
        clo, chi = _cluster_bootstrap_median_ci(logs, list(groups))

    return {
        "n": int(r.size),
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "log10_sd": float(np.std(logs, ddof=1)) if r.size > 1 else float("nan"),
        "min": float(r.min()),
        "max": float(r.max()),
        "median_ci_low": float(10 ** lo),
        "median_ci_high": float(10 ** hi),
        "median_ci_cluster_low": clo,
        "median_ci_cluster_high": chi,
        "fold_span": float(r.max() / r.min()),
    }


def implied_substrate_ratio(ratio):
    """Invert Cheng-Prusoff: IC50/Ki = 1 + [S]/Km implies [S]/Km = ratio - 1.

    A ratio below 1 returns a negative value, which is not a substrate
    concentration and cannot be physical under the competitive model. This
    function returns the arithmetic answer and lets the caller decide; use
    is_physical_ratio to filter, rather than silently clamping to zero.
    """
    return float(ratio) - 1.0


def is_physical_ratio(ratio):
    """True if IC50/Ki is at least 1, the minimum the competitive model allows.

    Under Cheng-Prusoff for a competitive inhibitor the ratio cannot be below
    1, because [S]/Km >= 0. A measured ratio below 1 therefore indicates
    either a non-competitive mechanism, or -- far more likely at this quality
    -- mismatched assay conditions or record error. It is not evidence about
    the conversion's validity either way.
    """
    return float(ratio) >= 1.0


#: Variance of log10(ratio) below which a set counts as constant. Noise-free
#: synthetic ratios differ at the 1e-31 level from rounding alone.
_CONSTANT_VAR = 1e-18


def document_effect(pairs, min_pairs=3):
    """How much of the log-ratio variance is between papers vs within.

    Cheng-Prusoff's central prediction is that the ratio belongs to the
    *assay*, not to the compound. So the variance decomposes: pairs from the
    same paper should agree, pairs from different papers should not. This
    returns the one-way decomposition of log10(ratio):

        total_var = between_document_var + within_document_var

    and reports document_explains, 1 - within_var/total_var, an adjusted
    R-squared-like fraction of the variance attributable to which paper the
    numbers came from. Groups with fewer than min_pairs pairs cannot support
    a within-group variance estimate and are excluded, so the fraction is not
    inflated by single-member groups having zero variance by construction.
    Pairs with no group (the loose join) are excluded.

    A fraction without a test says nothing about whether the papers really
    differ, so the one-way ANOVA is returned too: f_stat = MS_between /
    MS_within with df_between = k - 1 and df_within = N - k, and f_p its upper
    tail probability (scipy.stats.f.sf). All four are nan when fewer than two
    papers qualify. If within-paper variance is exactly zero and the papers
    differ, f_stat is inf and f_p is 0.
    """
    groups = defaultdict(list)
    for p in pairs:
        g = _group(p)
        if g is None:
            continue
        groups[g].append(math.log10(p["ratio"]))
    used = {d: v for d, v in groups.items() if len(v) >= min_pairs}
    nan = float("nan")
    base = {"n_groups_used": len(used),
            "n_pairs_used": sum(len(v) for v in used.values()),
            "f_stat": nan, "f_p": nan, "df_between": nan, "df_within": nan}
    if len(used) < 2:
        return dict(base, document_explains=nan, total_var=nan,
                    within_var=nan, between_var=nan)

    all_vals = [v for vals in used.values() for v in vals]
    total_var = float(np.var(all_vals, ddof=1))
    ss_within = float(sum((len(v) - 1) * np.var(v, ddof=1) for v in used.values()))
    df_within = sum(len(v) - 1 for v in used.values())
    within_var = float(ss_within / df_within)
    grand = float(np.mean(all_vals))
    ss_between = float(sum(len(v) * (np.mean(v) - grand) ** 2 for v in used.values()))
    df_between = len(used) - 1
    f_stat = f_p = nan
    if total_var <= _CONSTANT_VAR:
        pass  # every ratio identical to rounding: F is 0/0, report nan
    elif ss_within > 0.0:
        f_stat = float((ss_between / df_between) / (ss_within / df_within))
        f_p = float(_sstats.f.sf(f_stat, df_between, df_within))
    elif ss_between > 0.0:
        f_stat, f_p = float("inf"), 0.0
    base.update(f_stat=f_stat, f_p=f_p, df_between=df_between,
                df_within=df_within)
    # A perfectly constant set has zero total variance, and 1 - 0/0 is not a
    # number. Report nan rather than inventing a fraction of nothing.
    if total_var <= _CONSTANT_VAR:
        return dict(base, document_explains=nan, total_var=total_var,
                    within_var=within_var, between_var=0.0)
    between = float(total_var - within_var)
    # floating-point noise can make this a hair negative, which is not
    # meaningful and would poison a later sqrt
    if between < 0.0:
        between = 0.0
    return dict(base, document_explains=float(1.0 - within_var / total_var),
                total_var=total_var, within_var=within_var, between_var=between)


def ratio_vs_potency_slope(pairs, min_pairs=5):
    """Slopes of log10(ratio) against potency, fitted within each paper.

    This is the falsification test. Cheng-Prusoff makes the ratio an assay
    constant, so within one paper the slope should be zero: correcting a
    25 nM compound and a 76 uM compound from the same assay must use the same
    factor.

    Caveat that the numbers carry: the response is log10(IC50) - log10(Ki), so
    it SHARES a variable with either regressor, and the null slope is not
    exactly zero when the measurements are noisy. Independent noise in IC50
    pushes the slope on log10(IC50) upward; independent noise in Ki pushes the
    slope on log10(Ki) downward. Both slopes are therefore returned (slope
    and slope_vs_ki), each with its standard error, 95% t-interval and p
    value, and the pair of signs is what to read: a real dependence of the
    ratio on potency would move both in the same direction as a per-compound
    correction, whereas noise moves them in opposite directions.

    Returns a dict keyed by group with n, slope, intercept, slope_se,
    slope_ci_low/high, slope_p, pearson_r, slope_vs_ki, slope_vs_ki_se,
    slope_vs_ki_p, the IC50 span and the median ratio, for groups with at
    least min_pairs pairs. Groups with 5-14 points have wide intervals;
    read the interval, not just the point estimate.
    """
    groups = defaultdict(list)
    for p in pairs:
        g = _group(p)
        if g is None:
            continue
        groups[g].append((p["ic50_nM"], p["ki_nM"], p["ratio"]))
    out = {}
    for doc, vals in groups.items():
        if len(vals) < min_pairs:
            continue
        x = np.log10([v[0] for v in vals])
        xk = np.log10([v[1] for v in vals])
        y = np.log10([v[2] for v in vals])
        n = len(vals)
        fit = _sstats.linregress(x, y)
        fit_k = _sstats.linregress(xk, y)
        tcrit = float(_sstats.t.ppf(0.975, n - 2))
        out[doc] = {
            "n": n,
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "slope_se": float(fit.stderr),
            "slope_ci_low": float(fit.slope - tcrit * fit.stderr),
            "slope_ci_high": float(fit.slope + tcrit * fit.stderr),
            "slope_p": float(fit.pvalue),
            "pearson_r": float(fit.rvalue),
            "slope_vs_ki": float(fit_k.slope),
            "slope_vs_ki_se": float(fit_k.stderr),
            "slope_vs_ki_p": float(fit_k.pvalue),
            "ic50_min_nM": float(10 ** x.min()),
            "ic50_max_nM": float(10 ** x.max()),
            "ratio_median": float(10 ** np.median(y)),
        }
    return out


def predicted_ratio(s_conc, km):
    """The IC50/Ki ratio Cheng-Prusoff predicts for a competitive assay.

    This is correction_factor restated as a prediction about a measurable
    ratio rather than a multiplier, because that is the form the ChEMBL data
    can test: [S] and Km are unknown, but their ratio is one number per assay.
    """
    if not (math.isfinite(float(s_conc)) and float(s_conc) >= 0):
        raise ValueError(f"[S] must be finite and non-negative, got {s_conc!r}")
    if not (math.isfinite(float(km)) and float(km) > 0):
        raise ValueError(f"Km must be finite and positive, got {km!r}")
    return 1.0 + float(s_conc) / float(km)


def assay_residuals(pairs, s_conc, km):
    """Observed ratio vs the Cheng-Prusoff prediction, for known [S] and Km.

    This is the check the ChEMBL data cannot support on its own and that a
    paper reporting [S] can: for each pair, log10(observed) - log10(predicted).
    A conversion that works has residuals centred on zero with a spread no
    wider than the assays' own reproducibility.

    Returns dict with n, bias_log10 (mean residual), sd_log10, and
    fold_error_median (median of observed/predicted folded about 1, so over-
    and under-prediction do not cancel).
    """
    pred = predicted_ratio(s_conc, km)
    resid = [math.log10(p["ratio"] / pred) for p in pairs]
    if not resid:
        raise ValueError("assay_residuals needs at least one pair")
    arr = np.asarray(resid)
    return {
        "n": int(arr.size),
        "predicted_ratio": pred,
        "bias_log10": float(arr.mean()),
        "sd_log10": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "fold_error_median": float(10 ** np.median(np.abs(arr))),
    }


def two_substrate_check(pairs):
    """Direct (1 + [S]/Km) test from compounds measured at two stated [S].

    For a competitive inhibitor with the same Ki in both assays, Cheng-Prusoff
    gives IC50(S2) / IC50(S1) = (Km + S2) / (Km + S1). That ratio rho must lie
    between 1 (Km >> S) and S2/S1 (Km << S), and it fixes Km without needing
    the Ki at all:

        Km = (S2 - rho * S1) / (rho - 1)        for 1 < rho < S2/S1

    A second, independent estimate comes from each condition on its own,
    Km = S / (IC50/Ki - 1), which does use the Ki. If Cheng-Prusoff holds and
    the Ki was measured on the same enzyme the two should roughly agree.

    Uses pairs that carry ic50_substrate_uM and that share (molecule,
    document), taking each molecule's lowest and highest stated [S]. Returns
    a dict with n_molecules, rows (one per molecule) and the medians of rho,
    the pair-derived Km and the two single-condition Km values; n_feasible is
    the number of molecules with 1 < rho < S2/S1. Values are measured, not
    fitted: with n around ten the medians carry no interval here.
    """
    by = defaultdict(dict)
    for p in pairs:
        s = p.get("ic50_substrate_uM")
        if s is None or p.get("document_chembl_id") is None:
            continue
        by[(p["molecule_chembl_id"], p["document_chembl_id"])][s] = p
    rows = []
    for (mol, doc), d in sorted(by.items()):
        if len(d) < 2:
            continue
        s_lo, s_hi = min(d), max(d)
        lo, hi = d[s_lo], d[s_hi]
        rho = hi["ic50_nM"] / lo["ic50_nM"]
        feasible = 1.0 < rho < s_hi / s_lo
        km_pair = (s_hi - rho * s_lo) / (rho - 1.0) if feasible else float("nan")
        km_lo = s_lo / (lo["ratio"] - 1.0) if lo["ratio"] > 1.0 else float("nan")
        km_hi = s_hi / (hi["ratio"] - 1.0) if hi["ratio"] > 1.0 else float("nan")
        rows.append({
            "molecule_chembl_id": mol, "document_chembl_id": doc,
            "s_low_uM": s_lo, "s_high_uM": s_hi, "rho": rho,
            "rho_max_possible": s_hi / s_lo, "feasible": feasible,
            "km_from_ic50_pair_uM": km_pair,
            "km_from_low_uM": km_lo, "km_from_high_uM": km_hi,
        })

    def med(key):
        v = [r[key] for r in rows if math.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    return {
        "n_molecules": len(rows),
        "n_feasible": sum(1 for r in rows if r["feasible"]),
        "rho_median": med("rho"),
        "rho_min": min((r["rho"] for r in rows), default=float("nan")),
        "rho_max": max((r["rho"] for r in rows), default=float("nan")),
        "km_pair_median_uM": med("km_from_ic50_pair_uM"),
        "km_from_low_median_uM": med("km_from_low_uM"),
        "km_from_high_median_uM": med("km_from_high_uM"),
        "rows": rows,
    }


def within_document_spread(pairs, min_pairs=3):
    """Pooled within-paper log10 SD of the ratio, and the per-group values.

    This is the consistency measure that matters. Cheng-Prusoff predicts the
    ratio is constant within one assay, so the spread *inside* each paper is
    the reproducibility of the relation, while the difference *between*
    papers is the [S]/Km difference the theory expects. Judging the
    conversion on the pooled scatter of all papers at once would penalise
    exactly the behaviour the theory predicts, which is why this is
    separated out.

    Pairs with no group (the loose join) belong to no paper and are skipped:
    the SD of all of them together is the overall SD, not a within-paper one.
    With no qualifying paper the pooled value is nan.
    """
    groups = defaultdict(list)
    for p in pairs:
        g = _group(p)
        if g is None:
            continue
        groups[g].append(math.log10(p["ratio"]))
    per_doc = {}
    for doc, vals in groups.items():
        if len(vals) < 2:
            continue
        per_doc[doc] = {
            "n": len(vals),
            "log10_sd": float(np.std(vals, ddof=1)),
            "median_ratio": float(10 ** np.median(vals)),
            "fold_spread": float(10 ** (max(vals) - min(vals))),
        }
    usable = {d: v for d, v in per_doc.items() if v["n"] >= min_pairs}
    ss = sum((v["n"] - 1) * v["log10_sd"] ** 2 for v in usable.values())
    df = sum(v["n"] - 1 for v in usable.values())
    return {
        "pooled_log10_sd": float(math.sqrt(ss / df)) if df else float("nan"),
        "per_document": per_doc,
        "n_documents_used": len(usable),
        "n_pairs_used": sum(v["n"] for v in usable.values()),
    }


#: Labels verdict() can return, strongest first.
VERDICT_LABELS = ("partially supported", "not testable", "unsupported")


def verdict(pairs, max_within_log10_sd=0.30, max_median_ratio=50.0,
            max_between_fold=25.0, alpha=0.05):
    """Classify what the pair data supports, from the numbers only.

    Returns (label, reasons). The label is decided by the rules below, in
    order; the thresholds are arguments with fixed defaults, set before this
    dataset was summarised by the rule (0.30 log10 is a 2-fold one-sigma
    spread, and a median ratio above 50 implies [S]/Km above 49), and are not
    tuned to it. Each rule is pinned by a test whose input must produce it.

    1. "unsupported" if the median ratio is <= 1 or > max_median_ratio: the
       implied [S]/Km is impossible or absurd.
    2. "not testable" if no paper (group) contributes at least 3 pairs, so a
       within-paper spread cannot be measured. The loose join lands here: it
       pairs across papers and has no within-paper structure to test.
    3. "unsupported" if the pooled within-paper log10 SD exceeds
       max_within_log10_sd: the ratio is not a per-assay constant. (A
       within-paper scatter that is also as large as the between-paper
       scatter is the same finding and is caught here; the ANOVA F is
       reported in the reasons.)
    4. Otherwise "partially supported": inside each paper the ratio is
       reproducible. Cheng-Prusoff PERMITS the papers to differ (different
       [S]/Km) but does not require it: two papers can share one [S]/Km, and
       a perfectly constant ratio is the best case for the relation. So the
       between-paper ANOVA is reported (F, df, p against alpha, and the
       share of variance) and a non-significant result is stated as "papers
       not distinguishable", without changing the label. Neither [S] nor Km
       is recorded for most assays, so the absolute conversion is untested
       and this is the strongest label these files can license. There is no
       "validated" label: validating the values needs assay [S] and Km (see
       assay_residuals), which this function does not take.

    A papers-differ-by-more-than-max_between_fold finding adds a caution to
    the reasons of a "partially supported" result; it does not change the
    label.
    """
    s = summarize_ratios([p["ratio"] for p in pairs],
                         groups=[_group(p) for p in pairs])
    doc = document_effect(pairs)
    spread = within_document_spread(pairs)
    reasons = []

    if s["median"] <= 1.0 or s["median"] > max_median_ratio:
        reasons.append(
            f"median ratio {s['median']:.2f} implies [S]/Km "
            f"{implied_substrate_ratio(s['median']):.1f}, outside the range a "
            f"competitive assay plausibly ran at")
        return "unsupported", reasons

    wsd = spread["pooled_log10_sd"]
    if math.isnan(wsd):
        reasons.append(
            "no paper contributes enough pairs to measure a within-assay "
            "spread, so the ratios cannot be shown to be an assay constant "
            "or to differ between assays")
        return "not testable", reasons

    if wsd > max_within_log10_sd:
        reasons.append(
            f"within-paper log10 SD {wsd:.2f} exceeds {max_within_log10_sd:.2f} "
            f"(a {(10 ** wsd):.1f}-fold spread inside single assays), so the "
            f"ratio is not a per-assay constant and the conversion has no "
            f"single factor to stand on")
        return "unsupported", reasons

    f_stat = doc["f_stat"]
    testable = not math.isnan(f_stat)
    reasons.append(
        f"within-paper log10 SD {wsd:.2f} ({(10 ** wsd):.1f}-fold, limit "
        f"{max_within_log10_sd:.2f}): ratios are reproducible inside a "
        f"publication")
    if testable:
        sig = ("significant" if doc["f_p"] < alpha else
               "NOT significant (papers not distinguishable)")
        share = doc["document_explains"]
        share_txt = ("n/a" if math.isnan(share)
                     else f"{100 * share:.0f}% of the log-ratio variance")
        reasons.append(
            f"between-paper difference {sig} at alpha {alpha}: ANOVA "
            f"F({int(doc['df_between'])},{int(doc['df_within'])}) = "
            f"{f_stat:.2f}, p = {doc['f_p']:.2g}; papers account for "
            f"{share_txt}")
        between_fold = 10 ** math.sqrt(max(doc["between_var"], 0.0))
        if between_fold > max_between_fold:
            reasons.append(
                f"caution: papers differ by {between_fold:.0f}-fold, more "
                f"than {max_between_fold:.0f}-fold, so the implied [S]/Km "
                f"values are far apart for one enzyme and substrate")
    else:
        reasons.append(
            "fewer than two papers qualify, so the between-paper prediction "
            "is not tested")
    reasons.append(
        "no assay [S] or [E] is recorded for most assays, so the absolute "
        "conversion is untested compound by compound: consistency is "
        "necessary, not sufficient")
    return "partially supported", reasons
