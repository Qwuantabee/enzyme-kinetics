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

1. The ratio is a constant *within* one paper. A single paper runs one assay
   protocol, so one [S]/Km, so one ratio, regardless of how potent the
   compounds are. A ratio that tracked potency (tight compounds corrected
   more than weak ones) would falsify the relation.
2. The ratio may differ *between* papers, and that difference is
   information: it is the difference in their assay [S]/Km.

That pair of predictions is checkable without ever knowing [S] or Km, which
is the point of this module. It is the strongest test the ChEMBL records
actually support, and it is weaker than knowing [S]: it can show the ratios
are consistent with a per-assay substrate factor, but it cannot pin the
factor's value, because any [S]/Km can produce any ratio above 1.

What the records still lack, and always will
--------------------------------------------
ChEMBL's activity records for this target carry neither the assay [S] nor
[E]. So a compound whose ratio is 20 cannot be checked against its own
published Cheng-Prusoff correction. The ratio distribution can be checked;
the absolute conversion cannot. Both statements go in the report.

Honesty rules used here
-----------------------
* Values are pooled per (molecule, document) by geometric mean before taking
  a ratio. IC50 and Ki are log-scale quantities; averaging them on the
  arithmetic scale would make a compound with a 1000 nM and a 1 nM
  measurement average to 500.
* No value is dropped to make a statistic look better. Records that look
  wrong stay in and are flagged (`assay_class`), and the report gives the
  headline statistic with and without them.
* Non-positive and non-finite values are refused rather than clamped.
"""

import math
from collections import defaultdict

import numpy as np

#: Ki assay descriptions that are not enzyme-inhibition measurements at all.
#: One ChEMBL document (CHEMBL1124568) reports "Antibacterial activity
#: against Escherichia coli" as its Ki records -- a whole-cell potency
#: against a bacterium, not a binding constant against the purified enzyme.
#: Pairing those with a purified-enzyme IC50 compares two different
#: quantities, so they are labelled honestly rather than silently mixed in.
NON_ENZYME_MARKERS = ("antibacterial", "antimicrobial", "whole cell", "mic ")


def assay_class(description):
    """Classify an assay description as an enzyme assay or something else.

    Returns "enzyme" or "non_enzyme". Deliberately conservative: only
    descriptions that name a whole-cell/antibacterial endpoint are demoted,
    because a generic or empty description is more likely to be a terse
    enzyme assay than a whole-cell one.
    """
    d = (description or "").lower()
    if any(m in d for m in NON_ENZYME_MARKERS):
        return "non_enzyme"
    return "enzyme"


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


def join_ic50_ki_pairs(ic50_rows, ki_rows, strict_document=True):
    """Pair IC50 and Ki measurements of the same compound.

    `strict_document=True` (the default, and the defensible comparison) keeps
    only pairs where the IC50 and the Ki come from the same publication. That
    holds everything constant but the fact that one assay measured 50%
    inhibition and the other measured a binding constant: same compounds,
    same lab, same protocol family, same paper.

    `strict_document=False` pairs on molecule identity alone, taking each
    compound's pooled IC50 against its pooled Ki. That is a looser comparison
    -- the two values may come from different labs, different enzyme
    preparations and different substrate concentrations -- and it is included
    only to show how much extra scatter that looseness adds.

    Returns a list of dicts, sorted by ratio, with keys:
    molecule_chembl_id, document_chembl_id, ic50_nM, ki_nM, ratio,
    n_ic50_records, n_ki_records, ki_assay_class, ki_assay_description.

    New records must carry molecule_chembl_id, document_chembl_id and the
    value column; anything else is optional.
    """
    ic_pool, ic_n = _pool(ic50_rows, "ic50_nM")
    ki_pool, ki_n = _pool(ki_rows, "ki_nM")
    ki_desc = {}
    for r in ki_rows:
        ki_desc[(r["molecule_chembl_id"], r["document_chembl_id"])] = \
            r.get("assay_description", "")

    if strict_document:
        keys = [k for k in ic_pool if k in ki_pool]
    else:
        # molecule-level: pool every record of a compound, ignoring documents
        by_mol_ic = defaultdict(list)
        by_mol_ki = defaultdict(list)
        for r in ic50_rows:
            by_mol_ic[r["molecule_chembl_id"]].append(float(r["ic50_nM"]))
        for r in ki_rows:
            by_mol_ki[r["molecule_chembl_id"]].append(float(r["ki_nM"]))
        keys = []
        for m in by_mol_ic:
            if m in by_mol_ki:
                ic_pool[(m, None)] = geometric_mean(by_mol_ic[m])
                ki_pool[(m, None)] = geometric_mean(by_mol_ki[m])
                ic_n[(m, None)] = len(by_mol_ic[m])
                ki_n[(m, None)] = len(by_mol_ki[m])
                ki_desc[(m, None)] = ""
                keys.append((m, None))

    pairs = []
    for k in keys:
        ic50 = ic_pool[k]
        ki = ki_pool[k]
        if not (math.isfinite(ic50) and math.isfinite(ki)) or ki <= 0:
            continue
        pairs.append({
            "molecule_chembl_id": k[0],
            "document_chembl_id": k[1],
            "ic50_nM": ic50,
            "ki_nM": ki,
            "ratio": ic50 / ki,
            "n_ic50_records": ic_n[k],
            "n_ki_records": ki_n[k],
            "ki_assay_class": assay_class(ki_desc[k]),
            "ki_assay_description": ki_desc[k],
        })
    pairs.sort(key=lambda p: p["ratio"])
    return pairs


def summarize_ratios(ratios):
    """Distribution summary of IC50/Ki ratios on the log scale.

    Returns dict with n, median, q1, q3, iqr, log10_sd, min, max, and the
    median's 95% bootstrap interval. The log10 standard deviation is the
    honest spread measure for a ratio: on the linear scale one 30-fold
    outlier dominates the standard deviation and makes an otherwise tight
    distribution look wild.

    `median_ci` is a percentile bootstrap (2000 resamples) with a fixed seed,
    so the number is reproducible rather than dependent on the run.
    """
    r = np.asarray([float(x) for x in ratios], dtype=float)
    if r.size == 0:
        raise ValueError("summarize_ratios needs at least one ratio")
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("ratios must be positive and finite")
    logs = np.log10(r)
    med = float(10 ** np.median(logs))
    q1, q3 = (float(10 ** v) for v in np.percentile(logs, [25, 75]))

    rng = np.random.default_rng(0)
    boots = np.empty(2000)
    for i in range(2000):
        sample = rng.choice(logs, size=logs.size, replace=True)
        boots[i] = np.median(sample)
    lo, hi = np.percentile(boots, [2.5, 97.5])

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
        "fold_span": float(r.max() / r.min()),
    }


def implied_substrate_ratio(ratio):
    """Invert Cheng-Prusoff: IC50/Ki = 1 + [S]/Km implies [S]/Km = ratio - 1.

    A ratio below 1 returns a negative value, which is not a substrate
    concentration and cannot be physical under the competitive model. This
    function returns the arithmetic answer and lets the caller decide; use
    `is_physical_ratio` to filter, rather than silently clamping to zero.
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


def document_effect(pairs, min_pairs=3):
    """How much of the log-ratio variance is between documents vs within.

    Cheng-Prusoff's central prediction is that the ratio belongs to the
    *assay*, not to the compound. So the variance decomposes: pairs from the
    same paper should agree, pairs from different papers should not. This
    returns the one-way decomposition of log10(ratio):

        total_var = between_document_var + within_document_var

    and reports `document_explains`, the fraction of total variance
    attributable to which paper the numbers came from. A high value is the
    signature Cheng-Prusoff predicts. Groups with fewer than `min_pairs`
    pairs cannot support a within-group variance estimate and are counted in
    `n_groups_used` only, so the reported fraction is not inflated by
    single-member groups having zero variance by construction.
    """
    groups = defaultdict(list)
    for p in pairs:
        groups[p["document_chembl_id"]].append(math.log10(p["ratio"]))
    used = {d: v for d, v in groups.items() if len(v) >= min_pairs}
    if len(used) < 2:
        return {
            "document_explains": float("nan"),
            "total_var": float("nan"),
            "within_var": float("nan"),
            "between_var": float("nan"),
            "n_groups_used": len(used),
            "n_pairs_used": sum(len(v) for v in used.values()),
        }

    all_vals = [v for vals in used.values() for v in vals]
    total_var = float(np.var(all_vals, ddof=1))
    ss_within = sum((len(v) - 1) * np.var(v, ddof=1) for v in used.values())
    df_within = sum(len(v) - 1 for v in used.values())
    within_var = float(ss_within / df_within)
    # A perfectly constant set has zero total variance, and 1 - 0/0 is not a
    # number. Report nan rather than inventing a fraction of nothing.
    if total_var <= 0.0:
        return {
            "document_explains": float("nan"),
            "total_var": total_var,
            "within_var": within_var,
            "between_var": 0.0,
            "n_groups_used": len(used),
            "n_pairs_used": sum(len(v) for v in used.values()),
        }
    between = float(total_var - within_var)
    # floating-point noise can make this a hair negative, which is not
    # meaningful and would poison a later sqrt
    if between < 0.0:
        between = 0.0
    return {
        "document_explains": float(1.0 - within_var / total_var),
        "total_var": total_var,
        "within_var": within_var,
        "between_var": between,
        "n_groups_used": len(used),
        "n_pairs_used": sum(len(v) for v in used.values()),
    }


def ratio_vs_potency_slope(pairs, min_pairs=5):
    """Slope of log10(ratio) against log10(IC50), fitted within each document.

    This is the falsification test. Cheng-Prusoff makes the ratio an assay
    constant, so within one paper the slope should be zero: correcting a
    25 nM compound and a 76 uM compound from the same assay must use the same
    factor. If the slope is strongly non-zero and consistent in sign, the
    ratio is tracking potency rather than assay conditions, and the
    conversion is not behaving like Cheng-Prusoff.

    Returns a dict keyed by document with n, slope, pearson_r and the IC50
    span, for documents with at least `min_pairs` pairs. A slope's standard
    error is not reported because these groups have 5-14 points and the
    honest summary is the slope plus the correlation, not a false precision.
    """
    groups = defaultdict(list)
    for p in pairs:
        groups[p["document_chembl_id"]].append((p["ic50_nM"], p["ratio"]))
    out = {}
    for doc, vals in groups.items():
        if len(vals) < min_pairs:
            continue
        x = np.log10([v[0] for v in vals])
        y = np.log10([v[1] for v in vals])
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        out[doc] = {
            "n": len(vals),
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": r,
            "ic50_min_nM": float(10 ** x.min()),
            "ic50_max_nM": float(10 ** x.max()),
            "ratio_median": float(10 ** np.median(y)),
        }
    return out


def predicted_ratio(s_conc, km):
    """The IC50/Ki ratio Cheng-Prusoff predicts for a competitive assay.

    This is `correction_factor` restated as a prediction about a measurable
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


def within_document_spread(pairs, min_pairs=3):
    """Pooled within-document log10 SD of the ratio, and the per-doc values.

    This is the consistency measure that matters. Cheng-Prusoff predicts the
    ratio is constant within one assay, so the spread *inside* each paper is
    the reproducibility of the relation, while the difference *between*
    papers is the [S]/Km difference the theory expects. Judging the
    conversion on the pooled scatter of all papers at once would penalise
    exactly the behaviour the theory predicts, which is why this is
    separated out.
    """
    groups = defaultdict(list)
    for p in pairs:
        groups[p["document_chembl_id"]].append(math.log10(p["ratio"]))
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


def verdict(pairs, max_within_log10_sd=0.30, max_median_ratio=50.0,
            max_between_fold=25.0):
    """Classify what the pair data supports, from the numbers only.

    Returns (label, reasons) with label one of:

    * "unsupported" -- the ratios imply a physically impossible or absurd
      substrate factor, or they disagree with each other inside a single
      paper as much as between papers, so no per-assay factor exists either.
    * "partially supported" -- inside each paper the ratio is reproducible
      and the papers differ from each other, which is the structure
      Cheng-Prusoff predicts, but neither [S] nor Km is recorded so the
      absolute conversion cannot be checked. This is the strongest statement
      these files can license.
    * "validated" -- only reachable when the caller supplies assay [S] and Km
      (`has_assay_substrate=True`); consistency alone is not validation.

    The thresholds are set here, not tuned to this dataset: 0.30 in log10 is
    about a 2-fold one-sigma spread, looser than any real assay comparison
    should need, and `max_median_ratio=50` caps the implied [S]/Km at 49.
    """
    s = summarize_ratios([p["ratio"] for p in pairs])
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
            "no document contributes enough pairs to measure a within-assay "
            "spread, so the ratios cannot be shown to be an assay constant")
        return "partially supported", reasons

    if wsd > max_within_log10_sd:
        reasons.append(
            f"within-paper log10 SD {wsd:.2f} exceeds {max_within_log10_sd:.2f} "
            f"(a {(10 ** wsd):.1f}-fold spread inside single assays), so the "
            f"ratio is not a per-assay constant and the conversion has no "
            f"single factor to stand on")
        return "partially supported", reasons

    reasons.append(
        f"within-paper log10 SD {wsd:.2f} ({(10 ** wsd):.1f}-fold): ratios "
        f"are reproducible inside a publication")

    de = doc["document_explains"]
    between_fold = (10 ** math.sqrt(max(doc["between_var"], 0.0))
                    if not math.isnan(de) else float("nan"))
    if not math.isnan(de) and de >= 0.5:
        reasons.append(
            f"{100 * de:.0f}% of the log-ratio variance is between papers "
            f"({between_fold:.1f}-fold from assay conditions), as a per-assay "
            f"[S]/Km requires")
    elif math.isnan(de):
        reasons.append(
            "the ratios show no spread at all, so the between/within split is "
            "undefined -- consistency alone is reported as consistency")
    else:
        reasons.append(
            f"between-paper differences account for only {100 * de:.0f}% of "
            f"the variance, so papers are not cleanly separated by assay "
            f"conditions")

    if not math.isnan(between_fold) and between_fold > max_between_fold:
        reasons.append(
            f"papers differ by {between_fold:.0f}-fold, more than "
            f"{max_between_fold:.0f}-fold, so the implied [S]/Km values are "
            f"implausibly far apart for one enzyme and substrate")
        return "partially supported", reasons

    reasons.append(
        "but no assay [S] or [E] is recorded, so the absolute conversion is "
        "untested compound by compound: consistency is necessary, not "
        "sufficient")
    return "partially supported", reasons
