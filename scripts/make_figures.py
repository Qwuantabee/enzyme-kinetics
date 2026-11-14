"""Regenerate every file in figures/ from committed code, data and artefacts.

    python scripts/make_figures.py                   # all figures -> figures/
    python scripts/make_figures.py --list            # name, finding, sources
    python scripts/make_figures.py --only qsar_tail cp_ratio_spread
    python scripts/make_figures.py --out /tmp/figs   # write somewhere else

This is the only script that writes into figures/. Before it existed, seven
PNGs were side effects of three analysis scripts (validate_fits.py wrote five,
fit_published.py one, ic50_to_ki.py one); those scripts no longer draw, so a
figure cannot be silently overwritten by an older style. tests/test_figures.py
fails if a PNG in figures/ is not produced here, or the reverse.

Every figure carries a title that states its finding, axis labels with units,
a legend whenever more than one series is drawn, and a footer naming n and the
source artefact. Matplotlib Agg, 150 dpi, default C0/C3 palette.

Provenance, file by file
------------------------
Read from a committed results artefact (nothing is recomputed; a changed
artefact changes the figure):

  cp_two_substrate.png     results/cheng_prusoff_two_substrate.csv
                           (cross-checked against results_cheng_prusoff.txt)
  cp_ratio_spread.png      results/cheng_prusoff_pairs.csv, the within-group
                           SD from results/cheng_prusoff_ratio_stats.csv
  qsar_tail.png            results/qsar_tail_sweep.csv
  qsar_mw_scaffold.png     results_qsar.txt, results_single_scaffold.txt
  km_uncertainty.png       results_published.txt

Recomputed from committed data with committed code (seeded, deterministic,
about a second each):

  published_p99.png        data/frere2023_nitrocefin/*.csv (18 traces)
  ic50_conversion.png      data/chembl_dhfr/ecoli_dhfr_ic50.csv (482 records)

Recomputed from a seeded simulation. No data file exists for these because the
data ARE the simulation (src/enzkin/simulate.py, true Km = 25, Vmax = 100, 5%
noise, 8 points); the seeds are the ones scripts/validate_fits.py always used,
and tests/test_figures.py pins the regenerated numbers to the README tables:

  km_recovery.png          500 datasets per error model, seeds 0-499
  mm_fit.png               seed 42
  linearizations.png       seed 42
  inhibition_fit.png       mixed mechanism, Ki = 10, Kis = 40, seed 3

km_recovery.png replaces km_recovery_relative.png and km_recovery_absolute.png:
the finding is the contrast between the two error models, so both are drawn on
one shared axis. The old box plots also hid the catastrophic tail of the
Lineweaver-Burk fits (showfliers=False); this one plots every fit.

Why the Km/Vmax/ΔAICc worked-example numbers are not in a results file: the
simulation study prints them (python scripts/validate_fits.py) and its output
file results.txt is git-ignored. They are reproduced here instead, and the test
suite pins them.
"""

import argparse
import csv
import math
import re
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enzkin import (cheng_prusoff as cp, datasets, fitting, linearize,  # noqa: E402
                    models, progress, simulate)

FIG_DIR = ROOT / "figures"
RES = ROOT / "results"
DPI = 150
GREY = "0.45"
FS_SUPTITLE = 11      # figure title
FS_TITLE = 10         # axes title
FS_LABEL = 9.5        # axis labels
FS_TICK = 8.5         # tick labels
FS_NOTE = 8.5         # legends and in-plot annotations (never below 8)
FS_FOOT = 7           # footer naming n and the source artefact
STYLE = {"axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
         "legend.fontsize": FS_NOTE}
MINUS = "\u2212"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _grab(text, pattern, what, flags=re.S):
    """Regex search that fails loudly: a figure must never fall back to a
    default when the artefact it reads has changed format."""
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"could not find {what!r} in the artefact "
                         f"(pattern {pattern!r}); the file format changed")
    return m


def _finish(fig, footer, legend=None, ncol=2):
    """Lay the figure out with room for a wrapped footer (n and source) and,
    optionally, one figure-level legend below the panels, so a legend never
    sits on top of data."""
    width = max(60, int(fig.get_figwidth() * 17))
    text = "\n".join(textwrap.fill(par, width) for par in footer.split("\n"))
    lines = text.count("\n") + 1
    foot_in = 0.08 + 0.125 * lines
    leg_in = 0.04 + 0.2 * (-(-len(legend) // ncol)) if legend else 0.0
    top = 0.935 if fig.get_suptitle() else 1.0
    fig.tight_layout(rect=(0, (foot_in + leg_in) / fig.get_figheight(), 1, top))
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=ncol, frameon=False,
                   fontsize=FS_NOTE, bbox_to_anchor=(0.5, foot_in / fig.get_figheight()),
                   borderaxespad=0, columnspacing=1.8)
    fig.text(0.008, 0.05 / fig.get_figheight(), text, fontsize=FS_FOOT, color="0.35",
             ha="left", va="bottom")


def _fmt(x, nd=2):
    """Number with a true minus sign, as the README writes them."""
    return f"{x:.{nd}f}".replace("-", MINUS)


# ---------------------------------------------------------------------------
# artefact readers (kept separate from drawing so tests can check the numbers)
# ---------------------------------------------------------------------------

def read_two_substrate():
    """results/cheng_prusoff_two_substrate.csv -> dict of plain Python values."""
    rows = _csv(RES / "cheng_prusoff_two_substrate.csv")
    s_low = {float(r["s_low_uM"]) for r in rows}
    s_high = {float(r["s_high_uM"]) for r in rows}
    upper = {round(float(r["rho_max_possible"]), 4) for r in rows}
    if len(s_low) != 1 or len(s_high) != 1 or len(upper) != 1:
        raise ValueError("two-substrate file mixes substrate pairs")
    s_low, s_high = s_low.pop(), s_high.pop()
    km_low = [float(r["km_from_low_uM"]) for r in rows]
    km_high = [float(r["km_from_high_uM"]) for r in rows]
    km_low_med = statistics.median(k for k in km_low if math.isfinite(k))
    km_high_med = statistics.median(k for k in km_high if math.isfinite(k))

    def predicted(km):
        return (1 + s_high / km) / (1 + s_low / km)

    return {
        "ids": [r["molecule_chembl_id"] for r in rows],
        "document": rows[0]["document_chembl_id"],
        "ratio": [float(r["ic50_ratio_high_over_low"]) for r in rows],
        "inside": [r["inside_bounds"] == "True" for r in rows],
        "s_low": s_low, "s_high": s_high,
        "lower": 1.0, "upper": s_high / s_low,
        "median": statistics.median(float(r["ic50_ratio_high_over_low"])
                                    for r in rows),
        "km_low_median": km_low_med, "km_high_median": km_high_med,
        "pred_low": predicted(km_low_med), "pred_high": predicted(km_high_med),
    }


def read_pairs():
    """Strict (same-publication) IC50/Ki pairs grouped by assay condition,
    with the pooled within-group SD the artefact reports."""
    rows = [r for r in _csv(RES / "cheng_prusoff_pairs.csv")
            if r["join_mode"] == "same_document"]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(float(r["ic50_over_ki"]))
    stats = {r["subset"]: r for r in
             _csv(RES / "cheng_prusoff_ratio_stats.csv")}["same_document_all"]
    def order(item):
        doc, _, cond = item[0].partition("@")
        return (-len(item[1]), doc, float(cond.rstrip("uM")) if cond else 0.0)

    ordered = sorted(groups.items(), key=order)
    return {
        "groups": ordered,
        "pooled_sd": float(stats["within_group_log10_sd"]),
        "pooled_median": float(stats["median_ratio"]),
        "n": int(stats["n"]),
    }


def group_stats(values):
    logs = [math.log10(v) for v in values]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "spread": max(values) / min(values),
        "sd": statistics.stdev(logs) if len(values) > 1 else float("nan"),
    }


# cutoff (nM) -> x position, weakest-binder cut first; None = no cutoff
CUTOFFS = (None, 100000.0, 10000.0, 1000.0, 100.0)
CUTOFF_LABELS = ("all", "<100 \u00b5M", "<10 \u00b5M", "<1 \u00b5M",
                 "<100 nM\n(drug-relevant)")
SERIES = (("CHEMBL1129627", "1996 series (CHEMBL1129627)", "C3"),
          ("CHEMBL1124156", "1988 series (CHEMBL1124156)", "C0"))


def read_tail_sweep():
    """results/qsar_tail_sweep.csv -> {(document, subset): [row per cutoff]}"""
    out = {}
    for r in _csv(RES / "qsar_tail_sweep.csv"):
        cut = float(r["cutoff_nM"]) if r["cutoff_nM"] else None
        out.setdefault((r["document"], r["subset"]), {})[cut] = {
            "n": int(r["n_kept"]),
            "r2": float(r["loocv_r2"]),
            "single": float(r["best_single_r2"]),
            "p": float(r["p_perm"]) if r["p_perm"] else None,
        }
    for key, by_cut in out.items():
        missing = [c for c in CUTOFFS if c not in by_cut]
        if missing:
            raise ValueError(f"{key}: sweep file lacks cutoffs {missing}")
    return out


def read_mw_gain():
    """Ten-descriptor LOO R2 against molecular-weight-only LOO R2, whole series
    (results_qsar.txt) and largest scaffold (results_single_scaffold.txt)."""
    qsar = _text("results_qsar.txt")
    scaf = _text("results_single_scaffold.txt")
    out = []
    for doc, label in (("CHEMBL1129627", "1996"), ("CHEMBL1124156", "1988")):
        block = _grab(qsar, rf"--- {doc}:.*?(?=\n--- CHEMBL|\Z)", f"{doc} block").group(0)
        n_all = int(_grab(block, r"fitted:\s+n = (\d+)", "n").group(1))
        pars = _grab(block, r"best single descriptor (\w+) gives LOO R2 = "
                            r"(-?[\d.]+); the ten-descriptor model gives "
                            r"(-?[\d.]+) \(gain ([+-][\d.]+)\)", "parsimony line")
        out.append({"series": label, "subset": "whole series", "n": n_all,
                    "descriptor": pars.group(1), "single": float(pars.group(2)),
                    "ten": float(pars.group(3)), "gain": float(pars.group(4))})
        sblock = _grab(scaf, rf"--- {doc}:.*?(?=\n--- CHEMBL|\Z)", f"{doc} scaffold block").group(0)
        n_sc = int(_grab(sblock, r"SUBSET n = (\d+)", "scaffold n").group(1))
        sp = _grab(sblock, r"LOO R2 = (-?[\d.]+); best single descriptor (\w+) "
                           r"gives (-?[\d.]+) \(gain ([+-][\d.]+)\)", "scaffold line")
        out.append({"series": label, "subset": "largest scaffold", "n": n_sc,
                    "descriptor": sp.group(2), "single": float(sp.group(3)),
                    "ten": float(sp.group(1)), "gain": float(sp.group(4))})
    return out


def read_published():
    """results_published.txt -> the Km numbers behind 'the i.i.d. SE is not an
    uncertainty'."""
    t = _text("results_published.txt")
    ir = _grab(t, r"model: michaelis_menten\s+n=(\d+).*?km = ([\d.]+) \+/- ([\d.]+)",
               "initial-rate Km")
    iid = _grab(t, r"progress-curve fit: (\d+) trace\(s\), (\d+) points.*?"
                   r"km\s+= ([\d.]+) \+/- ([\d.]+)", "i.i.d. progress Km")
    cl = _grab(t, r"(\d+)-trace cluster bootstrap: km = ([\d.]+) \+/- ([\d.]+) uM,"
                  r"\s*95% \[([\d.]+), ([\d.]+)\]", "cluster bootstrap line")
    ratio = _grab(t, r"cluster SE / i\.i\.d\. SE = (\d+)x on Km", "SE ratio")
    pub = _grab(t, r"published\s+kcat = ([\d.]+) s\^-1, Km = ([\d.]+) uM",
                "published constants")
    return {
        "n_rates": int(ir.group(1)),
        "rate_km": float(ir.group(2)), "rate_se": float(ir.group(3)),
        "n_traces": int(iid.group(1)), "n_points": int(iid.group(2)),
        "iid_km": float(iid.group(3)), "iid_se": float(iid.group(4)),
        "n_boot": int(cl.group(1)), "cl_km": float(cl.group(2)),
        "cl_se": float(cl.group(3)),
        "cl_lo": float(cl.group(4)), "cl_hi": float(cl.group(5)),
        "se_ratio": int(ratio.group(1)),
        "published_km": float(pub.group(2)),
    }


# ---------------------------------------------------------------------------
# new figures: the review-pass findings, each read from a committed artefact
# ---------------------------------------------------------------------------

def fig_cp_two_substrate():
    d = read_two_substrate()
    order = np.argsort(d["ratio"])
    y = np.arange(len(order))
    ratios = np.array(d["ratio"])[order]
    ids = [d["ids"][i] for i in order]
    lo, hi = d["lower"], d["upper"]
    xmin, xmax = 0.6, 4.0

    fig, ax = plt.subplots(figsize=(6.8, 5.7))
    ax.axvspan(xmin, lo, color="C3", alpha=0.10, lw=0)
    ax.axvspan(hi, xmax, color="C3", alpha=0.10, lw=0)
    ax.axvspan(lo, hi, color="C0", alpha=0.13, lw=0)
    ax.axvline(d["median"], ls="--", lw=1.3, color=GREY)
    ax.axvline(d["pred_low"], ls=":", lw=1.3, color="C3")
    ax.axvline(d["pred_high"], ls=":", lw=1.3, color="C3")
    ax.scatter(ratios, y, s=50, color="C0", edgecolor="black", linewidth=0.5, zorder=3)
    ax.text(d["median"], len(y) - 0.35, f"median {d['median']:.2f}", ha="center",
            va="bottom", fontsize=FS_NOTE, color="0.25")
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=FS_NOTE - 0.5)
    ax.set_ylim(-0.8, len(y) + 0.4)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f"IC50 at {d['s_high']:g} \u00b5M substrate  /  "
                  f"IC50 at {d['s_low']:g} \u00b5M substrate  (dimensionless)")
    ax.set_ylabel("compound (all from " + d["document"] + ")")
    ax.set_title(f"All {len(y)} compounds sit inside the Cheng-Prusoff window\n"
                 f"({lo:.1f}\u2013{hi:.2f}; measured median {d['median']:.2f}, "
                 f"no intervals at n = {len(y)})")
    handles = [
        Patch(facecolor="C0", alpha=0.25,
              label=f"competitive window {lo:.1f}\u2013{hi:.2f}  (= {d['s_high']:g}/{d['s_low']:g})"),
        Patch(facecolor="C3", alpha=0.20, label="outside the window: not competitive"),
        Line2D([], [], marker="o", ls="", color="C0", markeredgecolor="black",
               markeredgewidth=0.5, label=f"one compound (n = {len(y)})"),
        Line2D([], [], ls="--", color=GREY, label=f"measured median {d['median']:.2f}"),
        Line2D([], [], ls=":", color="C3",
               label=f"predicted from single-[S] IC50/Ki: {d['pred_low']:.2f}, {d['pred_high']:.2f}"),
    ]
    _finish(fig, f"n = {len(y)} compounds, all in {d['document']}. "
                 "Source: results/cheng_prusoff_two_substrate.csv.", legend=handles, ncol=1)
    return fig


def fig_cp_ratio_spread():
    d = read_pairs()
    groups = d["groups"]
    pooled_fold = 10 ** d["pooled_sd"]
    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    ax.axhline(1.0, color=GREY, lw=0.9, ls="-")
    ax.axhline(d["pooled_median"], color=GREY, lw=1.2, ls="--")
    ax.text(len(groups) - 0.55, d["pooled_median"] * 1.06,
            f"pooled median {d['pooled_median']:.2f}", ha="right", va="bottom",
            fontsize=FS_NOTE, color="0.3")
    ax.text(len(groups) - 0.55, 0.93, "ratio 1: the competitive floor", ha="right",
            va="top", fontsize=FS_NOTE, color="0.3")

    labels = []
    for i, (name, vals) in enumerate(groups):
        s = group_stats(vals)
        enough = len(vals) >= 3
        med = s["median"]
        if enough:
            ax.add_patch(plt.Rectangle((i - 0.36, med / pooled_fold), 0.72,
                                       med * pooled_fold - med / pooled_fold,
                                       facecolor="C3", alpha=0.14, lw=0, zorder=1))
        xs = i + np.linspace(-0.2, 0.2, len(vals)) if len(vals) > 1 else [i]
        ax.scatter(xs, np.sort(vals), s=34, color="C0" if enough else "0.65",
                   edgecolor="black", linewidth=0.45, zorder=3)
        ax.hlines(med, i - 0.36, i + 0.36, color="black", lw=1.7, zorder=4)
        doc, _, cond = name.partition("@")
        head = f"{doc}\n" + (f"@ {cond[:-2]} \u00b5M, " if cond else "") + f"n = {len(vals)}"
        if enough:
            labels.append(head + f"\nmax/min {s['spread']:.2f}\u00d7\nSD {s['sd']:.2f}")
        else:
            labels.append(head + "\nnot in the\npooled SD")

    ax.set_yscale("log")
    ax.set_ylim(0.3, 45)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=FS_NOTE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("IC50 / Ki per compound\n(dimensionless, log scale)")
    ax.set_xlabel("assay condition (publication, or publication @ stated [S]); "
                  "SD = log10 SD within the group")
    ax.set_title("The pooled IC50/Ki median is not any one assay's conversion factor:\n"
                 "group medians run 3.0\u201320.9, within-group spreads 2.0\u00d7\u201326.9\u00d7",
                 pad=40)
    handles = [
        Line2D([], [], marker="o", ls="", color="C0", markeredgecolor="black",
               markeredgewidth=0.45, label="one IC50/Ki pair"),
        Line2D([], [], color="black", lw=1.7, label="group median"),
        Patch(facecolor="C3", alpha=0.25,
              label=f"\u00b11 pooled within-group SD: {d['pooled_sd']:.3f} log10 "
                    f"(\u00d7/\u00f7{pooled_fold:.2f})"),
        Line2D([], [], ls="--", color=GREY, label=f"pooled median, all {d['n']} pairs"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=FS_NOTE, frameon=False)
    _finish(fig, f"n = {d['n']} same-publication pairs. Sources: results/cheng_prusoff_pairs.csv, "
                 "results/cheng_prusoff_ratio_stats.csv; per-group table in "
                 "results_cheng_prusoff.txt.")
    return fig


def fig_qsar_tail():
    sweep = read_tail_sweep()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.7), sharey=True)
    x = np.arange(len(CUTOFFS))
    for ax, (subset, ptitle) in zip(axes, (("whole series", "whole series"),
                                           ("largest scaffold", "largest Murcko scaffold only"))):
        ax.axhline(0, color=GREY, lw=1.0, ls=":")
        ax.axvspan(len(CUTOFFS) - 1.5, len(CUTOFFS) - 0.5, color="0.5", alpha=0.12, lw=0)
        for doc, label, colour in SERIES:
            rows = sweep[(doc, subset)]
            ax.plot(x, [rows[c]["r2"] for c in CUTOFFS], "-o", color=colour, lw=1.9, ms=6.5,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
            ax.plot(x, [rows[c]["single"] for c in CUTOFFS], "--s", color=colour, lw=1.2,
                    ms=4.5, alpha=0.75, markeredgecolor="black", markeredgewidth=0.4,
                    zorder=2)
            ax.annotate(_fmt(rows[100.0]["r2"], 3), (x[-1], rows[100.0]["r2"]),
                        xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=FS_NOTE + 0.5, fontweight="bold", color=colour)
        n96 = [sweep[(SERIES[0][0], subset)][c]["n"] for c in CUTOFFS]
        n88 = [sweep[(SERIES[1][0], subset)][c]["n"] for c in CUTOFFS]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nn = {a} | {b}" for lab, a, b in
                            zip(CUTOFF_LABELS, n96, n88)], fontsize=FS_NOTE)
        ax.set_xlim(-0.4, len(CUTOFFS) - 0.3)
        ax.set_ylim(-0.3, 0.97)
        ax.set_xlabel("keep only compounds with Ki below   (n kept: 1996 | 1988)")
        ax.set_title(ptitle)
    axes[0].set_ylabel("leave-one-out R\u00b2  (1 = perfect, 0 = predicts the mean)")
    handles = []
    for doc, label, colour in SERIES:
        handles.append(Line2D([], [], color=colour, lw=1.9, marker="o",
                              markeredgecolor="black", markeredgewidth=0.5,
                              label=f"{label}: ten descriptors"))
        handles.append(Line2D([], [], color=colour, lw=1.2, ls="--", marker="s", ms=4.5,
                              alpha=0.75, markeredgecolor="black", markeredgewidth=0.4,
                              label=f"{label}: best single descriptor"))
    r96 = sweep[(SERIES[0][0], "whole series")][100.0]["r2"]
    r88 = sweep[(SERIES[1][0], "whole series")][100.0]["r2"]
    fig.suptitle("At the drug-relevant 100 nM cutoff the 1996 QSAR fit collapses "
                 f"(LOO R\u00b2 {_fmt(r96, 3)}); the 1988 fit survives ({r88:.3f})",
                 fontsize=FS_SUPTITLE)
    _finish(fig, "Compounds with Ki above the cutoff are dropped and the whole pipeline is "
                 "refitted, permutation null included. Source: results/qsar_tail_sweep.csv "
                 "(results_qsar_tail.txt).", legend=handles)
    return fig


def fig_qsar_mw_scaffold():
    rows = read_mw_gain()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    w = 0.36
    xs = np.arange(len(rows))
    ax.bar(xs - w / 2, [r["ten"] for r in rows], w, color="C0", edgecolor="black",
           linewidth=0.5, label="all ten descriptors")
    ax.bar(xs + w / 2, [r["single"] for r in rows], w, color="C3", edgecolor="black",
           linewidth=0.5, label="molecular weight alone")
    for x_, r in zip(xs, rows):
        ax.text(x_ - w / 2, r["ten"] + 0.012, f"{r['ten']:.3f}", ha="center", va="bottom",
                fontsize=FS_NOTE)
        ax.text(x_ + w / 2, r["single"] + 0.012, f"{r['single']:.3f}", ha="center",
                va="bottom", fontsize=FS_NOTE)
        star = r["series"] == "1988" and r["subset"] == "largest scaffold"
        ax.text(x_, max(r["ten"], r["single"]) + 0.08,
                f"gain {r['gain']:+.3f}".replace("-", MINUS), ha="center", va="bottom",
                fontsize=FS_NOTE + (1 if star else 0), fontweight="bold" if star else "normal",
                color="C3" if star else "0.25")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['series']} series\n{r['subset']}\nn = {r['n']}" for r in rows],
                       fontsize=FS_NOTE)
    ax.set_ylabel("leave-one-out R\u00b2")
    ax.set_ylim(0, 0.9)
    star_gain = [r for r in rows if r["series"] == "1988" and r["subset"] == "largest scaffold"][0]
    ax.set_title("Molecular weight carries the 1988 within-scaffold fit:\n"
                 f"the other descriptors add R\u00b2 {star_gain['gain']:+.3f}")
    ax.legend(loc="upper left", fontsize=FS_NOTE, frameon=False)
    _finish(fig, "Gain = ten-descriptor LOO R\u00b2 minus molecular-weight-only LOO R\u00b2. "
                 "Sources: results_qsar.txt (whole series), results_single_scaffold.txt "
                 "(largest scaffold).")
    return fig


def fig_km_uncertainty():
    d = read_published()
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    rows = [
        (f"initial rates\n({d['n_rates']} averaged rates)", d["rate_km"], d["rate_se"], None, "C0"),
        (f"progress curves, i.i.d. SE\n({d['n_points']} points as independent)", d["iid_km"],
         d["iid_se"], None, "C3"),
        (f"progress curves,\ntrace-level bootstrap ({d['n_boot']})", d["cl_km"], d["cl_se"],
         (d["cl_lo"], d["cl_hi"]), "C0"),
    ]
    ypos = [2, 1, 0]
    ax.axvline(d["published_km"], color=GREY, ls="--", lw=1.3)
    ax.text(d["published_km"] + 0.3, 2.62, f"published Km = {d['published_km']:g} \u00b5M",
            ha="left", va="top", fontsize=FS_NOTE, color="0.25")
    for (label, km, se, ci, colour), y in zip(rows, ypos):
        if ci is not None:
            ax.hlines(y, ci[0], ci[1], color=colour, lw=1.3, zorder=2)
            ax.vlines([ci[0], ci[1]], y - 0.09, y + 0.09, color=colour, lw=1.3, zorder=2)
        ax.errorbar(km, y, xerr=se, fmt="o", ms=7.5, color=colour, mec="black", mew=0.5,
                    elinewidth=3.6, capsize=0, zorder=3)
        text = f"{km:.2f} \u00b1 {se:.3g}"
        if ci is not None:
            text += f"\n95% CI {ci[0]:.2f}\u2013{ci[1]:.2f}"
        ax.text(max(km + se, ci[1] if ci else 0) + 0.6, y, text, va="center",
                fontsize=FS_NOTE, color="0.2")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_NOTE)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlim(22, 46)
    ax.set_xlabel("Km of P99 \u03b2-lactamase with nitrocefin (\u00b5M)")
    where = ("below" if d["published_km"] < d["cl_lo"] else
             "above" if d["published_km"] > d["cl_hi"] else "inside")
    ax.set_title(f"The i.i.d. Km SE ({d['iid_se']:.3g} \u00b5M) is {d['se_ratio']}\u00d7 too small: "
                 f"the honest SE is {d['cl_se']:.3g} \u00b5M,\nstill tighter than initial rates "
                 f"({d['rate_se']:.3g} \u00b5M); the published Km is {where} the 95% CI")
    handles = [
        Line2D([], [], marker="o", ls="-", lw=3.6, color="0.4", mec="black", mew=0.5,
               label="fitted Km \u00b1 1 SE"),
        Line2D([], [], color="0.4", lw=1.3, marker="|", ms=8, label="95% CI (trace bootstrap)"),
        Line2D([], [], ls="--", color=GREY, label="published Km"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=FS_NOTE, frameon=False)
    _finish(fig, f"n = {d['n_traces']} traces, 5 substrate concentrations (Fr\u00e8re et al. "
                 "2023). Source: results_published.txt.")
    return fig


# ---------------------------------------------------------------------------
# regenerated figures: simulation or committed raw data, same seeds as before
# ---------------------------------------------------------------------------

N_TRIALS = 500
TRUE_VMAX, TRUE_KM = 100.0, 25.0
NOISE = 0.05
METHODS = (("nonlinear", "nonlinear fit"), ("lineweaver_burk", "Lineweaver-\nBurk"),
           ("hanes_woolf", "Hanes-\nWoolf"), ("eadie_hofstee", "Eadie-\nHofstee"))


def km_recovery_errors(noise_type):
    """Km error (% of the true Km) of each method over seeds 0..499. The seeds,
    sizes and noise level are those scripts/validate_fits.py always used, so the
    SDs reproduce the README tables."""
    errs = {k: [] for k, _ in METHODS}
    for seed in range(N_TRIALS):
        S, v, truth = simulate.substrate_series(
            vmax=TRUE_VMAX, km=TRUE_KM, noise=NOISE, noise_type=noise_type, seed=seed)
        est = {
            "nonlinear": fitting.fit_michaelis_menten(S, v).params,
            "lineweaver_burk": linearize.params_from_lineweaver_burk(S, v),
            "hanes_woolf": linearize.params_from_hanes_woolf(S, v),
            "eadie_hofstee": linearize.params_from_eadie_hofstee(S, v),
        }
        for k, p in est.items():
            errs[k].append(100 * (p["km"] / truth["km"] - 1))
    return {k: np.array(v) for k, v in errs.items()}


def _pct_label(v, _pos=None):
    if v == 0:
        return "0"
    mag = abs(v)
    if mag < 1000:
        s = f"{mag:g}"
    else:
        exp = int(round(math.log10(mag)))
        s = f"10$^{{{exp}}}$"
    return (MINUS if v < 0 else "") + s


def fig_km_recovery():
    data = {nt: km_recovery_errors(nt) for nt in ("relative", "absolute")}
    titles = {"relative": "proportional error (noise scales with v)",
              "absolute": "constant absolute error (noise independent of v)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.9), sharey=True)
    rng = np.random.default_rng(0)           # horizontal jitter only (cosmetic)
    for ax, nt in zip(axes, ("relative", "absolute")):
        ax.set_yscale("symlog", linthresh=50, linscale=1.0)
        ax.axhline(0, color=GREY, lw=1.0, ls="--", zorder=1)
        labels = []
        for i, (key, label) in enumerate(METHODS):
            e = data[nt][key]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(e)), e, s=5, color="C0",
                       alpha=0.28, linewidths=0, zorder=2)
            ax.boxplot(e, positions=[i], widths=0.46, showfliers=False, whis=(5, 95),
                       patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor="none", edgecolor="black", linewidth=0.9),
                       whiskerprops=dict(color="black", linewidth=0.9),
                       capprops=dict(color="black", linewidth=0.9),
                       medianprops=dict(color="C3", linewidth=2.0))
            sd = e.std()
            sd_txt = f"{sd:.0f}%" if sd >= 1000 else f"{sd:.1f}%"
            bias_txt = f"{e.mean():+.1f}%".replace("-", MINUS)
            labels.append(f"{label}\nSD {sd_txt}\nbias {bias_txt}")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(labels, fontsize=FS_NOTE)
        ax.set_xlim(-0.55, len(METHODS) - 0.45)
        ax.set_title(titles[nt])
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_label))
        ax.set_yticks([-1e5, -1e4, -1e3, -100, 0, 100, 1e3, 1e4])
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("error in fitted Km (% of the true Km)\n"
                       "linear within \u00b150%, logarithmic beyond")
    sd_lb_abs = data["absolute"]["lineweaver_burk"].std()
    sd_lb_rel = data["relative"]["lineweaver_burk"].std()
    fig.suptitle("Lineweaver-Burk fails only under constant absolute error "
                 f"(Km SD {sd_lb_abs:.0f}% vs {sd_lb_rel:.1f}% under proportional error);\n"
                 "the nonlinear fit is never the worst", fontsize=FS_SUPTITLE)
    handles = [
        Line2D([], [], marker="o", ls="", color="C0", alpha=0.6, ms=4,
               label=f"one simulated experiment\n(n = {N_TRIALS} per method)"),
        Line2D([], [], color="C3", lw=2.0, label="median"),
        Line2D([], [], color="black", lw=0.9, marker="s", mfc="none", ms=8,
               label="box: middle 50%;\nwhiskers: 5th\u201395th percentile"),
        Line2D([], [], ls="--", color=GREY, label="no error"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=FS_NOTE - 0.5, frameon=True,
                   framealpha=0.92, edgecolor="0.8")
    _finish(fig, f"Simulation: {N_TRIALS} datasets per error model (seeds 0\u2013499), 8 substrate "
                 f"concentrations, {NOISE:.0%} noise, true Km = {TRUE_KM:g}, Vmax = {TRUE_VMAX:g}. "
                 "Every fit is plotted; none is hidden. SD and bias are over all 500 fits.")
    return fig


def _example_fit():
    S, v, truth = simulate.substrate_series(vmax=TRUE_VMAX, km=TRUE_KM, noise=NOISE, n=8,
                                            seed=42)
    return S, v, truth, fitting.fit_michaelis_menten(S, v)


def fig_mm_fit():
    S, v, truth, fit = _example_fit()
    km, vmax = fit.params["km"], fit.params["vmax"]
    smooth = np.geomspace(min(S) / 2, max(S) * 1.5, 300)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(smooth, models.michaelis_menten(smooth, truth["vmax"], truth["km"]), ls="--",
            lw=1.3, color=GREY, label=f"true curve (Km = {truth['km']:g}, Vmax = {truth['vmax']:g})")
    ax.plot(smooth, models.michaelis_menten(smooth, **fit.params), lw=1.8, color="C3",
            label=f"nonlinear fit (Km = {km:.1f}, Vmax = {vmax:.1f})")
    ax.scatter(S, v, s=32, zorder=3, edgecolor="black", linewidth=0.5,
               label=f"simulated data (n = {len(S)}, {NOISE:.0%} noise)")
    ax.set_xlabel("substrate [S] (arbitrary units)")
    ax.set_ylabel("initial rate v (arbitrary units)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"One {len(S)}-point titration recovers Km = {km:.1f} and "
                 f"Vmax = {vmax:.1f}\n(true values {truth['km']:g} and {truth['vmax']:g})")
    ax.legend(loc="lower right", fontsize=FS_NOTE, frameon=False)
    _finish(fig, f"Simulation seed 42. Fit SE: Km \u00b1 {fit.stderr['km']:.2f}, "
                 f"Vmax \u00b1 {fit.stderr['vmax']:.2f}.")
    return fig


def fig_linearizations():
    S, v, truth, fit = _example_fit()
    smooth = np.geomspace(min(S) / 3, max(S) * 3, 300)
    vfit = models.michaelis_menten(smooth, **fit.params)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 5.0))
    panels = (
        ("Lineweaver-Burk", linearize.lineweaver_burk(S, v), (1 / smooth, 1 / vfit),
         "1/[S]  (1/arbitrary units)", "1/v  (1/arbitrary units)"),
        ("Eadie-Hofstee", linearize.eadie_hofstee(S, v), (vfit / smooth, vfit),
         "v/[S]  (rate per substrate)", "v  (arbitrary units)"),
        ("Hanes-Woolf", linearize.hanes_woolf(S, v), (smooth, smooth / vfit),
         "[S]  (arbitrary units)", "[S]/v  (arbitrary time)"),
    )
    for ax, (name, (x, y), (fx, fy), xl, yl) in zip(axes, panels):
        ax.plot(fx, fy, color="C3", lw=1.7, zorder=2)
        ax.scatter(x, y, s=32, edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name)
    handles = [Line2D([], [], marker="o", ls="", color="C0", markeredgecolor="black",
                      markeredgewidth=0.5, label=f"simulated data (n = {len(S)})"),
               Line2D([], [], color="C3", lw=1.7,
                      label=f"nonlinear fit\nKm = {fit.params['km']:.1f}, "
                            f"Vmax = {fit.params['vmax']:.1f}")]
    fig.suptitle(f"The same {len(S)} points in the three linear plots: the nonlinear fit is a "
                 "straight line in each,\nand Lineweaver-Burk bunches the high-[S] points "
                 "at the origin", fontsize=FS_SUPTITLE)
    _finish(fig, "Simulation seed 42, 5% proportional noise, true Km = 25, Vmax = 100.",
            legend=handles)
    return fig


def fig_inhibition_fit():
    S, I, v, truth = simulate.inhibition_grid(mechanism="mixed", ki=10.0, kis=40.0,
                                              noise=0.03, seed=3)
    results = fitting.compare_models(S, I, v)
    best = results[0]
    fn, _ = models.INHIBITION_MODELS[best.model]
    smooth = np.geomspace(min(S[S > 0]) / 2, max(S) * 1.5, 300)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for j, lvl in enumerate(sorted(np.unique(I))):
        m = I == lvl
        ax.scatter(S[m], v[m], s=28, color=f"C{j}", zorder=3, edgecolor="black",
                   linewidth=0.4)
        ax.plot(smooth, fn(smooth, np.full_like(smooth, lvl), **best.params),
                color=f"C{j}", lw=1.6, label=f"[I] = {lvl:g}")
    daicc = {r.model: r.aicc - best.aicc for r in results}
    others = "; ".join(f"{k} {daicc[k]:.1f}" for k in
                       sorted((k for k in daicc if k != best.model), key=daicc.get))
    ax.set_xlabel("substrate [S] (arbitrary units)")
    ax.set_ylabel("initial rate v (arbitrary units)")
    ax.set_ylim(bottom=0)
    lowest = min(d for k, d in daicc.items() if k != best.model)
    ax.set_title(f"One global {best.model} fit describes all {len(np.unique(I))} inhibitor "
                 f"curves;\nthe other mechanisms lose by \u0394AICc \u2265 {lowest:.0f}")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", color="0.6", markeredgecolor="black",
                          markeredgewidth=0.4, label=f"data (n = {len(S)}); line = global fit"))
    ax.legend(handles=handles, loc="lower right", fontsize=FS_NOTE, frameon=False)
    _finish(fig, f"Simulation seed 3, truth: mixed, Ki = 10, Kis = 40; fitted Ki = "
                 f"{best.params['ki']:.1f}, Kis = {best.params['kis']:.1f}. \u0394AICc against "
                 f"the best model: {others}.")
    return fig


def fig_published_p99():
    stems = list(datasets.SUBSTRATE_UM)
    traces = [datasets.progress_curve(s) for s in stems]
    prog = progress.fit_progress_global(traces)
    table = datasets.initial_rate_table(0.15)
    S = np.array([r["s0_um"] for r in table])
    rates = np.array([r["v_um_per_min"] for r in table])
    sem = [r["v_sem"] for r in table]
    rate_fit = fitting.fit_michaelis_menten(S, rates)
    s0_levels = sorted({s0 for _, _, s0 in traces})
    colour = {s0: f"C{i}" for i, s0 in enumerate(s0_levels)}
    pub_km = datasets.PUBLISHED_KM_M * 1e6
    pub_vmax = (datasets.PUBLISHED_KCAT_S * datasets.ENZYME_CONC_M) * 60 * 1e6
    above_prog = 100 * (prog.km / pub_km - 1)
    above_rate = 100 * (rate_fit.params["km"] / pub_km - 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.9))
    ax = axes[0]
    for (t, p, s0), base in zip(traces, prog.baselines):
        ax.plot(t, p, lw=1.0, alpha=0.75, color=colour[s0])
        ax.plot(t, progress.product_from_time(t, prog.vmax, prog.km, s0, baseline=base),
                lw=0.9, color="black", ls="--")
    handles = [Line2D([], [], color=colour[s0], lw=1.8,
                      label=f"[S]\u2080 = {s0:g} \u00b5M  (n = {sum(1 for _, _, x in traces if x == s0)})")
               for s0 in s0_levels]
    handles.append(Line2D([], [], color="black", lw=1.0, ls="--",
                          label="one global fit to all traces"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              fontsize=FS_NOTE - 0.5, frameon=False)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("product (\u00b5M)")
    ax.set_title(f"{len(traces)} traces share one Vmax = {prog.vmax:.1f} \u00b5M/min\n"
                 f"and one Km = {prog.km:.1f} \u00b5M")

    ax = axes[1]
    ss = np.linspace(0, S.max() * 1.15, 300)
    ax.plot(ss, pub_vmax * ss / (pub_km + ss), color=GREY, lw=1.4, ls=":",
            label=f"published constants (Km {pub_km:g} \u00b5M, kcat {datasets.PUBLISHED_KCAT_S:g}/s)")
    ax.plot(ss, rate_fit.params["vmax"] * ss / (rate_fit.params["km"] + ss), color="C0",
            lw=1.7, label=f"initial-rate fit, Km = {rate_fit.params['km']:.1f} \u00b5M")
    ax.plot(ss, prog.vmax * ss / (prog.km + ss), color="C3", lw=1.7, ls="--",
            label=f"progress-curve fit, Km = {prog.km:.1f} \u00b5M")
    ax.errorbar(S, rates, yerr=sem, fmt="o", ms=5.5, capsize=3, color="C0", mec="black",
                mew=0.5, zorder=3, label="initial rates, 15% window (\u00b1 SEM)")
    ax.set_xlabel("starting substrate [S]\u2080 (\u00b5M)")
    ax.set_ylabel("rate v (\u00b5M/min)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=FS_NOTE - 0.5,
              frameon=False)
    ax.set_title("Michaelis-Menten curves from the two analyses\nof the same traces")
    fig.suptitle(f"Progress curves put Km at {prog.km:.1f} \u00b5M, {above_prog:.0f}% above the "
                 f"published {pub_km:g}; initial rates put it {above_rate:.0f}% above",
                 fontsize=FS_SUPTITLE)
    _finish(fig, f"n = {len(traces)} traces, {len(s0_levels)} [S]\u2080, "
                 f"{sum(len(t) for t, _, _ in traces)} points, "
                 f"{datasets.ENZYME_CONC_M * 1e9:.1f} nM enzyme (Fr\u00e8re et al. 2023; "
                 "data/frere2023_nitrocefin). Published Vmax = published kcat \u00d7 [E].")
    return fig


def fig_ic50_conversion():
    table = datasets.load_chembl_dhfr_ic50()
    ics = np.array([r["ic50_nM"] for r in table])
    enzyme_nm = 10.0
    thr = cp.tight_binding_threshold(enzyme_nm)
    n_tight = int(np.sum(ics < thr))
    n_high = int(np.sum(ics > 1e6))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.9))

    ax = axes[0]
    ratios = np.logspace(-1, 2, 200)
    ax.plot(ratios, [cp.correction_factor(r, 1.0, "competitive") for r in ratios], lw=2,
            color="C0", label="competitive: 1 + [S]/Km")
    ax.plot(ratios, [cp.correction_factor(r, 1.0, "uncompetitive") for r in ratios], lw=2,
            color="C3", label="uncompetitive: 1 + Km/[S]")
    ax.axhline(1.0, ls=":", color=GREY, lw=1.2, label="noncompetitive: 1")
    ax.axvline(1.0, ls="--", color=GREY, lw=0.9, alpha=0.7)
    ax.text(1.08, 0.04, "[S] = Km", transform=ax.get_xaxis_transform(), fontsize=FS_NOTE,
            color="0.35", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("[S] / Km (dimensionless)")
    ax.set_ylabel("IC50 / Ki (dimensionless)")
    lo_f = cp.correction_factor(0.1, 1.0, "competitive")
    hi_f = cp.correction_factor(100.0, 1.0, "competitive")
    ax.set_title(f"The IC50\u2192Ki correction runs from {lo_f:.1f}\u00d7 to {hi_f:.0f}\u00d7\n"
                 "as [S]/Km rises from 0.1 to 100")
    ax.legend(loc="upper left", fontsize=FS_NOTE, frameon=False)

    ax = axes[1]
    logs = np.log10(ics)
    ax.hist(logs, bins=40, color="C0", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvspan(logs.min() - 0.3, math.log10(thr), color="C3", alpha=0.13, lw=0)
    ax.axvline(math.log10(thr), color="C3", lw=1.1, ls="--")
    ax.set_xlim(logs.min() - 0.3, logs.max() + 0.3)
    ax.set_xlabel("log10 IC50 (nM)")
    ax.set_ylabel(f"IC50 records (n = {len(ics)})")
    ax.set_title(f"{n_tight} of {len(ics)} IC50 records ({100 * n_tight / len(ics):.0f}%) lie "
                 f"below 10\u00d7[E] = {thr:g} nM,\nwhere Cheng-Prusoff does not apply "
                 f"(assumed [E] = {enzyme_nm:g} nM)")
    handles = [Patch(facecolor="C3", alpha=0.25,
                     label=f"IC50 < {thr:g} nM: tight-binding\nregime ({n_tight} records)"),
               Patch(facecolor="C0", alpha=0.85, label=f"all records (n = {len(ics)})")]
    ax.legend(handles=handles, loc="upper right", fontsize=FS_NOTE, frameon=False)
    ax.annotate(f"{n_high} records above 10$^6$ nM\n(flagged outside_typical_range, kept)",
                xy=(7.6, 3), xytext=(5.4, 17), fontsize=FS_NOTE - 0.5, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    _finish(fig, f"E. coli DHFR (ChEMBL {datasets.CHEMBL_DHFR_TARGET}): {len(ics)} IC50 records, "
                 f"{len({r['molecule_chembl_id'] for r in table})} compounds. "
                 "Source: data/chembl_dhfr/ecoli_dhfr_ic50.csv.")
    return fig


# ---------------------------------------------------------------------------
# registry and command line
# ---------------------------------------------------------------------------

# name -> (builder, the finding the figure states, source artefact(s))
FIGURES = {
    "km_recovery.png": (
        fig_km_recovery,
        "Lineweaver-Burk fails only under constant absolute error",
        "seeded simulation (src/enzkin/simulate.py)"),
    "mm_fit.png": (
        fig_mm_fit,
        "one 8-point titration recovers Km = 25.5, Vmax = 98.0 (true 25, 100)",
        "seeded simulation"),
    "linearizations.png": (
        fig_linearizations,
        "the nonlinear fit is a straight line in all three linear plots",
        "seeded simulation"),
    "inhibition_fit.png": (
        fig_inhibition_fit,
        "one global mixed fit describes all four [I] curves; others lose by dAICc >= 53",
        "seeded simulation"),
    "published_p99.png": (
        fig_published_p99,
        "progress curves put Km 18% above the published value, initial rates 36%",
        "data/frere2023_nitrocefin/*.csv"),
    "ic50_conversion.png": (
        fig_ic50_conversion,
        "the IC50->Ki correction is 1x-100x; 39% of IC50 records are tight-binding",
        "data/chembl_dhfr/ecoli_dhfr_ic50.csv"),
    "cp_two_substrate.png": (
        fig_cp_two_substrate,
        "all 11 compounds inside the Cheng-Prusoff window, median 2.18",
        "results/cheng_prusoff_two_substrate.csv"),
    "cp_ratio_spread.png": (
        fig_cp_ratio_spread,
        "IC50/Ki group medians run 3.0-20.9, within-group spreads 2.0x-26.9x",
        "results/cheng_prusoff_pairs.csv + ratio_stats.csv"),
    "qsar_tail.png": (
        fig_qsar_tail,
        "1996 QSAR collapses at the 100 nM cutoff, 1988 survives",
        "results/qsar_tail_sweep.csv"),
    "qsar_mw_scaffold.png": (
        fig_qsar_mw_scaffold,
        "molecular weight alone carries the 1988 within-scaffold fit (+0.009)",
        "results_qsar.txt + results_single_scaffold.txt"),
    "km_uncertainty.png": (
        fig_km_uncertainty,
        "the i.i.d. Km SE is 12x too small; the honest SE is 2.21 uM",
        "results_published.txt"),
}


def build(name, out_dir):
    builder = FIGURES[name][0]
    with plt.rc_context(STYLE):
        fig = builder()
        path = Path(out_dir) / name
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(FIG_DIR), help="output directory (default figures/)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="figure names (with or without .png)")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    args = ap.parse_args(argv)
    if args.list:
        for name, (_, finding, src) in FIGURES.items():
            print(f"{name:24s} {finding}\n{'':24s} <- {src}")
        return 0
    names = list(FIGURES)
    if args.only:
        wanted = [n if n.endswith(".png") else n + ".png" for n in args.only]
        unknown = [n for n in wanted if n not in FIGURES]
        if unknown:
            ap.error(f"unknown figure(s) {unknown}; try --list")
        names = wanted
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"{name} -> {build(name, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
