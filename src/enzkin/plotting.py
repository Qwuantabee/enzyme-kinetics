"""Figures. Nothing clever, just the plots you actually need to look at."""

import matplotlib.pyplot as plt
import numpy as np

from . import linearize, models


def plot_fit(S, v, fit, ax=None, label=None, show_km=True):
    """Data points plus the fitted curve on untransformed axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    smooth = np.geomspace(min(S[S > 0]) / 2, max(S) * 1.5, 300)
    curve = models.michaelis_menten(smooth, **fit.params)

    ax.scatter(S, v, s=28, zorder=3, label=label or "data",
               edgecolor="black", linewidth=0.5)
    ax.plot(smooth, curve, lw=1.8, zorder=2, color="C3", label="fit")

    if show_km:
        km, vmax = fit.params["km"], fit.params["vmax"]
        ax.axhline(vmax, ls=":", lw=1, color="grey")
        ax.axvline(km, ls=":", lw=1, color="grey")
        ax.annotate(f"$V_{{max}}$ = {vmax:.1f}", xy=(smooth[0], vmax),
                    xytext=(0, 4), textcoords="offset points", fontsize=8,
                    color="grey")
        ax.annotate(f"$K_m$ = {km:.1f}", xy=(km, 0), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color="grey")

    ax.set_xlabel("[S]")
    ax.set_ylabel("v")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_inhibition(S, I, v, fit, ax=None):
    """One curve per inhibitor concentration, all from the single global fit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))

    fn, _ = models.INHIBITION_MODELS[fit.model]
    smooth = np.geomspace(min(S[S > 0]) / 2, max(S) * 1.5, 300)

    for j, lvl in enumerate(sorted(np.unique(I))):
        m = I == lvl
        ax.scatter(S[m], v[m], s=24, color=f"C{j}", zorder=3,
                   edgecolor="black", linewidth=0.4)
        ax.plot(smooth, fn(smooth, np.full_like(smooth, lvl), **fit.params),
                color=f"C{j}", lw=1.5, label=f"[I] = {lvl:g}")

    ax.set_xlabel("[S]")
    ax.set_ylabel("v")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{fit.model} fit", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_linearizations(S, v, fit, truth=None):
    """The three classic transforms side by side with the nonlinear fit
    drawn through them, so you can see that the nonlinear fit is a perfectly
    good straight line in every one of these spaces -- the transforms are
    fine to look at, they are just bad to regress on."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    smooth = np.geomspace(min(S[S > 0]) / 3, max(S) * 3, 300)
    vfit = models.michaelis_menten(smooth, **fit.params)

    x, y = linearize.lineweaver_burk(S, v)
    axes[0].scatter(x, y, s=24, edgecolor="black", linewidth=0.4)
    axes[0].plot(1 / smooth, 1 / vfit, color="C3", lw=1.5)
    axes[0].set(xlabel="1/[S]", ylabel="1/v", title="Lineweaver-Burk")

    x, y = linearize.eadie_hofstee(S, v)
    axes[1].scatter(x, y, s=24, edgecolor="black", linewidth=0.4)
    axes[1].plot(vfit / smooth, vfit, color="C3", lw=1.5)
    axes[1].set(xlabel="v/[S]", ylabel="v", title="Eadie-Hofstee")

    x, y = linearize.hanes_woolf(S, v)
    axes[2].scatter(x, y, s=24, edgecolor="black", linewidth=0.4)
    axes[2].plot(smooth, smooth / vfit, color="C3", lw=1.5)
    axes[2].set(xlabel="[S]", ylabel="[S]/v", title="Hanes-Woolf")

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_recovery(nl_errors, lb_errors, hw_errors=None):
    """Distribution of relative error in Km across many simulated datasets."""
    fig, ax = plt.subplots(figsize=(6, 4))
    series = [("nonlinear fit", nl_errors), ("Lineweaver-Burk", lb_errors)]
    if hw_errors is not None:
        series.append(("Hanes-Woolf", hw_errors))

    ax.boxplot([100 * np.asarray(e) for _, e in series],
               tick_labels=[n for n, _ in series], showfliers=False)
    ax.axhline(0, ls="--", lw=1, color="grey")
    ax.set_ylabel("error in $K_m$ (%)")
    ax.set_title("Parameter recovery over 500 simulated datasets", fontsize=10)
    fig.tight_layout()
    return fig
