"""
The project's five figures.

Keeping every plot in one module, with a single style function, is what makes
them look like a set rather than five unrelated pictures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis import AXIS_LABELS, SPECTRAL_ORDER, SPECTRAL_RANGES, T_EFF_SUN, TYPE_LABELS

TYPE_COLOURS = {
    "Brown Dwarf": "#6B4226", "Red Dwarf": "#E4572E", "White Dwarf": "#1B9AAA",
    "Main Sequence": "#F0A202", "Supergiant": "#5D2E8C", "Hypergiant": "#D81E5B",
}
#: Approximate true appearance of each spectral class, used for the band shading.
SPECTRAL_COLOURS = {
    "O": "#8092FF", "B": "#A9BEFF", "A": "#D0DCFF", "F": "#F4F4FF",
    "G": "#FFF3E2", "K": "#FFCE9A", "M": "#FF9257",
}
SUN_MV = 4.83               # solar absolute visual magnitude
_TICKS = [2000, 3000, 5000, 7000, 10000, 20000, 40000]


def configure_style() -> None:
    """Install the project-wide plotting style. Call once, before anything else."""
    sns.set_theme(context="notebook", style="whitegrid")
    mpl.rcParams.update({
        "savefig.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFD", "axes.edgecolor": "#8A909C",
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelsize": 11, "grid.color": "#C8CCD4", "grid.linewidth": 0.6,
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "font.sans-serif": ["DejaVu Sans"], "mathtext.default": "regular",
    })


def _save(fig, outdir: Path, name: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _order(df: pd.DataFrame) -> list[str]:
    present = set(df["Type_Label"].astype(str))
    return [TYPE_LABELS[t] for t in sorted(TYPE_LABELS) if TYPE_LABELS[t] in present]


def _temperature_axis(ax, sparse: bool = False) -> None:
    """Log temperature axis, inverted so that hot stars sit on the left."""
    ax.set_xscale("log")
    ticks = _TICKS[1::2] if sparse else _TICKS
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()


def _spectral_bands(ax) -> None:
    """Shade and label the Morgan-Keenan temperature ranges along the top."""
    xmin, xmax = sorted(ax.get_xlim())
    for letter in SPECTRAL_ORDER:
        low, high = SPECTRAL_RANGES[letter]
        low, high = max(low, xmin), min(high if np.isfinite(high) else xmax, xmax)
        if high <= low:
            continue
        ax.axvspan(low, high, color=SPECTRAL_COLOURS[letter], alpha=0.20, zorder=0, lw=0)
        ax.text(np.sqrt(low * high), 1.0, letter, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#4A5160")


def hr_diagram(df: pd.DataFrame, outdir: Path) -> Path:
    """The Hertzsprung-Russell diagram: the project's central figure.

    Temperature increases leftward and brightness upward, both by long-standing
    convention. Marker area encodes radius, which is what makes the four stellar
    populations legible at a glance.
    """
    fig, ax = plt.subplots(figsize=(11, 7.6))
    order = _order(df)

    # One mapping from log radius to marker area, reused by the size legend so
    # the two cannot drift apart.
    log_r = df["logR"].to_numpy()
    lo, hi = log_r.min(), log_r.max()
    size_of = lambda v: 22 + 400 * (np.asarray(v, float) - lo) / (hi - lo)

    for label in order:
        m = df["Type_Label"].astype(str) == label
        ax.scatter(df.loc[m, "Temperature"], df.loc[m, "A_M"], s=size_of(log_r[m.to_numpy()]),
                   c=TYPE_COLOURS[label], label=label, alpha=0.82,
                   edgecolor="white", linewidth=0.7, zorder=3)

    ax.set_xlim(df["Temperature"].min() * 0.72, df["Temperature"].max() * 1.28)
    _temperature_axis(ax)
    ax.set_ylim(df["A_M"].max() + 2, df["A_M"].min() - 2.5)     # magnitudes run backwards
    _spectral_bands(ax)

    ax.scatter([T_EFF_SUN], [SUN_MV], marker="*", s=400, c="#FFD400",
               edgecolor="#7A5B00", linewidth=1.1, zorder=6)
    ax.annotate("The Sun", xy=(T_EFF_SUN, SUN_MV), xytext=(-52, -40),
                textcoords="offset points", fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B0B6C0", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#7A5B00"), zorder=7)

    for text, xy in [("Supergiants & hypergiants", (17500, -12.2)),
                     ("Main sequence", (23000, 2.2)),
                     ("White dwarfs", (12500, 8.4)),
                     ("Red & brown dwarfs", (7600, 18.4))]:
        ax.annotate(text, xy=xy, fontsize=9.5, ha="center", va="center", color="#2B3140",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#B0B6C0", alpha=0.88),
                    zorder=5)

    ax.set_xlabel(AXIS_LABELS["Temperature"] + "   (hotter $\\longleftarrow$)")
    ax.set_ylabel(AXIS_LABELS["A_M"] + "   ($\\longleftarrow$ brighter)")
    ax.set_title("Hertzsprung–Russell diagram of 240 stars", pad=24)

    legend = ax.legend(title="Stellar type", loc="lower left", markerscale=0.55,
                       labelspacing=0.7, borderpad=0.7)
    for handle in legend.legend_handles:
        handle.set_sizes([70])
    ax.add_artist(legend)

    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor="#9AA3B2",
                              markeredgecolor="white", markersize=np.sqrt(size_of(np.log10(r))),
                              label=f"{r:g}") for r in [0.01, 0.1, 1, 10, 100, 1000]],
              title="Radius  $R/R_{\\odot}$", loc="upper right",
              labelspacing=1.5, borderpad=0.8, handletextpad=1.4)

    fig.text(0.5, -0.03, "Letters along the top are Harvard spectral classes; "
             "marker area encodes stellar radius.",
             ha="center", fontsize=9, color="#5A6070", style="italic")
    return _save(fig, outdir, "1_hr_diagram")


def physics_audit(df: pd.DataFrame, fit: dict, outdir: Path) -> Path:
    """Does the catalogue obey the Stefan-Boltzmann law?

    Left: catalogued luminosity against the value physics demands from the same
    star's radius and temperature. Right: the residuals, split by population.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    order = _order(df)

    ax = axes[0]
    lims = [min(df["L"].min(), df["L_sb"].min()) * 0.3,
            max(df["L"].max(), df["L_sb"].max()) * 3]
    ax.fill_between(lims, [v / 2 for v in lims], [v * 2 for v in lims],
                    color="#9AA3B2", alpha=0.18, zorder=0, label="within a factor of 2")
    ax.plot(lims, lims, "--", color="#2B3140", linewidth=1.4, zorder=1, label="perfect agreement")
    for label in order:
        m = df["Type_Label"].astype(str) == label
        ax.scatter(df.loc[m, "L_sb"], df.loc[m, "L"], s=38, c=TYPE_COLOURS[label],
                   alpha=0.82, edgecolor="white", linewidth=0.5, label=label, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims)
    ax.set_xlabel("Predicted  $L = 4\\pi R^{2}\\sigma T^{4}$   $[L_{\\odot}]$")
    ax.set_ylabel("Catalogued luminosity  $[L_{\\odot}]$")
    ax.set_title("(a) Observed vs. predicted luminosity")
    ax.legend(fontsize=7.6, loc="upper left")

    ax = axes[1]
    sns.boxplot(df, x="Type_Label", y="sb_residual", order=order, hue="Type_Label",
                hue_order=order, palette=[TYPE_COLOURS[t] for t in order], legend=False,
                dodge=False, ax=ax, width=0.62, fliersize=0,
                boxprops=dict(alpha=0.55), linecolor="#2B3140", linewidth=1.1)
    sns.stripplot(df, x="Type_Label", y="sb_residual", order=order, ax=ax,
                  color="#1F2430", size=3.2, alpha=0.55, jitter=0.22)
    ax.axhline(0, color="#D81E5B", linestyle="--", linewidth=1.6, label="exact agreement")
    ax.set_xlabel("")
    ax.set_ylabel("$\\log_{10}(L_{\\mathrm{obs}} / L_{\\mathrm{predicted}})$   [dex]")
    ax.set_title("(b) Residuals by population")
    ax.tick_params(axis="x", rotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Does the catalogue obey stellar physics?", fontsize=15,
                 fontweight="bold", y=1.13)
    fig.text(0.5, 1.045, f"Free fit: $L \\propto R^{{{fit['alpha']:.2f}}}"
             f"T^{{{fit['beta']:.2f}}}$   (theory: $R^{{2}}T^{{4}}$)",
             ha="center", fontsize=10.5, color="#3A4152")
    fig.text(0.5, -0.08, "Main-sequence stars sit on the one-to-one line; the other "
             "populations show systematic offsets that a law with no free parameters "
             "cannot accommodate.",
             ha="center", fontsize=9, color="#5A6070", style="italic")
    return _save(fig, outdir, "2_physics_audit")


def power_law(df: pd.DataFrame, fit: dict, outdir: Path) -> Path:
    """The main-sequence temperature-luminosity relation, with a bootstrap band.

    Restricted to the main sequence: fitting all six populations at once would
    give an exponent describing none of them.
    """
    fig, ax = plt.subplots(figsize=(8.6, 6))
    ms = df[df["Type_Label"].astype(str) == "Main Sequence"]

    ax.fill_between(fit["x_curve"], fit["y_lo"], fit["y_hi"], color="#D81E5B",
                    alpha=0.16, zorder=2, label="95% bootstrap band")
    ax.plot(fit["x_curve"], fit["y_curve"], color="#D81E5B", linewidth=2.2, zorder=4,
            label=f"$\\alpha = {fit['alpha']:.2f} \\pm {fit['alpha_err']:.2f}$,"
                  f"  $R^2 = {fit['r_squared']:.3f}$")
    ax.scatter(ms["Temperature"], ms["L"], s=60, c=TYPE_COLOURS["Main Sequence"],
               alpha=0.9, edgecolor="white", linewidth=0.7, zorder=3,
               label=f"main sequence ($n = {len(ms)}$)")

    ax.set(xscale="log", yscale="log")
    _temperature_axis(ax, sparse=True)
    ax.invert_xaxis()                     # undo: this plot reads left-to-right
    ax.set_xlabel(AXIS_LABELS["Temperature"])
    ax.set_ylabel(AXIS_LABELS["L"])
    ax.set_title("Main sequence:  $L \\propto T^{\\alpha}$")
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.5, -0.04, "Stellar structure theory predicts $\\alpha \\approx 5-8$, from "
             "$L \\propto M^{3.5}$ combined with $T \\propto M^{0.5}$.",
             ha="center", fontsize=9, color="#5A6070", style="italic")
    return _save(fig, outdir, "3_power_law")


def model_results(confusion: np.ndarray, class_names: list[str],
                  contributions: pd.DataFrame, outdir: Path) -> Path:
    """Classifier performance, and which measurement it actually depends on."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    with np.errstate(invalid="ignore"):
        norm = np.nan_to_num(confusion / confusion.sum(axis=1, keepdims=True))
    sns.heatmap(norm, annot=confusion, fmt="d", cmap="Blues", vmin=0, vmax=1,
                linewidths=1.2, linecolor="white", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws=dict(fontsize=10, fontweight="bold"))
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    ax.set_title(f"(a) Confusion matrix  —  accuracy {np.trace(confusion) / confusion.sum():.3f}")
    ax.tick_params(axis="x", rotation=35)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    ax = axes[1]
    contrib = contributions.sort_values("alone")
    y = np.arange(len(contrib))
    ax.barh(y + 0.19, contrib["alone"], 0.38, color="#F0A202", edgecolor="white",
            linewidth=1, label="this feature alone")
    ax.barh(y - 0.19, contrib["lost_if_removed"].clip(lower=0), 0.38, color="#D81E5B",
            edgecolor="white", linewidth=1, label="accuracy lost if removed")
    ax.axvline(1 / 6, color="#5A6070", linestyle=":", linewidth=1.4, label="random guessing")
    ax.set_yticks(y)
    ax.set_yticklabels(contrib.index)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cross-validated accuracy")
    ax.set_title("(b) Which measurement carries the signal?")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Classifying stellar type", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06, "Radius alone separates the populations almost completely; "
             "temperature alone barely does. Everything is also individually "
             "dispensable, because the four measurements are physically redundant.",
             ha="center", fontsize=9, color="#5A6070", style="italic")
    return _save(fig, outdir, "4_model_results")


def decision_regions(df: pd.DataFrame, grid: dict, outdir: Path) -> Path:
    """The classifier's decision boundaries, drawn on the HR diagram.

    This is where the machine learning and the astronomy meet: the partition the
    model learns is the one astronomers drew by eye a century ago.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    order = _order(df)
    cmap = mpl.colors.ListedColormap([TYPE_COLOURS[t] for t in order])

    ax.pcolormesh(10 ** grid["xx"], grid["yy"], grid["zz"], cmap=cmap, alpha=0.26,
                  shading="auto", zorder=0, vmin=-0.5, vmax=len(order) - 0.5)
    for label in order:
        m = df["Type_Label"].astype(str) == label
        ax.scatter(df.loc[m, "Temperature"], df.loc[m, "logL"], s=50,
                   c=TYPE_COLOURS[label], alpha=0.95, edgecolor="white",
                   linewidth=0.8, zorder=3)

    ax.set_xlim(10 ** grid["xx"].min(), 10 ** grid["xx"].max())
    _temperature_axis(ax)
    ax.set_ylim(grid["yy"].min(), grid["yy"].max())
    ax.set_xlabel(AXIS_LABELS["Temperature"] + "   (hotter $\\longleftarrow$)")
    ax.set_ylabel(AXIS_LABELS["logL"])
    ax.set_title("Random Forest decision regions on the HR diagram\n"
                 f"(trained on $\\log T$ and $\\log L$ only — test accuracy "
                 f"{grid['accuracy']:.3f})", pad=12)
    ax.legend(handles=[Patch(facecolor=TYPE_COLOURS[t], edgecolor="white", alpha=0.75, label=t)
                       for t in order],
              title="Predicted type", loc="lower left", labelspacing=0.6, borderpad=0.7)

    fig.text(0.5, -0.03, "Shaded areas are the model's prediction at every point; "
             "markers are the observed stars. The rectangular edges are the forest's "
             "own bias — every split is a threshold on one feature.",
             ha="center", fontsize=9, color="#5A6070", style="italic")
    return _save(fig, outdir, "5_decision_regions")
