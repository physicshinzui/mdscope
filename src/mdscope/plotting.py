from __future__ import annotations

from .config import AppConfig


def plot_contact_difference_bars(ax, labels, values, group_a, group_b):
    """Display selected contacts in the supplied ranking order, top to bottom."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    order = np.arange(len(values))
    blue, orange = "#2878B5", "#E38B2C"
    ax.barh(
        np.arange(len(values)), values[order], height=0.65,
        color=[blue if v < 0 else orange for v in values[order]],
        edgecolor="none", zorder=3,
    )
    ax.set_yticks(np.arange(len(values)), [labels[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(f"OCC({group_b}) - OCC({group_a})", labelpad=10)
    ax.set_ylabel("Residue pair", labelpad=12)
    ax.set_title(f"Top {len(values)} contact differences", pad=36)
    ax.text(0.25, 1.025, f"Reduced in {group_b}", transform=ax.transAxes,
            ha="center", va="bottom", color=blue, fontsize=9)
    ax.text(0.75, 1.025, f"Increased in {group_b}", transform=ax.transAxes,
            ha="center", va="bottom", color=orange, fontsize=9)
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#444444", linewidth=1.0, zorder=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(axis="y", length=0, pad=8)


def apply_publication_style(config: AppConfig) -> None:
    if not config.plot.publication_style:
        return

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "figure.titlesize": 10.0,
            "figure.dpi": float(config.output.dpi),
            "savefig.dpi": float(config.output.dpi),
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "image.cmap": "viridis",
        }
    )
