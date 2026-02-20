from __future__ import annotations

from .config import AppConfig


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
