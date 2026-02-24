from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig

# Reference for maximum solvent accessibility values used for rSASA:
# Tien, M. Z.; Meyer, A. G.; Sydykova, D. K.; Spielman, S. J.; Wilke, C. O.
# Maximum Allowed Solvent Accessibilites of Residues in Proteins.
# PLoS One 2013, 8 (11), e80635.
TIEN2013_MAX_ASA: dict[str, float] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}


@dataclass
class RunContext:
    config: AppConfig
    outdir: Path
    cache: dict[str, Any]


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    paths = {
        "tables": outdir / "tables",
        "figures": outdir / "figures",
        "data": outdir / "data",
        "representatives": outdir / "representatives",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _imports() -> tuple[Any, Any, Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import MDAnalysis as mda

    return mda, np, pd, plt


def _frame_slice(config: AppConfig) -> slice:
    return slice(config.frames.start, config.frames.stop, config.frames.step)


def _trajectory_names(config: AppConfig) -> list[str]:
    if config.system.trajectory_names:
        return config.system.trajectory_names
    return [f"traj{i}" for i, _ in enumerate(config.system.trajectories)]


def _load_universes(config: AppConfig) -> list[tuple[str, Any]]:
    mda, _, _, _ = _imports()
    names = _trajectory_names(config)
    universes = []
    topologies = config.system.expanded_topologies()
    for name, topology, trajectory in zip(names, topologies, config.system.trajectories):
        universes.append((name, mda.Universe(str(topology), str(trajectory))))
    return universes


def _save_plot(
    config: AppConfig,
    fig: Any,
    out_prefix: Path,
) -> None:
    for fmt in config.output.figure_formats:
        kwargs: dict[str, Any] = {}
        if fmt == "png":
            kwargs["dpi"] = config.output.dpi
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), bbox_inches="tight", **kwargs)


def _plot_timeseries_and_distribution(
    config: AppConfig,
    df: Any,
    x_col: str,
    y_col: str,
    hue_col: str,
    base_name: str,
    ylabel: str,
) -> None:
    _, np, _, plt = _imports()

    fig_ts, ax_ts = plt.subplots(figsize=(7.2, 4.5))
    for key, sub in df.groupby(hue_col):
        ax_ts.plot(sub[x_col], sub[y_col], label=str(key), lw=1.2)
    ax_ts.set_xlabel(x_col)
    ax_ts.set_ylabel(ylabel)
    ax_ts.legend(loc="best")
    _save_plot(config, fig_ts, Path(config.output.outdir) / "figures" / f"{base_name}_timeseries")
    plt.close(fig_ts)

    if not config.plot.timeseries_distribution:
        return

    fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
    kind = config.plot.distribution_kind
    for key, sub in df.groupby(hue_col):
        values = sub[y_col].dropna().to_numpy()
        if len(values) == 0:
            continue
        if kind in {"hist", "hist_kde"}:
            bins = config.plot.hist_bins
            if bins is None:
                bins = _auto_hist_bins(values, method=config.plot.hist_bin_method)
            ax_dist.hist(values, bins=bins, density=True, alpha=0.35, label=str(key))
        if kind in {"kde", "hist_kde"}:
            xs = np.linspace(values.min(), values.max(), 200)
            if values.std() > 0:
                bw = 1.06 * values.std() * (len(values) ** (-1.0 / 5.0))
                bw = max(bw, 1e-6)
                kernel = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2)
                density = kernel.sum(axis=1) / (len(values) * bw * np.sqrt(2.0 * np.pi))
                ax_dist.plot(xs, density, lw=1.2, label=f"{key} kde")
    ax_dist.set_xlabel(ylabel)
    ax_dist.set_ylabel("density")
    ax_dist.legend(loc="best")
    _save_plot(config, fig_dist, Path(config.output.outdir) / "figures" / f"{base_name}_distribution")
    plt.close(fig_dist)


def _auto_hist_bins(values: Any, method: str = "fd") -> int:
    _, np, _, _ = _imports()
    values = np.asarray(values)
    if len(values) < 2:
        return 1
    if np.allclose(values.min(), values.max()):
        return 1

    edges = np.histogram_bin_edges(values, bins=method)
    bins = max(len(edges) - 1, 1)
    return int(min(max(bins, 10), 120))


def _block_slices(n: int, n_blocks: int) -> list[tuple[int, int]]:
    _, np, _, _ = _imports()
    if n <= 0:
        return []
    blocks = np.array_split(np.arange(n), max(n_blocks, 1))
    spans: list[tuple[int, int]] = []
    for b in blocks:
        if len(b) == 0:
            continue
        spans.append((int(b[0]), int(b[-1]) + 1))
    return spans
