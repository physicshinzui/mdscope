from __future__ import annotations

from pathlib import Path

from .analysis._common import _imports, _plot_timeseries_and_distribution, ensure_dirs
from .analysis.cluster import plot_cluster_from_tables
from .analysis.pca import plot_pca_from_scores
from .analysis.pocket import plot_pocket_from_tables
from .analysis.water import plot_water_from_tables
from .config import AppConfig
from .plotting import apply_publication_style

REPLOTTERS = (
    "rmsd",
    "rg",
    "rmsf",
    "distance",
    "pca",
    "cluster",
    "water",
    "pocket",
)


def _load_table(path: Path):
    _, _, pd, _ = _imports()
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _replot_rmsd(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    df = _load_table(dirs["tables"] / "rmsd_vs_reference.csv")
    if len(df) == 0:
        return []
    _plot_timeseries_and_distribution(config, df, "frame", "rmsd", "trajectory", "rmsd", "RMSD")
    return ["rmsd"]


def _replot_rg(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    df = _load_table(dirs["tables"] / "rg_timeseries.csv")
    if len(df) == 0:
        return []
    _plot_timeseries_and_distribution(config, df, "frame", "rg", "trajectory", "rg", "Radius of gyration")
    return ["rg"]


def _replot_rmsf(config: AppConfig, outdir: Path) -> list[str]:
    _, _, _, plt = _imports()
    dirs = ensure_dirs(outdir)
    df = _load_table(dirs["tables"] / "rmsf_per_residue.csv")
    if len(df) == 0:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for traj_name, sub in df.groupby("trajectory"):
        ax.plot(sub["resid"], sub["rmsf"], lw=1.2, label=str(traj_name))
    ax.set_xlabel("resid")
    ax.set_ylabel("RMSF")
    ax.legend(loc="best")
    from .analysis._common import _save_plot

    _save_plot(config, fig, dirs["figures"] / "rmsf_per_residue")
    plt.close(fig)
    return ["rmsf_per_residue"]


def _replot_distance(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    df = _load_table(dirs["tables"] / "distance_timeseries.csv")
    if len(df) == 0:
        return []
    rendered: list[str] = []
    for pair_id, sub in df.groupby("pair_id"):
        _plot_timeseries_and_distribution(config, sub, "frame", "distance", "trajectory", f"distance_{pair_id}", f"Distance {pair_id}")
        rendered.append(f"distance_{pair_id}")
    return rendered


def _replot_pca(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    scores = _load_table(dirs["tables"] / "pca_scores.csv")
    plot_pca_from_scores(config, scores, outdir)
    return ["pca"]


def _replot_cluster(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    scores = _load_table(dirs["tables"] / "pca_scores.csv")
    labels_df = _load_table(dirs["tables"] / "hdbscan_labels.csv")
    plot_cluster_from_tables(config, scores, labels_df, outdir)
    return ["cluster"]


def _replot_water(config: AppConfig, outdir: Path) -> list[str]:
    return plot_water_from_tables(config, outdir)


def _replot_pocket(config: AppConfig, outdir: Path) -> list[str]:
    dirs = ensure_dirs(outdir)
    pockets_df = _load_table(dirs["tables"] / "pocket_fpocket_summary.csv")
    return plot_pocket_from_tables(config, pockets_df, outdir)


REPLOTTER_FUNCS = {
    "rmsd": _replot_rmsd,
    "rg": _replot_rg,
    "rmsf": _replot_rmsf,
    "distance": _replot_distance,
    "pca": _replot_pca,
    "cluster": _replot_cluster,
    "water": _replot_water,
    "pocket": _replot_pocket,
}


def replot_results(config: AppConfig, outdir: Path, only: set[str] | None = None) -> dict[str, list[str]]:
    apply_publication_style(config)
    selected = only or set(REPLOTTERS)
    unknown = sorted(selected - set(REPLOTTER_FUNCS))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown replot target(s): {joined}")

    results: dict[str, list[str]] = {}
    missing: list[str] = []
    for name in REPLOTTERS:
        if name not in selected:
            continue
        try:
            results[name] = REPLOTTER_FUNCS[name](config, outdir)
        except FileNotFoundError as exc:
            missing.append(f"{name}: {exc.filename}")
    if missing and only:
        joined = "; ".join(missing)
        raise RuntimeError(f"Missing input tables for replot: {joined}")
    return results
