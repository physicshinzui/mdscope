from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from mdscope.config import AppConfig
from mdscope.replot import replot_results


def test_replot_results_regenerates_pca_and_cluster_figures(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    outdir = tmp_path / "results"
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trajectory": "run1", "frame": 0, "PC1": 0.0, "PC2": 0.1},
            {"trajectory": "run1", "frame": 1, "PC1": 0.2, "PC2": 0.3},
            {"trajectory": "ref1", "frame": -1, "PC1": 1.0, "PC2": 1.1},
            {"trajectory": "ref2", "frame": -1, "PC1": 1.2, "PC2": 1.3},
        ]
    ).to_csv(tables / "pca_scores.csv", index=False)
    pd.DataFrame(
        [
            {"trajectory": "run1", "frame": 0, "label": 0, "cluster_name": "run1"},
            {"trajectory": "run1", "frame": 1, "label": 0, "cluster_name": "run1"},
        ]
    ).to_csv(tables / "hdbscan_labels.csv", index=False)

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run1"],
            },
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"pca": True, "cluster": True},
            "pca": {"free_energy_enabled": False},
        }
    )

    results = replot_results(cfg, outdir, only={"pca", "cluster"})

    assert "pca" in results
    assert "cluster" in results
    assert (outdir / "figures" / "pca_scatter.png").exists()
    assert (outdir / "figures" / "pca_scatter_run1.png").exists()
    assert (outdir / "figures" / "pca_scatter_clustered.png").exists()
    img = mpimg.imread(outdir / "figures" / "pca_scatter_clustered.png")
    rgb = img[..., :3].reshape(-1, 3)
    mask = np.any(rgb < 0.99, axis=1)
    unique_colors = np.unique(np.round(rgb[mask], 2), axis=0)
    assert len(unique_colors) >= 3


def test_replot_results_regenerates_rmsd_and_water_figures(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    outdir = tmp_path / "results"
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trajectory": "run1", "frame": 0, "rmsd": 1.0},
            {"trajectory": "run1", "frame": 1, "rmsd": 1.2},
        ]
    ).to_csv(tables / "rmsd_vs_reference.csv", index=False)
    pd.DataFrame(
        [
            {"trajectory": "run1", "frame": 0, "water_count": 2},
            {"trajectory": "run1", "frame": 1, "water_count": 3},
        ]
    ).to_csv(tables / "water_occupancy_timeseries.csv", index=False)
    pd.DataFrame(
        [
            {"trajectory": "run1", "r_angstrom": 1.0, "g_r": 0.5},
            {"trajectory": "run1", "r_angstrom": 2.0, "g_r": 1.0},
        ]
    ).to_csv(tables / "water_rdf.csv", index=False)
    pd.DataFrame(
        [
            {"trajectory": "run1", "water_id": 1, "start_frame": 0, "end_frame": 1, "duration_frames": 2, "duration_frame_span": 1},
        ]
    ).to_csv(tables / "water_residence_events.csv", index=False)

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run1"],
            },
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"rmsd": True, "water": True},
        }
    )

    results = replot_results(cfg, outdir, only={"rmsd", "water"})

    assert "rmsd" in results
    assert "water" in results
    assert (outdir / "figures" / "rmsd_timeseries.png").exists()
    assert (outdir / "figures" / "rmsd_distribution.png").exists()
    assert (outdir / "figures" / "water_occupancy_timeseries.png").exists()
    assert (outdir / "figures" / "water_rdf.png").exists()
    assert (outdir / "figures" / "water_residence_duration_distribution.png").exists()
