import json
from pathlib import Path

import pandas as pd

from mdscope.analysis.steps import RunContext, run_convergence
from mdscope.config import AppConfig


def test_run_convergence_writes_outputs(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj1 = tmp_path / "traj1.xtc"
    trj2 = tmp_path / "traj2.xtc"
    top.write_text("x")
    trj1.write_text("x")
    trj2.write_text("x")

    outdir = tmp_path / "results"
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    frames = list(range(20))
    rows_rmsd = []
    rows_rg = []
    rows_pca = []
    rows_cluster = []
    for traj_name, shift in [("WT", 0.0), ("F143W", 0.05)]:
        for frame in frames:
            rows_rmsd.append({"trajectory": traj_name, "frame": frame, "rmsd": 1.0 + 0.02 * (frame % 5) + shift})
            rows_rg.append({"trajectory": traj_name, "frame": frame, "rg": 12.0 + 0.03 * (frame % 4) + shift})
            rows_pca.append(
                {
                    "trajectory": traj_name,
                    "frame": frame,
                    "PC1": 0.1 * frame + shift,
                    "PC2": 0.05 * ((frame % 6) - 3) + shift,
                }
            )
            rows_cluster.append({"trajectory": traj_name, "frame": frame, "label": 0 if frame < 10 else 1})

    pd.DataFrame(rows_rmsd).to_csv(tables / "rmsd_vs_reference.csv", index=False)
    pd.DataFrame(rows_rg).to_csv(tables / "rg_timeseries.csv", index=False)
    pd.DataFrame(rows_pca).to_csv(tables / "pca_scores.csv", index=False)
    pd.DataFrame(rows_cluster).to_csv(tables / "hdbscan_labels.csv", index=False)

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj1), str(trj2)],
                "trajectory_names": ["WT", "F143W"],
            },
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"convergence": True},
            "convergence": {
                "enabled_metrics": ["rmsd", "rg", "pca", "cluster_occupancy"],
                "n_blocks": 4,
                "min_frames": 10,
                "rule": "k_of_n",
                "k_required": 2,
            },
        }
    )

    run_convergence(RunContext(config=cfg, outdir=outdir, cache={}))

    metric_status = outdir / "tables" / "convergence_metric_status.csv"
    summary = outdir / "tables" / "convergence_summary.csv"
    report = outdir / "data" / "convergence_report.json"
    fig_wt = outdir / "figures" / "convergence_cluster_occupancy_blocks_WT.png"
    fig_mut = outdir / "figures" / "convergence_cluster_occupancy_blocks_F143W.png"

    assert metric_status.exists()
    assert summary.exists()
    assert report.exists()
    assert fig_wt.exists()
    assert fig_mut.exists()

    status_df = pd.read_csv(metric_status)
    assert set(status_df["metric"]) == {"rmsd", "rg", "pca", "cluster_occupancy"}

    payload = json.loads(report.read_text())
    assert payload["n_metrics_total"] == 4
    assert isinstance(payload["overall_pass"], bool)
