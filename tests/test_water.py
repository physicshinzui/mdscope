from pathlib import Path

import pandas as pd

from mdscope.analysis._common import RunContext
from mdscope.analysis.water import run_water
from mdscope.config import AppConfig


def test_run_water_phase1_with_repo_data(tmp_path: Path) -> None:
    data_dir = Path("tests/data/ace-a-nme-with-water")
    top = data_dir / "ace-a-nme.gro"
    trj = data_dir / "hrex_pbc_corrected.xtc"
    assert top.exists()
    assert trj.exists()

    outdir = tmp_path / "results"
    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run1"],
                "selection": "all",
            },
            "frames": {"start": 0, "stop": 60, "step": 5},
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"water": True},
            "water": {
                "enabled_metrics": ["occupancy", "rdf", "residence"],
                "water_selection": "resname SOL",
                "oxygen_selection": "name OW",
                "region_mode": "selection",
                "region_selection": "name CA",
                "cutoff_angstrom": 4.0,
                "occupancy": {"count_mode": "water_residues", "block_summary": True},
                "rdf": {"solute_selection": "name CA", "range_angstrom": [0.0, 8.0], "nbins": 40},
                "residence": {"min_stay_frames": 1, "gap_tolerance_frames": 0, "report_top_waters": 10},
            },
        }
    )

    run_water(RunContext(config=cfg, outdir=outdir, cache={}))

    occ_csv = outdir / "tables" / "water_occupancy_timeseries.csv"
    occ_summary_csv = outdir / "tables" / "water_occupancy_summary.csv"
    occ_blocks_csv = outdir / "tables" / "water_occupancy_blocks.csv"
    rdf_csv = outdir / "tables" / "water_rdf.csv"
    residence_events_csv = outdir / "tables" / "water_residence_events.csv"
    residence_summary_csv = outdir / "tables" / "water_residence_summary.csv"
    residence_top_csv = outdir / "tables" / "water_residence_top_waters.csv"
    report_json = outdir / "data" / "water_report.json"

    for p in [
        occ_csv,
        occ_summary_csv,
        occ_blocks_csv,
        rdf_csv,
        residence_events_csv,
        residence_summary_csv,
        residence_top_csv,
        report_json,
    ]:
        assert p.exists(), f"missing output: {p}"

    occ_df = pd.read_csv(occ_csv)
    rdf_df = pd.read_csv(rdf_csv)
    events_df = pd.read_csv(residence_events_csv)
    summary_df = pd.read_csv(residence_summary_csv)

    assert len(occ_df) > 0
    assert set(["trajectory", "frame", "water_count"]).issubset(occ_df.columns)
    assert occ_df["water_count"].ge(0).all()
    assert occ_df["water_count"].max() > 0

    assert len(rdf_df) > 0
    assert set(["trajectory", "r_angstrom", "g_r"]).issubset(rdf_df.columns)

    assert set(["trajectory", "water_id", "start_frame", "end_frame", "duration_frames", "duration_frame_span"]).issubset(
        events_df.columns
    )
    assert set(["trajectory", "n_events", "mean_duration", "median_duration", "max_duration"]).issubset(summary_df.columns)
    if len(events_df) > 0:
        assert events_df["duration_frames"].ge(1).all()
        assert events_df["duration_frame_span"].ge(0).all()
