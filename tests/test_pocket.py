from pathlib import Path

import pandas as pd

from mdscope.analysis.steps import RunContext, run_pocket
from mdscope.config import AppConfig, generate_template


def test_generate_full_template_contains_pocket() -> None:
    text = generate_template("full")
    assert "pocket:" in text
    assert "input_mode:" in text


def test_run_pocket_explicit_pdbs_with_mocked_fpocket(tmp_path: Path, monkeypatch) -> None:
    # Minimal required system files (unused by pocket explicit mode but required by AppConfig)
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    source_dir = tmp_path / "pdbs"
    source_dir.mkdir()
    pdb1 = source_dir / "state_a.pdb"
    pdb2 = source_dir / "state_b.pdb"
    pdb1.write_text("HEADER state_a\nEND\n")
    pdb2.write_text("HEADER state_b\nEND\n")

    outdir = tmp_path / "results"
    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run1"],
            },
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"pocket": True},
            "pocket": {
                "backend": "fpocket",
                "input_mode": "explicit_pdbs",
                "source_dir": str(source_dir),
                "pdb_glob": "*.pdb",
                "top_n_pockets": 1,
                "rank_by": "score",
            },
        }
    )

    import mdscope.analysis.steps as steps

    monkeypatch.setattr(steps, "_fpocket_available", lambda: True)

    def _fake_run_fpocket_command(pdb_path: Path, cwd: Path):
        outdir_local = cwd / f"{pdb_path.stem}_out"
        outdir_local.mkdir(parents=True, exist_ok=True)
        (outdir_local / f"{pdb_path.stem}_info.txt").write_text(
            "\n".join(
                [
                    "Pocket 1 :",
                    "Score : 15.2",
                    "Druggability Score : 0.71",
                    "Volume : 220.0",
                    "Number of alpha spheres : 35",
                    "",
                    "Pocket 2 :",
                    "Score : 8.4",
                    "Druggability Score : 0.20",
                    "Volume : 90.0",
                ]
            )
            + "\n"
        )
        return 0, "ok", ""

    monkeypatch.setattr(steps, "_run_fpocket_command", _fake_run_fpocket_command)

    run_pocket(RunContext(config=cfg, outdir=outdir, cache={}))

    status_csv = outdir / "tables" / "pocket_fpocket_structure_status.csv"
    summary_csv = outdir / "tables" / "pocket_fpocket_summary.csv"
    top_csv = outdir / "tables" / "pocket_fpocket_top_hits.csv"
    report_json = outdir / "data" / "pocket_fpocket_report.json"
    assert status_csv.exists()
    assert summary_csv.exists()
    assert top_csv.exists()
    assert report_json.exists()

    status_df = pd.read_csv(status_csv)
    summary_df = pd.read_csv(summary_csv)
    top_df = pd.read_csv(top_csv)
    assert set(status_df["status"]) == {"ok"}
    assert len(status_df) == 2
    assert len(summary_df) == 4  # 2 pockets x 2 structures
    assert len(top_df) == 2  # top1 per structure
    assert summary_df["score"].max() == 15.2
