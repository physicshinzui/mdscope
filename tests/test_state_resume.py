from pathlib import Path

from mdscope.config import AppConfig
from mdscope.pipeline import run_pipeline


def test_resume_skips_done(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["traj0"],
            },
            "output": {"outdir": str(tmp_path / "results")},
        }
    )

    run_pipeline(
        config=cfg,
        config_text="a",
        resume=False,
        force=False,
        only={"ligand_site"},
        force_steps=set(),
        from_step=None,
        until_step=None,
    )

    done = Path(cfg.output.outdir) / ".state" / "ligand_site.done"
    assert done.exists()

    run_pipeline(
        config=cfg,
        config_text="a",
        resume=True,
        force=False,
        only={"ligand_site"},
        force_steps=set(),
        from_step=None,
        until_step=None,
    )
    assert done.exists()
