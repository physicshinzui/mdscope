from pathlib import Path

import mdscope.pipeline as pipeline
from mdscope.config import AppConfig


def test_resume_skips_done(tmp_path: Path, monkeypatch) -> None:
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
            "analyses": {"rmsd": True},
        }
    )

    def _noop_execute_step(step: str, ctx) -> None:
        return None

    monkeypatch.setattr(pipeline, "execute_step", _noop_execute_step)

    pipeline.run_pipeline(
        config=cfg,
        config_text="a",
        resume=False,
        force=False,
        only={"rmsd"},
        force_steps=set(),
        from_step=None,
        until_step=None,
    )

    done = Path(cfg.output.outdir) / ".state" / "rmsd.done"
    assert done.exists()

    pipeline.run_pipeline(
        config=cfg,
        config_text="a",
        resume=True,
        force=False,
        only={"rmsd"},
        force_steps=set(),
        from_step=None,
        until_step=None,
    )
    assert done.exists()
