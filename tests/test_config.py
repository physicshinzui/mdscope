from pathlib import Path

import pytest

from mdscope.config import generate_template, load_config


def test_generate_template_contains_system() -> None:
    text = generate_template("standard")
    assert "system:" in text
    assert "topology:" in text


def test_load_config_valid(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "preset: standard",
                "system:",
                f"  topology: {top}",
                "  trajectories:",
                f"    - {trj}",
                "  trajectory_names:",
                "    - t0",
            ]
        )
    )

    cfg = load_config(cfg_path)
    assert cfg.system.trajectory_names == ["t0"]


def test_load_config_with_per_trajectory_topologies(tmp_path: Path) -> None:
    top1 = tmp_path / "top1.pdb"
    top2 = tmp_path / "top2.pdb"
    trj1 = tmp_path / "traj1.xtc"
    trj2 = tmp_path / "traj2.xtc"
    top1.write_text("x")
    top2.write_text("x")
    trj1.write_text("x")
    trj2.write_text("x")

    cfg_path = tmp_path / "cfg_multi_top.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "preset: standard",
                "system:",
                "  topologies:",
                f"    - {top1}",
                f"    - {top2}",
                "  trajectories:",
                f"    - {trj1}",
                f"    - {trj2}",
                "  trajectory_names:",
                "    - t0",
                "    - t1",
            ]
        )
    )

    cfg = load_config(cfg_path)
    expanded = cfg.system.expanded_topologies()
    assert expanded == [top1, top2]


def test_load_config_rejects_unknown_pca_fit_trajectory(tmp_path: Path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg_path = tmp_path / "cfg_invalid_fit.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "preset: standard",
                "system:",
                f"  topology: {top}",
                "  trajectories:",
                f"    - {trj}",
                "  trajectory_names:",
                "    - t0",
                "pca:",
                "  fit_trajectory: missing",
            ]
        )
    )

    with pytest.raises(ValueError, match="pca.fit_trajectory must be one of: t0"):
        load_config(cfg_path)
