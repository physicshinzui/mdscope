from mdscope.config import AppConfig
from mdscope.pipeline import resolve_targets


def test_only_cluster_expands_dependency(tmp_path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["t0"],
            },
            "analyses": {"pca": True, "cluster": True},
        }
    )

    steps = resolve_targets(cfg, only={"cluster"}, from_step=None, until_step=None)
    names = [s.name for s in steps]
    assert "pca" in names
    assert "cluster" in names


def test_only_convergence_expands_dependencies(tmp_path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["t0"],
            },
            "analyses": {"rmsd": True, "rg": True, "pca": True, "cluster": True, "convergence": True},
        }
    )

    steps = resolve_targets(cfg, only={"convergence"}, from_step=None, until_step=None)
    names = [s.name for s in steps]
    assert "rmsd" in names
    assert "rg" in names
    assert "pca" in names
    assert "cluster" in names
    assert "convergence" in names
