from pathlib import Path

import pytest

from mdscope.config import load_config


def test_local_data_config_resolves() -> None:
    cfg_path = Path("configs/local_dual.yaml")
    base = cfg_path.parent.resolve()
    required = [
        base / "../../data/WT/topology.pdb",
        base / "../../data/WT/samples.xtc",
        base / "../../data/F143W/topology.pdb",
        base / "../../data/F143W/samples.xtc",
        base / "../../data/pdbs/2YXJ.pdb",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("local WT/F143W data not available")
    cfg = load_config(cfg_path)
    expanded = cfg.system.expanded_topologies()
    assert len(expanded) == 2
    assert all(p.exists() for p in expanded)
    assert len(cfg.system.trajectories) == 2
    assert cfg.rmsd.reference is not None
    assert cfg.rmsd.reference.exists()


def test_repo_managed_chignolin_data_config_resolves(tmp_path: Path) -> None:
    data_dir = (Path(__file__).parent / "data" / "chignolin").resolve()
    top = data_dir / "chignolin.pdb"
    traj = data_dir / "traj.xtc"
    ref = data_dir / "5awl.pdb"
    assert top.exists()
    assert traj.exists()
    assert ref.exists()

    cfg_path = tmp_path / "chignolin.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "preset: standard",
                "system:",
                f"  topology: {top}",
                "  trajectories:",
                f"    - {traj}",
                "  trajectory_names:",
                "    - chignolin",
                "  selection: protein",
                "  align_selection: backbone",
                "rmsd:",
                f"  reference: {ref}",
                '  selection: "name CA"',
                '  align_selection: "protein and name CA"',
            ]
        )
    )

    cfg = load_config(cfg_path)
    assert cfg.system.topology is not None and cfg.system.topology.exists()
    assert len(cfg.system.trajectories) == 1 and cfg.system.trajectories[0].exists()
    assert cfg.rmsd.reference is not None and cfg.rmsd.reference.exists()
