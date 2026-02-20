from pathlib import Path

from mdscope.config import load_config


def test_local_data_config_resolves() -> None:
    cfg_path = Path("configs/local_dual.yaml")
    cfg = load_config(cfg_path)
    expanded = cfg.system.expanded_topologies()
    assert len(expanded) == 2
    assert all(p.exists() for p in expanded)
    assert len(cfg.system.trajectories) == 2
    assert cfg.rmsd.reference is not None
    assert cfg.rmsd.reference.exists()
