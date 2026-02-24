import numpy as np

from mdscope.analysis.pca import _pairwise_distance_vector
from mdscope.config import AppConfig


def test_pairwise_distance_vector_upper_triangle_order() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    vec = _pairwise_distance_vector(pts)

    # Order follows np.triu_indices(k=1): (0,1), (0,2), (1,2)
    assert np.allclose(vec, [1.0, 2.0, np.sqrt(5.0)])


def test_pca_config_accepts_distance_feature_mode(tmp_path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
            },
            "pca": {"feature_mode": "distance"},
        }
    )

    assert cfg.pca.feature_mode == "distance"
