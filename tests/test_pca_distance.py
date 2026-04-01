import numpy as np

from mdscope.analysis._common import ensure_dirs
from mdscope.analysis.pca import (
    _pairwise_distance_vector,
    _pymol_selection_from_atoms,
    _write_pca_pymol_outputs,
    _write_pca_reference_pymol_outputs,
)
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


class _FakeAtom:
    def __init__(self, resid: int, resname: str, name: str, segid: str = "", chainID: str = "") -> None:
        self.resid = resid
        self.resname = resname
        self.name = name
        self.segid = segid
        self.chainID = chainID


def test_pymol_selection_from_atoms_groups_by_residue() -> None:
    atoms = [
        _FakeAtom(10, "GLY", "CA", chainID="A"),
        _FakeAtom(10, "GLY", "N", chainID="A"),
        _FakeAtom(11, "ALA", "CA", chainID="A"),
    ]

    selection = _pymol_selection_from_atoms(atoms)

    assert "(chain A and resi 10 and resn GLY and name CA+N)" in selection
    assert "(chain A and resi 11 and resn ALA and name CA)" in selection


def test_write_pca_pymol_outputs_writes_pml_and_json(tmp_path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run-1"],
            },
            "output": {"outdir": str(tmp_path / "results")},
        }
    )
    dirs = ensure_dirs(cfg.output.outdir)

    _write_pca_pymol_outputs(
        dirs,
        cfg,
        [
            {
                "trajectory": "run-1",
                "selection_mode": "direct_selection",
                "atom_count": 2,
                "source_selection": "backbone",
                "pymol_selection": "(chain A and resi 10 and resn GLY and name CA+N)",
            }
        ],
    )

    pml_text = (dirs["data"] / "pca_atom_selections.pml").read_text()
    report = (dirs["data"] / "pca_atom_selection_report.json").read_text()

    assert "select pca_atoms_run_1, (chain A and resi 10 and resn GLY and name CA+N)" in pml_text
    assert '"command_file": "pca_atom_selections.pml"' in report
    assert '"pymol_object_name": "pca_atoms_run_1"' in report


def test_write_pca_reference_pymol_outputs_writes_projected_and_skipped_json(tmp_path) -> None:
    top = tmp_path / "top.pdb"
    trj = tmp_path / "traj.xtc"
    top.write_text("x")
    trj.write_text("x")

    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj)],
                "trajectory_names": ["run1"],
            },
            "output": {"outdir": str(tmp_path / "results")},
        }
    )
    dirs = ensure_dirs(cfg.output.outdir)

    _write_pca_reference_pymol_outputs(
        dirs,
        cfg,
        [
            {
                "reference_name": "ref-1",
                "reference_pdb": "ref1.pdb",
                "selection_mode": "direct_selection",
                "atom_count": 2,
                "mapping_strategy": "direct_selection",
                "source_selection": "backbone",
                "pymol_selection": "(chain A and resi 10 and resn GLY and name CA+N)",
                "status": "projected",
            },
            {
                "reference_name": "ref-2",
                "reference_pdb": "ref2.pdb",
                "selection_mode": "site_from_reference_ligand",
                "atom_count": 1,
                "mapping_strategy": "align",
                "missing_site_atoms": 2,
                "source_selection": {"site_atom_selection": "name CA"},
                "status": "skipped",
            },
        ],
    )

    pml_text = (dirs["data"] / "pca_reference_atom_selections.pml").read_text()
    report = (dirs["data"] / "pca_reference_atom_selection_report.json").read_text()

    assert "select pca_ref_atoms_ref_1, (chain A and resi 10 and resn GLY and name CA+N)" in pml_text
    assert "pca_ref_atoms_ref_2" not in pml_text
    assert '"command_file": "pca_reference_atom_selections.pml"' in report
    assert '"pymol_object_name": "pca_ref_atoms_ref_1"' in report
    assert '"status": "skipped"' in report
