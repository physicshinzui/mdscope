from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

from mdscope.analysis._common import RunContext
from mdscope.analysis.contacts import _bh_fdr, run_contacts
from mdscope.config import AppConfig


def _contact_universe(has_interchain_contact: bool, mutated: bool = False):
    universe = mda.Universe.empty(
        8,
        n_residues=4,
        atom_resindex=[0, 0, 1, 1, 2, 2, 3, 3],
        residue_segindex=[0, 0, 0, 0],
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["CA", "CB"] * 4)
    universe.add_TopologyAttr("types", ["C"] * 8)
    universe.add_TopologyAttr("masses", [12.0] * 8)
    resnames = ["ALA"] * 4
    if mutated:
        resnames[1] = "CYS"
    universe.add_TopologyAttr("resnames", resnames)
    universe.add_TopologyAttr("resids", [1, 2, 3, 4])
    universe.add_TopologyAttr("segids", ["PROT"])
    universe.add_TopologyAttr(
        "bonds",
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
    )
    chain_two_x = 2.5 if has_interchain_contact else 30.0
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [chain_two_x, 0.0, 0.0],
            [chain_two_x + 0.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    universe.load_new(np.stack([coordinates, coordinates]), order="fac")
    return universe


def test_bh_fdr_is_monotonic_in_rank() -> None:
    p_values = np.asarray([0.04, 0.001, 0.03, 0.2])
    q_values = _bh_fdr(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(q_values[order]) >= -1e-12)
    assert np.all((q_values >= 0.0) & (q_values <= 1.0))


def test_run_contacts_compares_independent_trajectories(tmp_path: Path, monkeypatch) -> None:
    top = tmp_path / "top.pdb"
    trj_a = tmp_path / "a.xtc"
    trj_b = tmp_path / "b.xtc"
    for path in (top, trj_a, trj_b):
        path.write_text("placeholder")

    outdir = tmp_path / "results"
    cfg = AppConfig.model_validate(
        {
            "system": {
                "topology": str(top),
                "trajectories": [str(trj_a), str(trj_b)],
                "trajectory_names": ["WT_r01", "MUT_r01"],
            },
            "output": {"outdir": str(outdir), "figure_formats": ["png"]},
            "analyses": {"contacts": True},
            "contacts": {
                "selection": "protein",
                "atom_selection": "not name H*",
                "cutoff_angstrom": 4.5,
                "min_sequence_separation": 3,
                "minimum_mean_occupancy": 0.0,
                "residue_number_offsets": {1: 10, 2: 20},
                "residue_numbering_name": "DMS",
                "comparison": {
                    "group_a": "WT",
                    "trajectories_a": ["WT_r01"],
                    "group_b": "MUT",
                    "trajectories_b": ["MUT_r01"],
                },
            },
        }
    )

    monkeypatch.setattr(
        "mdscope.analysis.contacts._load_universes",
        lambda _cfg: [
            ("WT_r01", _contact_universe(False)),
            ("MUT_r01", _contact_universe(True, mutated=True)),
        ],
    )
    run_contacts(RunContext(config=cfg, outdir=outdir, cache={}))

    ranking = pd.read_csv(outdir / "tables" / "contact_ranking.csv")
    assert len(ranking) >= 1
    assert ranking.iloc[0]["difference_b_minus_a"] == 1.0
    assert ranking.iloc[0]["residue_i_chain"] != ranking.iloc[0]["residue_j_chain"]
    assert ranking.iloc[0]["residue_numbering"] == "DMS"
    assert "residue_i_display_resid" in ranking.columns
    assert "contact_label_topology_a" in ranking.columns
    assert (outdir / "tables" / "contact_residue_hotspots.csv").exists()
    assert (outdir / "figures" / "contact_map_difference_MUT_minus_WT.png").exists()
    assert (outdir / "figures" / "contact_top_differences.png").exists()
