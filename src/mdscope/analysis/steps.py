# Backward-compatibility shim: re-exports all public symbols from the refactored submodules.
# New code should import directly from mdscope.analysis or the relevant submodule.
from __future__ import annotations

from ._common import (
    TIEN2013_MAX_ASA,
    RunContext,
    _auto_hist_bins,
    _block_slices,
    _frame_slice,
    _imports,
    _load_universes,
    _plot_timeseries_and_distribution,
    _save_plot,
    _trajectory_names,
    ensure_dirs,
)
from .cluster import run_cluster
from .convergence import _jsd_1d, _jsd_2d, _jsd_from_prob, run_convergence
from .distance import run_distance
from .dssp import _dssp_precheck_diagnostics, _raise_dssp_precheck_error, run_dssp
from .pca import _collect_matrix, _collect_matrix_from_atom_indices, _plot_pca_free_energy_rt, run_pca
from .pocket import (
    _find_fpocket_info_file,
    _fpocket_available,
    _parse_fpocket_info,
    _pocket_collect_inputs,
    _run_fpocket_command,
    run_pocket,
)
from .ramachandran import run_ramachandran
from .representative import run_representative
from .rg import run_rg
from .rmsd import _compute_rmsd, _kabsch_fit, _write_atom_subset_pdb, _write_debug_pdb, run_rmsd
from .rmsf import run_rmsf
from .sasa import run_sasa
from .water import (
    _nearby_water_oxygen_indices,
    _plot_water_rdf,
    _plot_water_residence_distribution,
    _water_oxygen_atoms,
    _water_region_atoms,
    run_water,
)

STEP_HANDLERS = {
    "rmsd": run_rmsd,
    "rmsf": run_rmsf,
    "dssp": run_dssp,
    "pca": run_pca,
    "cluster": run_cluster,
    "representative": run_representative,
    "rg": run_rg,
    "sasa": run_sasa,
    "distance": run_distance,
    "ramachandran": run_ramachandran,
    "convergence": run_convergence,
    "pocket": run_pocket,
    "water": run_water,
}
