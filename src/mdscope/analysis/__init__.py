from __future__ import annotations

from ._common import RunContext
from .rmsd import run_rmsd
from .rmsf import run_rmsf
from .rg import run_rg
from .water import run_water
from .pca import run_pca
from .cluster import run_cluster
from .representative import run_representative
from .distance import run_distance
from .ramachandran import run_ramachandran
from .convergence import run_convergence
from .dssp import run_dssp
from .sasa import run_sasa
from .pocket import run_pocket

STEP_HANDLERS = {
    "rmsd": run_rmsd, "rmsf": run_rmsf, "dssp": run_dssp,
    "pca": run_pca, "cluster": run_cluster, "representative": run_representative,
    "rg": run_rg, "sasa": run_sasa, "distance": run_distance,
    "ramachandran": run_ramachandran, "convergence": run_convergence,
    "pocket": run_pocket, "water": run_water,
}
__all__ = ["STEP_HANDLERS", "RunContext"]
