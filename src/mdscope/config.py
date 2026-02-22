from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

StepName = Literal[
    "rmsd",
    "rmsf",
    "dssp",
    "pca",
    "cluster",
    "representative",
    "rg",
    "sasa",
    "distance",
    "ramachandran",
    "convergence",
]


class SystemConfig(BaseModel):
    topology: Path | None = None
    topologies: list[Path] | None = None
    trajectories: list[Path] = Field(min_length=1)
    trajectory_names: list[str] | None = None
    selection: str = "protein"
    align_selection: str = "backbone"

    @model_validator(mode="after")
    def validate_names(self) -> "SystemConfig":
        if self.trajectory_names and len(self.trajectory_names) != len(self.trajectories):
            raise ValueError("trajectory_names must match trajectories length")
        if self.topology is None and not self.topologies:
            raise ValueError("system.topology or system.topologies is required")
        if self.topologies and len(self.topologies) != len(self.trajectories):
            raise ValueError("system.topologies must match trajectories length")
        return self

    def expanded_topologies(self) -> list[Path]:
        if self.topologies:
            return list(self.topologies)
        if self.topology is None:
            raise ValueError("topology is not configured")
        return [self.topology for _ in self.trajectories]


class FramesConfig(BaseModel):
    start: int = 0
    stop: int | None = None
    step: int = 1


class OutputConfig(BaseModel):
    outdir: Path = Path("results")
    figure_formats: list[Literal["png", "pdf", "svg"]] = ["png"]
    dpi: int = 300

    @field_validator("dpi")
    @classmethod
    def validate_dpi(cls, value: int) -> int:
        if value < 72 or value > 1200:
            raise ValueError("dpi must be in [72, 1200]")
        return value


class RuntimeConfig(BaseModel):
    seed: int = 42
    resume: bool = False


class RmsdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: Path | None = None
    map_mode: Literal["strict", "align", "user_map"] = "align"
    map_file: Path | None = None
    selection: str = "name CA"
    align: bool = True
    align_selection: str = "protein and name CA"
    debug_write_aligned_pdb: bool = True
    debug_max_frames: int = 1
    region_mode: Literal["global", "ligand_site"] = "global"
    ligand_selection: str | None = None
    site_cutoff: float = 4.0
    min_mapped: int = 30
    min_coverage: float = 0.7
    export_below_threshold: bool = False
    threshold_angstrom: float = 1.5
    export_selection: str = "all"
    max_export_frames: int = 100


class RmsfConfig(BaseModel):
    selection: str = "protein and name CA"
    align: bool = True
    align_to: Literal["average", "first"] = "average"
    align_selection: str = "protein and backbone"


class PcaConfig(BaseModel):
    mode: Literal["project", "joint"] = "project"
    fit_trajectory: str | None = None
    align: bool = True
    selection: str = "backbone"
    n_components: int = 10
    use_pcs: list[int] = [1, 2, 3, 4, 5]
    reference_pdbs: list[Path] = []
    reference_names: list[str] = []
    site_from_reference_ligand: bool = False
    site_reference_pdb: Path | None = None
    site_ligand_selection: str | None = None
    site_cutoff: float = 4.0
    site_atom_selection: str = "name CA"
    site_align_selection: str = "protein and name CA"
    site_map_mode: Literal["strict", "align", "user_map"] = "align"
    site_map_file: Path | None = None
    free_energy_enabled: bool = True
    free_energy_bins: int | Literal["auto_fd"] = "auto_fd"
    free_energy_level_step_rt: float = 1.0
    free_energy_smooth_sigma: float = 2.0
    free_energy_per_trajectory: bool = False

    @field_validator("free_energy_bins")
    @classmethod
    def validate_free_energy_bins(cls, value: int | str) -> int | str:
        if isinstance(value, str):
            if value != "auto_fd":
                raise ValueError("pca.free_energy_bins must be an integer or 'auto_fd'")
            return value
        if value < 10 or value > 400:
            raise ValueError("pca.free_energy_bins must be in [10, 400]")
        return value

    @field_validator("free_energy_level_step_rt")
    @classmethod
    def validate_free_energy_level_step_rt(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pca.free_energy_level_step_rt must be > 0")
        return value

    @field_validator("free_energy_smooth_sigma")
    @classmethod
    def validate_free_energy_smooth_sigma(cls, value: float) -> float:
        if value < 0:
            raise ValueError("pca.free_energy_smooth_sigma must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_refs(self) -> "PcaConfig":
        if self.reference_names and len(self.reference_names) != len(self.reference_pdbs):
            raise ValueError("reference_names must match reference_pdbs length")
        if self.site_from_reference_ligand and not self.site_ligand_selection:
            raise ValueError("pca.site_ligand_selection is required when pca.site_from_reference_ligand=true")
        if self.site_map_mode == "user_map" and not self.site_map_file:
            raise ValueError("pca.site_map_file is required when pca.site_map_mode=user_map")
        return self


class ClusterConfig(BaseModel):
    method: Literal["hdbscan"] = "hdbscan"
    min_cluster_size: int = 100
    min_samples: int | None = None
    metric: str = "euclidean"
    selection_method: Literal["eom", "leaf"] = "eom"
    allow_single_cluster: bool = False
    representative_method: Literal["medoid", "centroid_nearest", "random"] = "medoid"
    representative_random_n: int = 1
    representative_random_seed: int | None = None
    representative_scope: Literal["global", "per_trajectory", "both"] = "both"


class SasaConfig(BaseModel):
    selection: str = "protein"
    level: Literal["atom", "residue"] = "residue"
    probe_radius: float = 1.4
    n_sphere_points: int = 960
    relative: bool = True
    reference_scale: Literal["tien2013"] = "tien2013"
    rsasa_clip: bool = False


class DsspConfig(BaseModel):
    selection: str = "protein"
    coil_code: str = "C"

    @field_validator("coil_code")
    @classmethod
    def validate_coil_code(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("dssp.coil_code must be a single character")
        return value


class PlotConfig(BaseModel):
    publication_style: bool = True
    timeseries_distribution: bool = True
    distribution_kind: Literal["hist", "kde", "hist_kde", "violin"] = "hist_kde"
    hist_bin_method: Literal["fd", "sturges", "sqrt", "auto"] = "fd"
    hist_bins: int | None = None


class DistanceItem(BaseModel):
    id: str
    sel1: str
    sel2: str


class DistanceConfig(BaseModel):
    pairs: list[DistanceItem] = []


class RamachandranConfig(BaseModel):
    mode: Literal["global", "per_residue", "both"] = "both"
    selection: str = "protein"
    residues: list[str] = []


class Convergence1DConfig(BaseModel):
    jsd_max: float = 0.08


class ConvergencePcaConfig(BaseModel):
    pcs: list[int] = [1, 2]
    jsd_max: float = 0.12


class ConvergenceClusterConfig(BaseModel):
    jsd_max: float = 0.15


class ConvergenceConfig(BaseModel):
    enabled_metrics: list[Literal["rmsd", "rg", "pca", "cluster_occupancy"]] = [
        "rmsd",
        "rg",
        "pca",
        "cluster_occupancy",
    ]
    n_blocks: int = 5
    min_frames: int = 500
    rule: Literal["all_of", "k_of_n"] = "k_of_n"
    k_required: int = 3
    rmsd: Convergence1DConfig = Convergence1DConfig()
    rg: Convergence1DConfig = Convergence1DConfig(jsd_max=0.06)
    pca: ConvergencePcaConfig = ConvergencePcaConfig()
    cluster_occupancy: ConvergenceClusterConfig = ConvergenceClusterConfig()


class AnalysesConfig(BaseModel):
    rmsd: bool = True
    rmsf: bool = True
    dssp: bool = True
    pca: bool = True
    cluster: bool = True
    representative: bool = True
    rg: bool = True
    sasa: bool = True
    distance: bool = False
    ramachandran: bool = False
    convergence: bool = False


class AppConfig(BaseModel):
    preset: Literal["standard", "full"] = "standard"
    system: SystemConfig
    frames: FramesConfig = FramesConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    output: OutputConfig = OutputConfig()
    analyses: AnalysesConfig = AnalysesConfig()
    rmsd: RmsdConfig = RmsdConfig()
    rmsf: RmsfConfig = RmsfConfig()
    pca: PcaConfig = PcaConfig()
    cluster: ClusterConfig = ClusterConfig()
    dssp: DsspConfig = DsspConfig()
    sasa: SasaConfig = SasaConfig()
    plot: PlotConfig = PlotConfig()
    distance: DistanceConfig = DistanceConfig()
    ramachandran: RamachandranConfig = RamachandranConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()

    @model_validator(mode="after")
    def validate_modes(self) -> "AppConfig":
        if self.pca.mode == "project" and not self.pca.fit_trajectory:
            names = self.system.trajectory_names or []
            self.pca.fit_trajectory = names[0] if names else "traj0"
        if self.rmsd.map_mode == "user_map" and not self.rmsd.map_file:
            raise ValueError("rmsd.map_file is required when rmsd.map_mode=user_map")
        if self.rmsd.region_mode == "ligand_site" and not self.rmsd.ligand_selection:
            raise ValueError("rmsd.ligand_selection is required when rmsd.region_mode=ligand_site")
        return self


PRESETS: dict[str, dict] = {
    "standard": {},
    "full": {
        "analyses": {
            "distance": True,
            "ramachandran": True,
            "convergence": True,
        }
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def apply_preset(config: dict) -> dict:
    preset = config.get("preset", "standard")
    preset_overrides = PRESETS.get(preset, {})
    return deep_merge(config, preset_overrides)


def load_config(config_path: Path) -> AppConfig:
    raw = load_yaml(config_path)
    merged = apply_preset(raw)
    try:
        cfg = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    base_dir = config_path.parent.resolve()
    if cfg.system.topology:
        cfg.system.topology = _resolve(base_dir, cfg.system.topology)
    if cfg.system.topologies:
        cfg.system.topologies = [_resolve(base_dir, p) for p in cfg.system.topologies]
    cfg.system.trajectories = [_resolve(base_dir, p) for p in cfg.system.trajectories]
    cfg.output.outdir = _resolve(base_dir, cfg.output.outdir)
    if cfg.rmsd.reference:
        cfg.rmsd.reference = _resolve(base_dir, cfg.rmsd.reference)
    if cfg.rmsd.map_file:
        cfg.rmsd.map_file = _resolve(base_dir, cfg.rmsd.map_file)
    cfg.pca.reference_pdbs = [_resolve(base_dir, p) for p in cfg.pca.reference_pdbs]
    if cfg.pca.site_reference_pdb:
        cfg.pca.site_reference_pdb = _resolve(base_dir, cfg.pca.site_reference_pdb)
    if cfg.pca.site_map_file:
        cfg.pca.site_map_file = _resolve(base_dir, cfg.pca.site_map_file)

    missing = []
    if cfg.system.topology and not cfg.system.topology.exists():
        missing.append(str(cfg.system.topology))
    for top in cfg.system.expanded_topologies():
        if not top.exists():
            missing.append(str(top))
    for trj in cfg.system.trajectories:
        if not trj.exists():
            missing.append(str(trj))
    if cfg.rmsd.reference and not cfg.rmsd.reference.exists():
        missing.append(str(cfg.rmsd.reference))
    for pdb in cfg.pca.reference_pdbs:
        if not pdb.exists():
            missing.append(str(pdb))
    if cfg.pca.site_reference_pdb and not cfg.pca.site_reference_pdb.exists():
        missing.append(str(cfg.pca.site_reference_pdb))
    if cfg.pca.site_map_file and not cfg.pca.site_map_file.exists():
        missing.append(str(cfg.pca.site_map_file))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Input files not found: {joined}")

    return cfg


def _resolve(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def generate_template(preset: str = "standard") -> str:
    if preset == "full":
        return _generate_full_template_with_comments()

    base = {
        "preset": preset,
        "system": {
            "topology": "../data/system/topology.pdb",
            "trajectories": ["../data/system/run1.xtc", "../data/system/run2.xtc"],
            "trajectory_names": ["run1", "run2"],
            "selection": "protein",
            "align_selection": "backbone",
        },
        "frames": {"start": 0, "stop": None, "step": 10},
        "runtime": {"seed": 42, "resume": False},
        "output": {"outdir": "results", "figure_formats": ["png", "pdf"], "dpi": 300},
        "plot": {
            "publication_style": True,
            "timeseries_distribution": True,
            "distribution_kind": "hist_kde",
            "hist_bin_method": "fd",
            "hist_bins": None,
        },
        "rmsd": {
            "reference": "../data/reference/reference.pdb",
            "map_mode": "align",
            "selection": "name CA",
            "align": True,
            "align_selection": "protein and name CA",
            "debug_write_aligned_pdb": True,
            "debug_max_frames": 1,
            "region_mode": "global",
            "ligand_selection": "not protein and not resname SOL and not resname HOH",
            "site_cutoff": 4.0,
            "export_below_threshold": False,
            "threshold_angstrom": 1.5,
            "export_selection": "all",
            "max_export_frames": 100,
        },
        "rmsf": {
            "selection": "protein and name CA",
            "align": True,
            "align_to": "average",
            "align_selection": "protein and backbone",
        },
        "pca": {
            "mode": "project",
            "fit_trajectory": "run1",
            "align": True,
            "n_components": 10,
            "use_pcs": [1, 2, 3, 4, 5],
            "reference_pdbs": ["../data/reference/reference.pdb"],
            "reference_names": ["reference_1"],
            "free_energy_enabled": True,
            "free_energy_bins": "auto_fd",
            "free_energy_level_step_rt": 1.0,
            "free_energy_smooth_sigma": 2.0,
            "free_energy_per_trajectory": False,
            "site_from_reference_ligand": False,
            "site_reference_pdb": "../data/reference/reference.pdb",
            "site_ligand_selection": "resname LIG",
            "site_cutoff": 5.0,
            "site_atom_selection": "name CA",
            "site_align_selection": "protein and name CA",
            "site_map_mode": "align",
        },
        "cluster": {
            "method": "hdbscan",
            "min_cluster_size": 100,
            "representative_method": "medoid",
            "representative_random_n": 1,
            "representative_random_seed": 42,
        },
        "sasa": {
            "selection": "protein",
            "level": "residue",
            "probe_radius": 1.4,
            "relative": True,
            "reference_scale": "tien2013",
            "rsasa_clip": False,
        },
        "dssp": {"selection": "protein", "coil_code": "C"},
        "distance": {
            "pairs": [
                {
                    "id": "pair_1",
                    "sel1": "segid A and resid 10 and name CA",
                    "sel2": "segid A and resid 20 and name CA",
                }
            ],
        },
        "ramachandran": {"mode": "both", "selection": "protein", "residues": ["A:10", "A:20"]},
        "convergence": {
            "enabled_metrics": ["rmsd", "rg", "pca", "cluster_occupancy"],
            "n_blocks": 5,
            "min_frames": 500,
            "rule": "k_of_n",
            "k_required": 3,
            "rmsd": {"jsd_max": 0.08},
            "rg": {"jsd_max": 0.06},
            "pca": {"pcs": [1, 2], "jsd_max": 0.12},
            "cluster_occupancy": {"jsd_max": 0.15},
        },
    }
    merged = apply_preset(base)
    return yaml.safe_dump(merged, sort_keys=False, allow_unicode=False)


def _generate_full_template_with_comments() -> str:
    return """# mdscope full preset template
# Tip: execution on/off is controlled only by `analyses.*`
# Tip: values shown here are examples; adjust selections and thresholds to your system.
preset: full

system:
  # Shared topology used for all trajectories (choose this OR `topologies` below).
  topology: ../data/system/topology.pdb
  # Per-trajectory topologies (same length/order as `trajectories`).
  # topologies:
  #   - ../data/system/run1_topology.pdb
  #   - ../data/system/run2_topology.pdb
  topologies: null
  # One or more trajectory files (XTC/DCD/...); order must match trajectory_names/topologies.
  trajectories:
    - ../data/system/run1.xtc
    - ../data/system/run2.xtc
  # Display names used in tables/plots.
  trajectory_names: [run1, run2]
  # Default atom selection used by analyses that rely on `system.selection` (currently mainly Rg).
  selection: protein
  # Default alignment selection used by analyses that rely on `system.align_selection` (mainly PCA pre-alignment).
  align_selection: backbone

frames:
  # Frame slicing applied to most analyses (Python slice semantics: start:stop:step).
  start: 0
  # null means "until the end of the trajectory".
  stop: null
  # Use >1 for faster exploratory runs.
  step: 10

runtime:
  # Random seed used where stochastic steps exist (e.g., random representative sampling).
  seed: 42
  # If true, skip steps with matching .state/*.done checkpoints.
  resume: false

output:
  # Base output directory for all generated files.
  outdir: results
  # Multi-format figure export. png uses `dpi`; vector formats ignore dpi.
  figure_formats: [png, pdf]
  # Raster DPI for PNG export.
  dpi: 300

analyses:
  # Master on/off switches for pipeline steps. Dependencies may still run when required.
  rmsd: true
  rmsf: true
  dssp: true
  pca: true
  cluster: true
  representative: true
  rg: true
  sasa: true
  distance: true
  ramachandran: true
  convergence: true

plot:
  # Apply shared "publication-like" matplotlib style.
  publication_style: true
  # If true, also save distribution plots for time-series metrics (RMSD/Rg/SASA etc.).
  timeseries_distribution: true
  # Distribution panel style: hist | kde | hist_kde | violin (where implemented).
  distribution_kind: hist_kde
  # Histogram binning rule when hist_bins is null: fd | sturges | sqrt | auto.
  hist_bin_method: fd
  # Override histogram bin count (null = auto by hist_bin_method).
  hist_bins: null

rmsd:
  # External reference structure (PDB/MAE readable by MDAnalysis). If null, first trajectory topology is used.
  reference: ../data/reference/reference.pdb
  # Residue mapping strategy between trajectory and reference: strict | align | user_map.
  map_mode: align
  map_file: null  # required when map_mode: user_map
  # Atom selection used for RMSD atom pairing and RMSD value calculation.
  selection: name CA
  # If true, superpose mobile coordinates onto reference before RMSD calculation.
  align: true
  # Selection used for residue mapping / alignment logic (not necessarily the same as `selection`).
  align_selection: protein and name CA
  # Debug mode: write paired reference/mobile-aligned PDBs for visual inspection.
  debug_write_aligned_pdb: true
  # Max number of frames per trajectory for debug aligned PDB export.
  debug_max_frames: 1
  # global = use mapped region globally; ligand_site = restrict to residues near reference ligand.
  region_mode: global  # global | ligand_site
  # Ligand selection in the reference structure (used when region_mode=ligand_site).
  ligand_selection: not protein and not resname SOL and not resname HOH
  # Distance cutoff (angstrom) for defining ligand-site residues in the reference.
  site_cutoff: 4.0
  # Minimum number of mapped residues required to proceed.
  min_mapped: 30
  # Minimum mapping coverage ratio (definition depends on region_mode).
  min_coverage: 0.7
  # Export frames below RMSD threshold as PDB:
  export_below_threshold: false
  # RMSD threshold in angstrom for export_below_threshold.
  threshold_angstrom: 1.5
  # Atom selection written to exported PDBs (`all` to export whole frame).
  export_selection: all
  # Per-trajectory cap on exported frames.
  max_export_frames: 100

rmsf:
  # Atom selection for RMSF values (commonly CA atoms).
  selection: protein and name CA
  # If true, align trajectory before RMSF.
  align: true
  # Alignment reference for RMSF: average structure (typical) or first frame.
  align_to: average  # average | first
  # Atom selection used for RMSF alignment.
  align_selection: protein and backbone

pca:
  # project = fit PCA on one trajectory then project others; joint = fit on concatenated trajectories.
  mode: project  # project | joint
  # Trajectory name used as PCA fit source in project mode; also alignment reference trajectory when pca.align=true.
  fit_trajectory: run1
  # Align trajectories before collecting PCA coordinates.
  align: true
  # Atom selection flattened into PCA input vectors.
  selection: backbone
  # Maximum principal components to compute.
  n_components: 10
  # Components used downstream for clustering (1-based indexing in config).
  use_pcs: [1, 2, 3, 4, 5]
  # Optional static structures to project into PCA space and mark on plots.
  reference_pdbs: [../data/reference/reference.pdb]
  # Labels for `reference_pdbs` (same length and order).
  reference_names: [reference_1]
  # PC1-PC2 free energy (RT units):
  free_energy_enabled: true
  # 2D histogram bins for free-energy map; use auto_fd for Freedman-Diaconis binning.
  free_energy_bins: auto_fd  # auto_fd or integer
  # Contour interval in RT units.
  free_energy_level_step_rt: 1.0
  # Gaussian smoothing sigma in histogram-bin units (0 = no smoothing).
  free_energy_smooth_sigma: 2.0
  # If true, also write per-trajectory free-energy maps in addition to combined map.
  free_energy_per_trajectory: false
  # Ligand-site PCA (optional):
  # If true, define PCA atom set from residues near a ligand in a reference structure, then map to each trajectory.
  site_from_reference_ligand: false
  # Reference PDB used to define ligand-site residues for PCA.
  site_reference_pdb: ../data/reference/reference.pdb
  # Ligand selection in the PCA site reference.
  site_ligand_selection: resname LIG
  # Ligand-site residue cutoff in angstrom.
  site_cutoff: 5.0
  # Atom selection within mapped site residues used as PCA coordinates.
  site_atom_selection: name CA
  # Selection used for trajectory alignment in site-PCA mode.
  site_align_selection: protein and name CA
  site_map_mode: align  # strict | align | user_map
  site_map_file: null  # required when site_map_mode: user_map

cluster:
  # Clustering is performed on PCA scores (selected PCs via pca.use_pcs).
  method: hdbscan
  # HDBSCAN core hyperparameter: minimum cluster size.
  min_cluster_size: 100
  # HDBSCAN min_samples (null = library default / fallback behavior in code).
  min_samples: null
  # Distance metric in PCA score space.
  metric: euclidean
  # HDBSCAN cluster selection method.
  selection_method: eom
  # If true, allow a single cluster result instead of all-noise in weakly separated data.
  allow_single_cluster: false
  # Representative extraction strategy for each cluster.
  representative_method: medoid  # medoid | centroid_nearest | random
  # Number of random samples per cluster when representative_method=random.
  representative_random_n: 1
  # RNG seed for random representative sampling.
  representative_random_seed: 42
  # Output representative structures globally / per trajectory / both.
  representative_scope: both  # global | per_trajectory | both

sasa:
  # Atom selection for SASA calculation.
  selection: protein
  # Output granularity: atom | residue.
  level: residue
  # Solvent probe radius in angstrom (water ~1.4 A).
  probe_radius: 1.4
  # Surface point density used by mdakit-sasa (higher = slower, smoother).
  n_sphere_points: 960
  # If true, report relative SASA (rSASA) using reference_scale.
  relative: true
  # Reference maximum ASA scale for rSASA normalization.
  reference_scale: tien2013
  # If true, clip rSASA values to [0, 1].
  rsasa_clip: false

dssp:
  # Selection passed to MDAnalysis DSSP (typically protein only).
  selection: protein
  # Symbol used for coil/other state in 3-state collapsed output.
  coil_code: C

distance:
  # Empty list is allowed.
  pairs:
    - id: pair_1
      # Any valid MDAnalysis selection strings are allowed.
      sel1: segid A and resid 10 and name CA
      sel2: segid A and resid 20 and name CA

ramachandran:
  # global = combined plot, per_residue = one plot per residue, both = both outputs.
  mode: both  # global | per_residue | both
  # Residue selection used for phi/psi extraction (usually protein).
  selection: protein
  # Optional explicit subset for per-residue outputs; format examples: A:10, B:20.
  residues: [A:10, A:20]

convergence:
  # Metrics used for convergence decision. Inputs are read from existing output CSVs.
  enabled_metrics: [rmsd, rg, pca, cluster_occupancy]
  # Number of time blocks used for within-trajectory stability checks.
  n_blocks: 5
  # Minimum frames required per trajectory per metric; shorter series fail that metric.
  min_frames: 500
  # all_of = all enabled metrics must pass; k_of_n = at least k_required metrics must pass.
  rule: k_of_n  # all_of | k_of_n
  # Used only when rule=k_of_n.
  k_required: 3
  rmsd:
    # JSD divergence threshold (base=2, computed as distance^2).
    jsd_max: 0.08
  rg:
    # JSD divergence threshold for Rg distribution stability/agreement.
    jsd_max: 0.06
  pca:
    # PCA components (1-based) used for 2D JSD convergence check (first two values are used as x/y).
    pcs: [1, 2]
    # JSD divergence threshold for PC-space 2D histogram distributions.
    jsd_max: 0.12
  cluster_occupancy:
    # JSD divergence threshold for HDBSCAN cluster occupancy vectors.
    jsd_max: 0.15
"""
