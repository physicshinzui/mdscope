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
    "ligand_site",
    "distance",
    "ramachandran",
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

    enabled: bool = True
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


class RmsfConfig(BaseModel):
    enabled: bool = True
    selection: str = "protein and name CA"
    align: bool = True
    align_to: Literal["average", "first"] = "average"
    align_selection: str = "protein and backbone"


class PcaConfig(BaseModel):
    enabled: bool = True
    mode: Literal["project", "joint"] = "project"
    fit_trajectory: str | None = None
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
    free_energy_level_step_rt: float = 0.2
    free_energy_smooth_sigma: float = 1.0
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
    enabled: bool = True
    method: Literal["hdbscan"] = "hdbscan"
    min_cluster_size: int = 100
    min_samples: int | None = None
    metric: str = "euclidean"
    selection_method: Literal["eom", "leaf"] = "eom"
    allow_single_cluster: bool = False
    representative_method: Literal["medoid", "centroid_nearest"] = "medoid"
    representative_scope: Literal["global", "per_trajectory", "both"] = "both"


class SasaConfig(BaseModel):
    enabled: bool = True
    selection: str = "protein"
    level: Literal["atom", "residue"] = "residue"
    probe_radius: float = 1.4
    n_sphere_points: int = 960


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
    enabled: bool = False
    pairs: list[DistanceItem] = []


class RamachandranConfig(BaseModel):
    mode: Literal["global", "per_residue", "both"] = "both"
    selection: str = "protein"
    residues: list[str] = []


class AnalysesConfig(BaseModel):
    rmsd: bool = True
    rmsf: bool = True
    dssp: bool = True
    pca: bool = True
    cluster: bool = True
    representative: bool = True
    rg: bool = True
    sasa: bool = True
    ligand_site: bool = True
    distance: bool = False
    ramachandran: bool = False


class AppConfig(BaseModel):
    preset: Literal["quick", "standard", "full"] = "standard"
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
    "quick": {
        "analyses": {
            "dssp": False,
            "cluster": False,
            "representative": False,
            "ligand_site": False,
            "ramachandran": False,
        }
    },
    "standard": {},
    "full": {
        "analyses": {
            "distance": True,
            "ramachandran": True,
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
    base = {
        "preset": preset,
        "system": {
            "topology": "../data/WT/topology.pdb",
            "trajectories": ["../data/WT/samples.xtc", "../data/F143W/samples.xtc"],
            "trajectory_names": ["WT", "F143W"],
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
            "enabled": True,
            "reference": "../data/pdbs/2YXJ.pdb",
            "map_mode": "align",
            "selection": "name CA",
            "align": True,
            "align_selection": "protein and name CA",
            "debug_write_aligned_pdb": True,
            "debug_max_frames": 1,
            "region_mode": "global",
            "ligand_selection": "not protein and not resname SOL and not resname HOH",
            "site_cutoff": 4.0,
        },
        "rmsf": {
            "enabled": True,
            "selection": "protein and name CA",
            "align": True,
            "align_to": "average",
            "align_selection": "protein and backbone",
        },
        "pca": {
            "enabled": True,
            "mode": "project",
            "fit_trajectory": "WT",
            "n_components": 10,
            "use_pcs": [1, 2, 3, 4, 5],
            "reference_pdbs": ["../data/pdbs/2YXJ.pdb"],
            "reference_names": ["xtal_2YXJ"],
            "free_energy_enabled": True,
            "free_energy_bins": "auto_fd",
            "free_energy_level_step_rt": 0.2,
            "free_energy_smooth_sigma": 1.0,
            "free_energy_per_trajectory": False,
            "site_from_reference_ligand": False,
            "site_reference_pdb": "../data/pdbs/2YXJ.pdb",
            "site_ligand_selection": "resname N3C",
            "site_cutoff": 5.0,
            "site_atom_selection": "name CA",
            "site_align_selection": "protein and name CA",
            "site_map_mode": "align",
        },
        "cluster": {"enabled": True, "method": "hdbscan", "min_cluster_size": 100, "representative_method": "medoid"},
        "sasa": {"enabled": True, "selection": "protein", "level": "residue", "probe_radius": 1.4},
        "dssp": {"selection": "protein", "coil_code": "C"},
        "distance": {
            "enabled": False,
            "pairs": [
                {
                    "id": "active_site",
                    "sel1": "segid A and resid 45 and name CA",
                    "sel2": "segid A and resid 143 and name CA",
                }
            ],
        },
        "ramachandran": {"mode": "both", "selection": "protein", "residues": ["A:45", "A:143"]},
    }
    merged = apply_preset(base)
    return yaml.safe_dump(merged, sort_keys=False, allow_unicode=False)
