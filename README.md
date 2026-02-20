# mdscope

Config-first MD trajectory analysis automation tool built on MDAnalysis.

## What It Does

- Batch analysis of one or more trajectories using a single YAML config
- Resume/restart-safe execution by step (`.state` checkpoints)
- Supported steps:
  - `rmsd`
  - `rmsf`
  - `dssp`
  - `pca`
  - `cluster` (HDBSCAN)
  - `representative`
  - `rg`
  - `sasa` (optional dependency)
  - `distance`
  - `ramachandran`
- Publication-oriented plotting defaults (multi-format + DPI control)

## Installation

### pip

```bash
pip install -e .
```

SASA support:

```bash
pip install -e '.[sasa]'
```

### conda (example)

```bash
conda create -n mdscope python=3.11 -y
conda activate mdscope
pip install -e '.[dev,sasa]'
```

## Quick Start

1. Create a config template:

```bash
mdscope init-config --preset standard -o config.yaml
```

2. Validate:

```bash
mdscope validate-config -c config.yaml
```

3. Run:

```bash
mdscope run -c config.yaml --resume
```

Local example in this repo:

```bash
mdscope validate-config -c configs/local_dual.yaml
mdscope run -c configs/local_dual.yaml --only rmsd pca cluster representative rg distance dssp
```

## Core CLI

```bash
mdscope list-presets
mdscope doctor
mdscope run --help
```

Common run patterns:

```bash
# Re-run specific steps only
mdscope run -c config.yaml --only pca cluster representative --force-step pca --force-step cluster

# Resume unfinished pipeline
mdscope run -c config.yaml --resume

# Dry-run config/selection
mdscope run -c config.yaml --only rmsd pca --dry-run
```

`--only` supports both styles:

```bash
mdscope run -c config.yaml --only rmsd pca cluster
mdscope run -c config.yaml --only rmsd pca cluster representative
```

## Input Model

`system` supports:

- `topology`: one topology shared by all trajectories
- `topologies`: one topology per trajectory (same length as `trajectories`)

Example:

```yaml
system:
  topologies:
    - ../data/WT/topology.pdb
    - ../data/F143W/topology.pdb
  trajectories:
    - ../data/WT/samples.xtc
    - ../data/F143W/samples.xtc
  trajectory_names: [WT, F143W]
```

## Analysis Notes

### RMSD

- `rmsd.align_selection`: residues used for mapping/alignment logic
- `rmsd.selection`: atom set used for RMSD pair matching
- `rmsd.region_mode: ligand_site` + `rmsd.ligand_selection`: ligand-site-focused RMSD
- Debug PDB outputs are available for alignment inspection

### RMSF

- Default behavior aligns to average structure (`rmsf.align_to: average`)

### DSSP

- Controlled by `analyses.dssp` and `dssp.*` options (`selection`, `coil_code`)
- Outputs:
  - `tables/dssp_fraction_timeseries.csv`
  - `tables/dssp_fraction_per_residue.csv`
  - `figures/dssp_fraction_timeseries_<trajectory>.(png|pdf|svg)`
  - `figures/dssp_fraction_per_residue_<trajectory>.(png|pdf|svg)`
  - `figures/dssp_heatmap_residue_time_<trajectory>.(png|pdf|svg)`
- 3-state normalization:
  - Helix: `H/G/I` -> `H`
  - Sheet: `E/B` -> `S`
  - Others -> `C` (or configured `coil_code`)

### PCA + Free Energy Landscape

- PCA supports:
  - `mode: project` (fit on one trajectory, project others)
  - `mode: joint` (fit on concatenated trajectories)
- Optional reference PDB projection points are plotted as `x`
- Free energy landscape (PC1-PC2):
  - `pca.free_energy_enabled: true`
  - `pca.free_energy_bins: auto_fd` or integer
  - `pca.free_energy_level_step_rt` (contour interval in RT units)
  - `pca.free_energy_smooth_sigma` (Gaussian smoothing in bin units)
  - Colorbar range fixed to `0.0-10.0 RT`

CLI overrides:

```bash
mdscope run -c config.yaml --only pca --force-step pca \
  --free-energy --fe-bins-auto-fd --fe-level-step-rt 0.5 --fe-smooth-sigma 1.0
```

### Clustering / Representatives

- HDBSCAN runs on selected PCA components (`pca.use_pcs`)
- `hdbscan_labels.csv` includes both numeric `label` and display `cluster_name`
- `cluster_name` is trajectory-oriented (dominant trajectory in cluster)
- Representative structures are exported as PDB per cluster

## Output Structure

```
<outdir>/
  tables/
  figures/
  data/
  representatives/
  .state/
```

## Plotting Defaults

- `plot.publication_style: true` enables common publication-style rcParams
- Multi-format export via `output.figure_formats` (`png`, `pdf`, `svg`)
- DPI control via `output.dpi`
- Recommended aspect usage:
  - Time series / RMSF: horizontal
  - 2D maps (PCA/FEL/Ramachandran): square

## Validation and Troubleshooting

```bash
mdscope validate-config -c config.yaml
mdscope doctor
```

If a step fails:

- Re-run from checkpoint:
  - `mdscope run -c config.yaml --resume`
- Re-run one step:
  - `mdscope run -c config.yaml --only pca --force-step pca`

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

## Data and Git Policy

- Keep large MD raw data outside this repository.
- Commit configs/scripts/tests; do not commit heavy trajectories.
- Recommended: ignore generated outputs (already covered by `.gitignore`).

## License

MIT
