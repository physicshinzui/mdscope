# mdscope

Config-first MD trajectory analysis automation tool.

## Install

```bash
pip install -e .
```

SASA extra:

```bash
pip install -e '.[sasa]'
```

## Quick start

1. Create a template config:

```bash
mdscope init-config --preset standard -o config.yaml
```

2. Validate config:

```bash
mdscope validate-config -c config.yaml
```

3. Run pipeline:

```bash
mdscope run -c config.yaml --resume
```

`system` input supports either one topology for all trajectories (`topology`) or one topology per trajectory (`topologies`).

Local dataset in this repository:

```bash
mdscope validate-config -c configs/local_dual.yaml
mdscope run -c configs/local_dual.yaml --only rmsd pca cluster representative rg distance
```

## User-friendly design

- Config-first: one command for daily use (`mdscope run -c config.yaml`)
- Minimal runtime overrides: `--outdir`, `--resume`, `--only`, `--force-step`
- Restart support with step checkpoints in `results/.state/`
- Optional multi-format figures (`png/pdf/svg`) and configurable `dpi`
- Timeseries analyses can emit both timeseries and distribution plots
- Distribution histogram bins can be auto-estimated from data (default: Freedman-Diaconis)

## Commands

- `mdscope init-config`
- `mdscope validate-config`
- `mdscope list-presets`
- `mdscope doctor`
- `mdscope run`

## RMSD selection control

`rmsd.align_selection` controls residue mapping, while `rmsd.selection` controls atoms used for RMSD.
Set `rmsd.region_mode: ligand_site` with `rmsd.ligand_selection` to compute RMSD only around ligand-contact residues in the reference structure.

RMSF is computed on aligned coordinates by default (`rmsf.align: true`, `rmsf.align_to: average`).

PCA can optionally use ligand-site residues defined on a reference structure (`pca.site_from_reference_ligand: true`).
