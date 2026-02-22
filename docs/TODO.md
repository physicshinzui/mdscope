# mdscope TODO

## Current Priorities
- [ ] Standardize `configs/dipeptide.yaml` as an official template
- [ ] Add template config locations and use-cases to `README`
  - `configs/dipeptide.yaml` (small peptide / ACE-X-NME template)
  - `configs/local_dual.yaml` (local dual-trajectory example)

## Convergence (JSD-based)

- [ ] Write a `convergence` implementation manual (docs)
  - Block splitting
  - JSD definition (`scipy.spatial.distance.jensenshannon`, `base=2`, divergence=`distance^2`)
  - Metric-specific distribution construction (RMSD/Rg/PCA/cluster occupancy)
  - `within` / `between` / `overall` decision logic
- [ ] Add a `convergence` tuning guide
  - `n_blocks`, `min_frames`
  - Initial `jsd_max` values and tuning strategy

## RMSD

- [ ] Document the `rmsd` implementation design
  - Custom residue mapping + MDAnalysis `rmsd()` function
  - Why high-level `MDAnalysis.analysis.rms.RMSD` is not used currently
- [ ] (Optional) Evaluate `rmsd.engine: custom | mdanalysis`
  - `custom`: supports residue mapping / `ligand_site`
  - `mdanalysis`: simpler path for straightforward systems

## Documentation

- [ ] Add an implementation manual
  - `docs/IMPLEMENTATION_MANUAL.md`
  - Focus on "what this code actually does"
- [ ] Add a reading guide for the `Cluster Occupancy by Block` figure to README/manual

## Packaging / Release

- [ ] Perform a license audit (dependencies + code provenance)
- [ ] Decide how to handle the `LICENSE` file until a license is selected
  - Current status: `README` says `TBD`, `pyproject.toml` license field removed
  - To verify: `LICENSE` file still exists in the repository

## Nice-to-have (Later)

- [ ] Evaluate a template-copy feature such as `init-config --template dipeptide`
- [ ] Evaluate a `list-templates` CLI command
- [ ] Split heavy real-data analyses into a separate optional GitHub Actions workflow
