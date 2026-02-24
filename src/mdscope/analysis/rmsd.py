from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._common import RunContext, _frame_slice, _imports, _load_universes, _plot_timeseries_and_distribution, ensure_dirs
from .mapping import build_residue_mapping, filter_mapping_to_reference_resindices, ligand_site_resindices, residue_records


def _kabsch_fit(mobile: Any, reference: Any) -> Any:
    _, np, _, _ = _imports()
    from MDAnalysis.analysis import align

    mob_center = mobile.mean(axis=0)
    ref_center = reference.mean(axis=0)
    m0 = mobile - mob_center
    r0 = reference - ref_center
    rot, _ = align.rotation_matrix(m0, r0)
    return m0 @ rot + ref_center


def _compute_rmsd(
    mobile_coords: Any,
    ref_coords: Any,
    superpose: bool,
) -> tuple[float, int]:
    from MDAnalysis.analysis.rms import rmsd as mda_rmsd

    m = mobile_coords
    r = ref_coords
    if len(m) != len(r):
        raise RuntimeError("mobile/ref atom count mismatch in RMSD calculation")
    if len(m) == 0:
        raise RuntimeError("no atoms available for RMSD")
    value = float(mda_rmsd(m, r, center=superpose, superposition=superpose))
    return value, len(m)


def _write_atom_subset_pdb(universe: Any, atom_indices: list[int], out_path: Path) -> None:
    if not atom_indices:
        return
    unique = sorted(set(int(i) for i in atom_indices))
    atoms = universe.atoms[unique]
    if len(atoms) == 0:
        return
    atoms.write(str(out_path))


def _write_debug_pdb(atoms: Any, coords: Any, out_path: Path, pairwise_ids: bool = False) -> None:
    lines = []
    for i, (atom, xyz) in enumerate(zip(atoms, coords), start=1):
        name = f"{atom.name:>4}"[:4]
        if pairwise_ids:
            resname = "MAP"
            chain = "A"
            resid = i
        else:
            resname = f"{atom.resname:>3}"[:3]
            chain = (str(atom.segid).strip()[:1] or "A")
            resid = int(atom.resid)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        element = (getattr(atom, "element", "") or atom.name[:1] or "C").strip()[:2].rjust(2)
        lines.append(
            f"ATOM  {i:5d} {name} {resname} {chain}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}"
        )
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def run_rmsd(ctx: RunContext) -> None:
    mda, _, pd, _ = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    universes = _load_universes(cfg)
    reference = mda.Universe(str(cfg.rmsd.reference)) if cfg.rmsd.reference else universes[0][1]
    alignment_selection = cfg.rmsd.align_selection
    rmsd_selection = cfg.rmsd.selection
    ref_residues = list(reference.select_atoms(alignment_selection).residues)
    ref_res = residue_records(ref_residues)

    ligand_site_ref_resindices: set[int] = set()
    if cfg.rmsd.region_mode == "ligand_site":
        ligand_sel = cfg.rmsd.ligand_selection or ""
        ligand_atoms = reference.select_atoms(ligand_sel)
        if len(ligand_atoms) == 0:
            raise RuntimeError(f"No ligand atoms found in reference for selection: {ligand_sel}")
        ligand_site_ref_resindices = ligand_site_resindices(reference, ligand_sel, cfg.rmsd.site_cutoff)
        if not ligand_site_ref_resindices:
            raise RuntimeError("No protein residues found around ligand in reference structure")

    all_rows = []
    mapping_rows = []
    mapping_mode_rows = []
    export_rows = []
    export_root = dirs["representatives"] / "rmsd_below_threshold"
    export_root.mkdir(parents=True, exist_ok=True)
    export_count_by_traj: dict[str, int] = {}

    for traj_name, u in universes:
        export_count_by_traj.setdefault(traj_name, 0)
        mobile_residues, ref_residues, mapping, selected_mapping_strategy = build_residue_mapping(
            mobile_universe=u,
            reference_universe=reference,
            align_selection=alignment_selection,
            map_mode=cfg.rmsd.map_mode,
            map_file=cfg.rmsd.map_file,
        )
        mobile_res = residue_records(mobile_residues)
        mapping_mode_rows.append(
            {
                "trajectory": traj_name,
                "selected_mapping_strategy": selected_mapping_strategy,
                "mapped_residues": len(mapping),
            }
        )

        if cfg.rmsd.region_mode == "ligand_site":
            mapping = filter_mapping_to_reference_resindices(mapping, ref_residues, ligand_site_ref_resindices)

        if len(mapping) < cfg.rmsd.min_mapped:
            raise RuntimeError(f"RMSD mapping too small for {traj_name}: {len(mapping)} < {cfg.rmsd.min_mapped}")

        if cfg.rmsd.region_mode == "ligand_site":
            # In ligand-site mode, evaluate coverage against the ligand-site target size,
            # not against the full protein length.
            region_target = max(min(len(ligand_site_ref_resindices), len(mobile_res)), 1)
            coverage = len(mapping) / region_target
        else:
            coverage = len(mapping) / max(len(mobile_res), 1)
        if coverage < cfg.rmsd.min_coverage:
            raise RuntimeError(f"RMSD coverage too low for {traj_name}: {coverage:.3f} < {cfg.rmsd.min_coverage}")

        mobile_indices: list[int] = []
        ref_indices: list[int] = []
        for mob_i, ref_i in mapping:
            mob_atoms = mobile_residues[mob_i].atoms.select_atoms(rmsd_selection)
            ref_atoms = ref_residues[ref_i].atoms.select_atoms(rmsd_selection)
            if len(mob_atoms) == 0 or len(ref_atoms) == 0:
                continue
            ref_by_name = {atom.name: atom.index for atom in ref_atoms}
            for atom in mob_atoms:
                ref_idx = ref_by_name.get(atom.name)
                if ref_idx is None:
                    continue
                mobile_indices.append(int(atom.index))
                ref_indices.append(int(ref_idx))

        if len(ref_indices) < 3:
            raise RuntimeError(
                f"RMSD atom pair count too small for {traj_name}: {len(ref_indices)} from selection '{rmsd_selection}'"
            )

        ref_atom_group = reference.atoms[ref_indices]
        ref_coords = ref_atom_group.positions.copy()
        mobile_atom_group = u.atoms[mobile_indices]

        if cfg.rmsd.region_mode == "ligand_site":
            _write_atom_subset_pdb(
                reference,
                ref_indices,
                dirs["data"] / f"rmsd_ligand_site_reference_{traj_name}.pdb",
            )
            _write_atom_subset_pdb(
                u,
                mobile_indices,
                dirs["data"] / f"rmsd_ligand_site_{traj_name}.pdb",
            )

        for mob_i, ref_i in mapping:
            mapping_rows.append(
                {
                    "trajectory": traj_name,
                    "target_resid": mobile_res[mob_i][0],
                    "target_resname": mobile_res[mob_i][1],
                    "target_chain": mobile_res[mob_i][2],
                    "ref_resid": ref_res[ref_i][0],
                    "ref_resname": ref_res[ref_i][1],
                    "ref_chain": ref_res[ref_i][2],
                    "region_mode": cfg.rmsd.region_mode,
                }
            )

        debug_written = 0
        for ts in u.trajectory[_frame_slice(cfg)]:
            mob_coords = mobile_atom_group.positions.copy()
            rmsd, n_used = _compute_rmsd(
                mob_coords,
                ref_coords,
                superpose=cfg.rmsd.align,
            )
            all_rows.append({"trajectory": traj_name, "frame": int(ts.frame), "rmsd": rmsd, "n_atoms_used": n_used})
            if (
                cfg.rmsd.export_below_threshold
                and rmsd <= cfg.rmsd.threshold_angstrom
                and export_count_by_traj[traj_name] < max(cfg.rmsd.max_export_frames, 0)
            ):
                if cfg.rmsd.export_selection.strip().lower() == "all":
                    export_ag = u.atoms
                else:
                    export_ag = u.select_atoms(cfg.rmsd.export_selection)
                if len(export_ag) > 0:
                    out_pdb = export_root / f"{traj_name}_frame_{int(ts.frame):05d}_rmsd_{rmsd:.3f}.pdb"
                    export_ag.write(str(out_pdb))
                    export_rows.append(
                        {
                            "trajectory": traj_name,
                            "frame": int(ts.frame),
                            "rmsd": float(rmsd),
                            "n_atoms_exported": int(len(export_ag)),
                            "pdb_path": str(out_pdb),
                        }
                    )
                    export_count_by_traj[traj_name] += 1
            if cfg.rmsd.debug_write_aligned_pdb and debug_written < max(cfg.rmsd.debug_max_frames, 0):
                if cfg.rmsd.align:
                    mob_coords_aligned = _kabsch_fit(mob_coords, ref_coords)
                else:
                    mob_coords_aligned = mob_coords
                _write_debug_pdb(
                    ref_atom_group,
                    ref_coords,
                    dirs["data"] / f"rmsd_debug_ref_{traj_name}_frame{int(ts.frame)}.pdb",
                    pairwise_ids=True,
                )
                _write_debug_pdb(
                    mobile_atom_group,
                    mob_coords_aligned,
                    dirs["data"] / f"rmsd_debug_mobile_aligned_{traj_name}_frame{int(ts.frame)}.pdb",
                    pairwise_ids=True,
                )
                debug_written += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(dirs["tables"] / "rmsd_vs_reference.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(dirs["tables"] / "residue_mapping.csv", index=False)
    if cfg.rmsd.export_below_threshold:
        pd.DataFrame(export_rows).to_csv(dirs["tables"] / "rmsd_below_threshold_frames.csv", index=False)

    threshold_summary_rows: list[dict[str, Any]] = []
    if len(df) > 0:
        export_df = pd.DataFrame(export_rows) if len(export_rows) > 0 else pd.DataFrame(columns=["trajectory"])
        for traj_name, sub in df.groupby("trajectory"):
            vals = sub["rmsd"].dropna()
            n_eval = int(len(vals))
            n_below = int((vals <= cfg.rmsd.threshold_angstrom).sum()) if n_eval > 0 else 0
            frac = float(n_below / n_eval) if n_eval > 0 else float("nan")
            n_exported = int((export_df["trajectory"] == traj_name).sum()) if "trajectory" in export_df.columns else 0
            threshold_summary_rows.append(
                {
                    "trajectory": str(traj_name),
                    "threshold_angstrom": float(cfg.rmsd.threshold_angstrom),
                    "n_frames_evaluated": n_eval,
                    "n_frames_below_threshold": n_below,
                    "fraction_below_threshold": frac,
                    "percent_below_threshold": float(frac * 100.0) if n_eval > 0 else float("nan"),
                    "export_below_threshold": bool(cfg.rmsd.export_below_threshold),
                    "n_frames_exported": n_exported,
                    "max_export_frames": int(cfg.rmsd.max_export_frames),
                    "export_fraction_among_below": float(n_exported / n_below) if n_below > 0 else float("nan"),
                }
            )
    pd.DataFrame(threshold_summary_rows).to_csv(dirs["tables"] / "rmsd_threshold_summary.csv", index=False)

    report = {
        "mapped_pairs": int(len(mapping_rows)),
        "map_mode": cfg.rmsd.map_mode,
        "align_selection": alignment_selection,
        "selection": rmsd_selection,
        "align": cfg.rmsd.align,
        "region_mode": cfg.rmsd.region_mode,
        "debug_write_aligned_pdb": cfg.rmsd.debug_write_aligned_pdb,
        "debug_max_frames": cfg.rmsd.debug_max_frames,
        "ligand_site_residue_count": len(ligand_site_ref_resindices),
        "coverage_mode": "ligand_site_region" if cfg.rmsd.region_mode == "ligand_site" else "global_protein",
        "ligand_site_reference_pdb": str(dirs["data"] / "rmsd_ligand_site_reference_<trajectory>.pdb")
        if cfg.rmsd.region_mode == "ligand_site"
        else None,
        "mapping_strategy_per_trajectory": mapping_mode_rows,
        "export_below_threshold": cfg.rmsd.export_below_threshold,
        "threshold_angstrom": cfg.rmsd.threshold_angstrom,
        "export_selection": cfg.rmsd.export_selection,
        "max_export_frames": cfg.rmsd.max_export_frames,
        "exported_frame_count": len(export_rows),
        "exported_frame_count_by_trajectory": export_count_by_traj,
        "export_dir": str(export_root),
        "threshold_summary": threshold_summary_rows,
    }
    (dirs["data"] / "mapping_report.json").write_text(json.dumps(report, indent=2))
    _plot_timeseries_and_distribution(cfg, df, "frame", "rmsd", "trajectory", "rmsd", "RMSD")
