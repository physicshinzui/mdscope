from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .mapping import (
    build_residue_mapping,
    filter_mapping_to_reference_resindices,
    ligand_site_resindices,
    residue_records,
)


@dataclass
class RunContext:
    config: AppConfig
    outdir: Path
    cache: dict[str, Any]


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    paths = {
        "tables": outdir / "tables",
        "figures": outdir / "figures",
        "data": outdir / "data",
        "representatives": outdir / "representatives",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _imports() -> tuple[Any, Any, Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import MDAnalysis as mda

    return mda, np, pd, plt


def _frame_slice(config: AppConfig) -> slice:
    return slice(config.frames.start, config.frames.stop, config.frames.step)


def _trajectory_names(config: AppConfig) -> list[str]:
    if config.system.trajectory_names:
        return config.system.trajectory_names
    return [f"traj{i}" for i, _ in enumerate(config.system.trajectories)]


def _load_universes(config: AppConfig) -> list[tuple[str, Any]]:
    mda, _, _, _ = _imports()
    names = _trajectory_names(config)
    universes = []
    topologies = config.system.expanded_topologies()
    for name, topology, trajectory in zip(names, topologies, config.system.trajectories):
        universes.append((name, mda.Universe(str(topology), str(trajectory))))
    return universes


def _save_plot(
    config: AppConfig,
    fig: Any,
    out_prefix: Path,
) -> None:
    for fmt in config.output.figure_formats:
        kwargs: dict[str, Any] = {}
        if fmt == "png":
            kwargs["dpi"] = config.output.dpi
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), bbox_inches="tight", **kwargs)


def _plot_timeseries_and_distribution(
    config: AppConfig,
    df: Any,
    x_col: str,
    y_col: str,
    hue_col: str,
    base_name: str,
    ylabel: str,
) -> None:
    _, np, _, plt = _imports()

    fig_ts, ax_ts = plt.subplots(figsize=(7.2, 4.5))
    for key, sub in df.groupby(hue_col):
        ax_ts.plot(sub[x_col], sub[y_col], label=str(key), lw=1.2)
    ax_ts.set_xlabel(x_col)
    ax_ts.set_ylabel(ylabel)
    ax_ts.legend(loc="best")
    _save_plot(config, fig_ts, Path(config.output.outdir) / "figures" / f"{base_name}_timeseries")
    plt.close(fig_ts)

    if not config.plot.timeseries_distribution:
        return

    fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
    kind = config.plot.distribution_kind
    for key, sub in df.groupby(hue_col):
        values = sub[y_col].dropna().to_numpy()
        if len(values) == 0:
            continue
        if kind in {"hist", "hist_kde"}:
            bins = config.plot.hist_bins
            if bins is None:
                bins = _auto_hist_bins(values, method=config.plot.hist_bin_method)
            ax_dist.hist(values, bins=bins, density=True, alpha=0.35, label=str(key))
        if kind in {"kde", "hist_kde"}:
            xs = np.linspace(values.min(), values.max(), 200)
            if values.std() > 0:
                bw = 1.06 * values.std() * (len(values) ** (-1.0 / 5.0))
                bw = max(bw, 1e-6)
                kernel = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2)
                density = kernel.sum(axis=1) / (len(values) * bw * np.sqrt(2.0 * np.pi))
                ax_dist.plot(xs, density, lw=1.2, label=f"{key} kde")
    ax_dist.set_xlabel(ylabel)
    ax_dist.set_ylabel("density")
    ax_dist.legend(loc="best")
    _save_plot(config, fig_dist, Path(config.output.outdir) / "figures" / f"{base_name}_distribution")
    plt.close(fig_dist)


def _auto_hist_bins(values: Any, method: str = "fd") -> int:
    _, np, _, _ = _imports()
    values = np.asarray(values)
    if len(values) < 2:
        return 1
    if np.allclose(values.min(), values.max()):
        return 1

    edges = np.histogram_bin_edges(values, bins=method)
    bins = max(len(edges) - 1, 1)
    return int(min(max(bins, 10), 120))


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

    for traj_name, u in universes:
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
    }
    (dirs["data"] / "mapping_report.json").write_text(json.dumps(report, indent=2))
    _plot_timeseries_and_distribution(cfg, df, "frame", "rmsd", "trajectory", "rmsd", "RMSD")


def run_rmsf(ctx: RunContext) -> None:
    _, _, pd, plt = _imports()
    from MDAnalysis.analysis.align import AlignTraj, AverageStructure
    from MDAnalysis.analysis.rms import RMSF

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    rows = []
    for name, u in _load_universes(cfg):
        if cfg.rmsf.align:
            if cfg.rmsf.align_to == "average":
                avg = AverageStructure(u, u, select=cfg.rmsf.align_selection, ref_frame=0).run()
                AlignTraj(u, avg.results.universe, select=cfg.rmsf.align_selection, in_memory=True).run()
            else:
                ref_u = u.copy()
                ref_u.trajectory[0]
                AlignTraj(u, ref_u, select=cfg.rmsf.align_selection, in_memory=True).run()

        ag = u.select_atoms(cfg.rmsf.selection)
        rmsf = RMSF(ag).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        for atom, value in zip(ag, rmsf.results.rmsf):
            rows.append(
                {
                    "trajectory": name,
                    "resid": int(atom.resid),
                    "resname": atom.resname,
                    "rmsf": float(value),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "rmsf_per_residue.csv", index=False)

    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for traj_name, sub in df.groupby("trajectory"):
        ax.plot(sub["resid"], sub["rmsf"], lw=1.2, label=str(traj_name))
    ax.set_xlabel("resid")
    ax.set_ylabel("RMSF")
    ax.legend(loc="best")
    _save_plot(cfg, fig, dirs["figures"] / "rmsf_per_residue")
    plt.close(fig)


def run_rg(ctx: RunContext) -> None:
    mda, _, pd, _ = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    rows = []
    for name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.system.selection)
        for ts in u.trajectory[_frame_slice(cfg)]:
            rows.append({"trajectory": name, "frame": int(ts.frame), "rg": float(ag.radius_of_gyration())})

    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "rg_timeseries.csv", index=False)
    _plot_timeseries_and_distribution(cfg, df, "frame", "rg", "trajectory", "rg", "Radius of gyration")


def _collect_matrix(u: Any, selection: str, frame_slice: slice) -> tuple[Any, list[int]]:
    _, np, _, _ = _imports()
    ag = u.select_atoms(selection)
    rows = []
    frames = []
    for ts in u.trajectory[frame_slice]:
        rows.append(ag.positions.reshape(-1).copy())
        frames.append(int(ts.frame))
    return np.vstack(rows), frames


def _collect_matrix_from_atom_indices(u: Any, atom_indices: list[int], frame_slice: slice) -> tuple[Any, list[int]]:
    _, np, _, _ = _imports()
    ag = u.atoms[atom_indices]
    rows = []
    frames = []
    for ts in u.trajectory[frame_slice]:
        rows.append(ag.positions.reshape(-1).copy())
        frames.append(int(ts.frame))
    return np.vstack(rows), frames


def _plot_pca_free_energy_rt(
    cfg: AppConfig,
    scores: Any,
    out_prefix: Path,
    title: str,
) -> None:
    _, np, _, plt = _imports()

    data = scores[(scores["frame"] != -1) & scores["PC1"].notna() & scores["PC2"].notna()]
    if len(data) < 10:
        return

    x = data["PC1"].to_numpy()
    y = data["PC2"].to_numpy()
    bins_cfg = cfg.pca.free_energy_bins
    if isinstance(bins_cfg, str) and bins_cfg == "auto_fd":
        xbins = _auto_hist_bins(x, method="fd")
        ybins = _auto_hist_bins(y, method="fd")
        bins_2d: Any = (xbins, ybins)
    else:
        bins_2d = int(bins_cfg)

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins_2d)
    total = float(hist.sum())
    if total <= 0:
        return

    prob = hist / total
    sigma = float(cfg.pca.free_energy_smooth_sigma)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        prob = gaussian_filter(prob, sigma=sigma, mode="nearest")
        sm_total = float(prob.sum())
        if sm_total > 0:
            prob = prob / sm_total
    with np.errstate(divide="ignore", invalid="ignore"):
        free_e = -np.log(prob)
    mask = prob > 0
    if not np.any(mask):
        return
    free_e = free_e - np.nanmin(free_e[mask])
    free_e[~mask] = np.nan

    max_e = float(np.nanmax(free_e))
    if not np.isfinite(max_e):
        return
    fe_max = 10.0
    step = cfg.pca.free_energy_level_step_rt
    levels = np.arange(0.0, fe_max + step, step)
    if len(levels) < 2:
        levels = np.array([0.0, step], dtype=float)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 6))
    free_e_plot = np.clip(free_e, 0.0, fe_max)
    cf = ax.contourf(X, Y, free_e_plot, levels=levels, cmap="viridis", vmin=0.0, vmax=fe_max)
    cs = ax.contour(X, Y, free_e_plot, levels=levels, colors="black", linewidths=0.4, alpha=0.45)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    refs = scores[(scores["frame"] == -1) & scores["PC1"].notna() & scores["PC2"].notna()]
    if len(refs) > 0:
        ax.scatter(refs["PC1"], refs["PC2"], marker="x", s=48, linewidths=1.6, c="red", label="reference")
        ax.legend(loc="best")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Free energy (RT)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    _save_plot(cfg, fig, out_prefix)
    plt.close(fig)


def run_pca(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    from sklearn.decomposition import PCA

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

    names = _trajectory_names(cfg)
    matrices: dict[str, Any] = {}
    frames_map: dict[str, list[int]] = {}
    universes = _load_universes(cfg)
    site_key_order: list[tuple[str, int, str]] | None = None
    if cfg.pca.site_from_reference_ligand:
        site_ref_path = cfg.pca.site_reference_pdb or cfg.rmsd.reference
        if site_ref_path is None:
            raise RuntimeError("pca.site_reference_pdb or rmsd.reference is required when pca.site_from_reference_ligand=true")
        if not cfg.pca.site_ligand_selection:
            raise RuntimeError("pca.site_ligand_selection is required when pca.site_from_reference_ligand=true")
        ref_u = _imports()[0].Universe(str(site_ref_path))
        ref_site_res = ligand_site_resindices(ref_u, cfg.pca.site_ligand_selection, cfg.pca.site_cutoff)
        if not ref_site_res:
            raise RuntimeError("No reference ligand-site residues for PCA site mode")

        atom_idx_by_traj: dict[str, dict[tuple[str, int, str], int]] = {}
        map_rows = []
        for name, u in universes:
            mob_res, ref_res, mapping, strategy = build_residue_mapping(
                mobile_universe=u,
                reference_universe=ref_u,
                align_selection=cfg.pca.site_align_selection,
                map_mode=cfg.pca.site_map_mode,
                map_file=cfg.pca.site_map_file,
            )
            mapping = filter_mapping_to_reference_resindices(mapping, ref_res, ref_site_res)
            if len(mapping) < 1:
                raise RuntimeError(f"No mapped ligand-site residues for trajectory: {name}")

            atom_map: dict[tuple[str, int, str], int] = {}
            for mob_i, ref_i in mapping:
                mob_atoms = mob_res[mob_i].atoms.select_atoms(cfg.pca.site_atom_selection)
                ref_atoms = ref_res[ref_i].atoms.select_atoms(cfg.pca.site_atom_selection)
                if len(mob_atoms) == 0 or len(ref_atoms) == 0:
                    continue
                ref_name_to_atom = {str(a.name): a for a in ref_atoms}
                for ma in mob_atoms:
                    ra = ref_name_to_atom.get(str(ma.name))
                    if ra is None:
                        continue
                    key = (str(ra.segid), int(ra.resid), str(ra.name))
                    atom_map[key] = int(ma.index)
                map_rows.append(
                    {
                        "trajectory": name,
                        "mapping_strategy": strategy,
                        "target_resid": int(mob_res[mob_i].resid),
                        "target_chain": str(mob_res[mob_i].segid),
                        "ref_resid": int(ref_res[ref_i].resid),
                        "ref_chain": str(ref_res[ref_i].segid),
                        "status": "mapped",
                    }
                )

            atom_idx_by_traj[name] = atom_map

        common_keys = set.intersection(*[set(v.keys()) for v in atom_idx_by_traj.values()]) if atom_idx_by_traj else set()
        if len(common_keys) < 3:
            raise RuntimeError(f"PCA site mode found too few common mapped atoms: {len(common_keys)}")
        key_order = sorted(common_keys)
        site_key_order = key_order

        for name, u in universes:
            atom_indices = [atom_idx_by_traj[name][k] for k in key_order]
            mat, frames = _collect_matrix_from_atom_indices(u, atom_indices, _frame_slice(cfg))
            matrices[name] = mat
            frames_map[name] = frames

        pd.DataFrame(map_rows).to_csv(dirs["tables"] / "pca_site_residue_mapping.csv", index=False)
        (dirs["data"] / "pca_site_selection_report.json").write_text(
            json.dumps(
                {
                    "site_reference_pdb": str(site_ref_path),
                    "site_ligand_selection": cfg.pca.site_ligand_selection,
                    "site_cutoff": cfg.pca.site_cutoff,
                    "site_map_mode": cfg.pca.site_map_mode,
                    "site_atom_selection": cfg.pca.site_atom_selection,
                    "common_mapped_atoms": len(common_keys),
                },
                indent=2,
            )
        )
    else:
        for name, u in universes:
            mat, frames = _collect_matrix(u, cfg.pca.selection, _frame_slice(cfg))
            matrices[name] = mat
            frames_map[name] = frames

    fit_name = cfg.pca.fit_trajectory or names[0]
    fit_matrix = matrices[fit_name]
    pca = PCA(n_components=min(cfg.pca.n_components, fit_matrix.shape[1], fit_matrix.shape[0]))

    if cfg.pca.mode == "joint":
        concat = np.vstack([matrices[n] for n in names])
        pca.fit(concat)
    else:
        pca.fit(fit_matrix)

    scores_rows = []
    for name in names:
        sc = pca.transform(matrices[name])
        for idx, frame in enumerate(frames_map[name]):
            row = {"trajectory": name, "frame": frame}
            for pc_i in range(sc.shape[1]):
                row[f"PC{pc_i + 1}"] = float(sc[idx, pc_i])
            scores_rows.append(row)

    for idx, pdb_path in enumerate(cfg.pca.reference_pdbs):
        mda, _, _, _ = _imports()
        ref_u = mda.Universe(str(pdb_path))
        if cfg.pca.site_from_reference_ligand:
            if site_key_order is None:
                raise RuntimeError("Internal error: missing site key order for PCA site mode")
            ref_atoms = ref_u.select_atoms(cfg.pca.site_atom_selection)
            ref_key_to_pos = {(str(a.segid), int(a.resid), str(a.name)): a.position.copy() for a in ref_atoms}
            missing = [k for k in site_key_order if k not in ref_key_to_pos]
            if missing:
                raise RuntimeError(
                    f"PCA reference atom mapping mismatch for {pdb_path}: "
                    f"{len(missing)} site atoms are missing for selection '{cfg.pca.site_atom_selection}'."
                )
            coords = [ref_key_to_pos[k] for k in site_key_order]
            ref_vec = np.array(coords).reshape(1, -1)
        else:
            ref_sel = cfg.pca.selection
            ref_atoms = ref_u.select_atoms(ref_sel)
            expected_atoms = pca.n_features_in_ // 3
            if len(ref_atoms) != expected_atoms:
                raise RuntimeError(
                    f"PCA reference atom count mismatch for {pdb_path}: "
                    f"selection '{ref_sel}' gives {len(ref_atoms)} atoms, expected {expected_atoms}. "
                    "Use a selection that matches the PCA fit atoms exactly."
                )
            ref_vec = ref_atoms.positions.reshape(1, -1)
        ref_sc = pca.transform(ref_vec)
        ref_name = cfg.pca.reference_names[idx] if idx < len(cfg.pca.reference_names) else pdb_path.stem
        row = {"trajectory": ref_name, "frame": -1}
        for pc_i in range(ref_sc.shape[1]):
            row[f"PC{pc_i + 1}"] = float(ref_sc[0, pc_i])
        scores_rows.append(row)

    scores = pd.DataFrame(scores_rows)
    scores.to_csv(dirs["tables"] / "pca_scores.csv", index=False)
    eig = pd.DataFrame({"component": [i + 1 for i in range(len(pca.explained_variance_))], "eigenvalue": pca.explained_variance_})
    eig.to_csv(dirs["tables"] / "pca_eigenvalues.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    for tr, sub in scores.groupby("trajectory"):
        if "PC1" in sub and "PC2" in sub:
            marker = "x" if (sub["frame"] == -1).all() else "o"
            size = 42 if marker == "x" else 8
            alpha = 0.9 if marker == "x" else 0.6
            lw = 1.5 if marker == "x" else 0.0
            ax.scatter(sub["PC1"], sub["PC2"], s=size, alpha=alpha, marker=marker, linewidths=lw, label=str(tr))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    _save_plot(cfg, fig, dirs["figures"] / "pca_scatter")
    plt.close(fig)

    if cfg.pca.free_energy_enabled:
        _plot_pca_free_energy_rt(
            cfg,
            scores,
            dirs["figures"] / "pca_free_energy_rt",
            "PCA Free Energy Landscape (RT)",
        )
        if cfg.pca.free_energy_per_trajectory:
            for traj_name, sub in scores.groupby("trajectory"):
                if (sub["frame"] == -1).all():
                    continue
                _plot_pca_free_energy_rt(
                    cfg,
                    sub,
                    dirs["figures"] / f"pca_free_energy_rt_{traj_name}",
                    f"PCA Free Energy Landscape (RT): {traj_name}",
                )


def run_cluster(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    import hdbscan

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    scores_path = dirs["tables"] / "pca_scores.csv"
    if not scores_path.exists():
        raise RuntimeError("pca_scores.csv not found; run pca step first")

    scores = pd.read_csv(scores_path)
    pcs = [f"PC{i}" for i in cfg.pca.use_pcs]
    pcs = [c for c in pcs if c in scores.columns]
    X = scores[pcs].to_numpy()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.cluster.min_cluster_size,
        min_samples=cfg.cluster.min_samples,
        metric=cfg.cluster.metric,
        cluster_selection_method=cfg.cluster.selection_method,
        allow_single_cluster=cfg.cluster.allow_single_cluster,
    )
    labels = clusterer.fit_predict(X)

    out = scores[["trajectory", "frame"]].copy()
    out["label"] = labels
    out["probability"] = getattr(clusterer, "probabilities_", np.zeros(len(out)))

    label_to_name: dict[int, str] = {-1: "noise"}
    used_names: set[str] = {"noise"}
    for label in sorted(int(v) for v in out["label"].unique() if int(v) >= 0):
        members = out[out["label"] == label]
        if len(members) == 0:
            label_to_name[label] = f"cluster_{label}"
            continue
        dominant = str(members["trajectory"].value_counts().idxmax())
        name = dominant
        if name in used_names:
            name = f"{dominant}_{label}"
        used_names.add(name)
        label_to_name[label] = name
    out["cluster_name"] = out["label"].map(lambda x: label_to_name.get(int(x), f"cluster_{int(x)}"))
    out.to_csv(dirs["tables"] / "hdbscan_labels.csv", index=False)

    summary = (
        out[out["label"] >= 0]
        .groupby(["label", "cluster_name"])
        .size()
        .reset_index(name="size")
        .sort_values("label")
    )
    summary.to_csv(dirs["tables"] / "cluster_summary.csv", index=False)

    merged = scores.merge(out[["trajectory", "frame", "label", "cluster_name"]], on=["trajectory", "frame"], how="left")
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    labels_sorted = sorted(int(v) for v in merged["label"].dropna().unique())
    color_by_label = {label: cmap(i % 10) for i, label in enumerate(labels_sorted)}

    normal = merged[merged["frame"] != -1]
    for label, sub in normal.groupby("label"):
        label_i = int(label)
        cluster_name = str(sub["cluster_name"].iloc[0]) if "cluster_name" in sub else ("noise" if label_i == -1 else f"cluster_{label_i}")
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=10,
            alpha=0.7,
            marker="o",
            linewidths=0.0,
            color=color_by_label.get(label_i),
            label=cluster_name,
        )

    refs = merged[merged["frame"] == -1]
    for ref_name, sub in refs.groupby("trajectory"):
        label_i = int(sub["label"].iloc[0])
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=48,
            alpha=0.95,
            marker="x",
            linewidths=1.6,
            color=color_by_label.get(label_i),
            label=str(ref_name),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    _save_plot(cfg, fig, dirs["figures"] / "pca_scatter_clustered")
    plt.close(fig)


def run_representative(ctx: RunContext) -> None:
    _, np, pd, _ = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    scores = pd.read_csv(dirs["tables"] / "pca_scores.csv")
    labels = pd.read_csv(dirs["tables"] / "hdbscan_labels.csv")
    merged = scores.merge(labels[["trajectory", "frame", "label", "cluster_name"]], on=["trajectory", "frame"], how="left")

    pc_cols = [f"PC{i}" for i in cfg.pca.use_pcs if f"PC{i}" in merged.columns]

    universes = {name: u for name, u in _load_universes(cfg)}
    representatives = []

    for label, sub in merged.groupby("label"):
        if int(label) < 0:
            continue
        X = sub[pc_cols].to_numpy()
        if len(X) == 0:
            continue
        center = X.mean(axis=0)
        idx = int(np.argmin(((X - center) ** 2).sum(axis=1)))
        row = sub.iloc[idx]
        traj_name = str(row["trajectory"])
        frame = int(row["frame"])
        cluster_name = str(row.get("cluster_name", f"cluster_{int(label)}"))
        u = universes[traj_name]
        u.trajectory[frame]
        out_pdb = dirs["representatives"] / f"cluster_{int(label)}_rep.pdb"
        u.atoms.write(str(out_pdb))
        representatives.append(
            {
                "label": int(label),
                "cluster_name": cluster_name,
                "trajectory": traj_name,
                "frame": frame,
                "pdb": str(out_pdb),
            }
        )

    pd.DataFrame(representatives).to_csv(dirs["tables"] / "representatives.csv", index=False)


def run_distance(ctx: RunContext) -> None:
    _, np, pd, _ = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    rows = []
    for traj_name, u in _load_universes(cfg):
        for pair in cfg.distance.pairs:
            a = u.select_atoms(pair.sel1)
            b = u.select_atoms(pair.sel2)
            if len(a) == 0 or len(b) == 0:
                continue
            for ts in u.trajectory[_frame_slice(cfg)]:
                da = a.center_of_geometry()
                db = b.center_of_geometry()
                dist = float(np.linalg.norm(da - db))
                rows.append({"trajectory": traj_name, "pair_id": pair.id, "frame": int(ts.frame), "distance": dist})

    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "distance_timeseries.csv", index=False)
    if len(df) > 0:
        for pair_id, sub in df.groupby("pair_id"):
            _plot_timeseries_and_distribution(cfg, sub, "frame", "distance", "trajectory", f"distance_{pair_id}", f"Distance {pair_id}")


def run_ramachandran(ctx: RunContext) -> None:
    _, _, pd, plt = _imports()
    from MDAnalysis.analysis.dihedrals import Ramachandran

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    rows = []

    for traj_name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.ramachandran.selection)
        rama = Ramachandran(ag).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        angles = rama.results.angles
        residues = list(ag.residues)
        for frame_i, frame_vals in enumerate(angles):
            for res_i, (phi, psi) in enumerate(frame_vals):
                if res_i >= len(residues):
                    continue
                residue = residues[res_i]
                rows.append(
                    {
                        "trajectory": traj_name,
                        "frame_index": frame_i,
                        "chain": residue.segid,
                        "resid": int(residue.resid),
                        "resname": residue.resname,
                        "phi": float(phi),
                        "psi": float(psi),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "phi_psi_timeseries.csv", index=False)
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["phi"], df["psi"], s=3, alpha=0.25)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("phi")
    ax.set_ylabel("psi")
    _save_plot(cfg, fig, dirs["figures"] / "ramachandran_global")
    plt.close(fig)


def run_dssp(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from MDAnalysis.analysis.dssp import DSSP

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

    # Cleanup legacy combined DSSP plots to avoid mixing old/new naming schemes.
    for fmt in cfg.output.figure_formats:
        legacy_ts = dirs["figures"] / f"dssp_fraction_timeseries.{fmt}"
        if legacy_ts.exists():
            legacy_ts.unlink()
        legacy_res = dirs["figures"] / f"dssp_fraction_per_residue.{fmt}"
        if legacy_res.exists():
            legacy_res.unlink()

    frame_rows = []
    residue_rows = []
    heatmap_payloads: list[tuple[str, Any, list[int]]] = []

    for traj_name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.dssp.selection)
        if len(ag) == 0:
            continue

        frame_numbers = [int(ts.frame) for ts in u.trajectory[_frame_slice(cfg)]]
        dssp = DSSP(ag).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        codes = np.asarray(dssp.results.dssp)
        if codes.ndim != 2 or codes.shape[0] == 0:
            continue
        resids = np.asarray(dssp.results.resids, dtype=int)
        if len(resids) != codes.shape[1]:
            raise RuntimeError(
                f"DSSP residue axis mismatch for {traj_name}: codes={codes.shape[1]} resids={len(resids)}"
            )

        n_frames = min(len(frame_numbers), codes.shape[0])
        frame_numbers = frame_numbers[:n_frames]
        codes = codes[:n_frames, :]

        # Normalize DSSP to 3-state classes used in output: H (helix), S (sheet), C (coil).
        codes_norm = np.full(codes.shape, cfg.dssp.coil_code, dtype="<U1")
        codes_norm[np.isin(codes, ["H", "G", "I"])] = "H"
        codes_norm[np.isin(codes, ["E", "B"])] = "S"
        heatmap_payloads.append((traj_name, codes_norm.copy(), list(resids)))

        for fi in range(n_frames):
            row_codes = codes_norm[fi]
            n_res = len(row_codes)
            if n_res == 0:
                continue
            n_h = int(np.sum(row_codes == "H"))
            n_s = int(np.sum(row_codes == "S"))
            n_c = int(np.sum(row_codes == cfg.dssp.coil_code))
            frame_rows.append(
                {
                    "trajectory": traj_name,
                    "frame": int(frame_numbers[fi]),
                    "frame_index": fi,
                    "helix_fraction": float(n_h / n_res),
                    "sheet_fraction": float(n_s / n_res),
                    "beta_fraction": float(n_s / n_res),
                    "coil_fraction": float(n_c / n_res),
                }
            )

        for ri, resid in enumerate(resids):
            col = codes_norm[:, ri]
            n_obs = len(col)
            if n_obs == 0:
                continue
            residue_rows.append(
                {
                    "trajectory": traj_name,
                    "resid": int(resid),
                    "helix_fraction": float(np.sum(col == "H") / n_obs),
                    "sheet_fraction": float(np.sum(col == "S") / n_obs),
                    "beta_fraction": float(np.sum(col == "S") / n_obs),
                    "coil_fraction": float(np.sum(col == cfg.dssp.coil_code) / n_obs),
                }
            )

    frame_df = pd.DataFrame(frame_rows)
    if len(frame_df) == 0:
        pd.DataFrame([{"status": "skipped", "reason": "No DSSP values produced"}]).to_csv(
            dirs["tables"] / "dssp_status.csv", index=False
        )
        return

    frame_df = frame_df.sort_values(["trajectory", "frame"]).reset_index(drop=True)
    residue_df = pd.DataFrame(residue_rows).sort_values(["trajectory", "resid"]).reset_index(drop=True)
    frame_df.to_csv(dirs["tables"] / "dssp_fraction_timeseries.csv", index=False)
    residue_df.to_csv(dirs["tables"] / "dssp_fraction_per_residue.csv", index=False)

    for traj_name, sub in frame_df.groupby("trajectory"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(sub["frame"], sub["helix_fraction"], lw=1.2, ls="-", label="H")
        ax.plot(sub["frame"], sub["sheet_fraction"], lw=1.2, ls="-", label="S")
        ax.plot(sub["frame"], sub["coil_fraction"], lw=1.2, ls="-", label="C")
        ax.set_xlabel("frame")
        ax.set_ylabel("fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(str(traj_name))
        ax.legend(loc="best")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_fraction_timeseries_{traj_name}")
        plt.close(fig)

    for traj_name, sub in residue_df.groupby("trajectory"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(sub["resid"], sub["helix_fraction"], lw=1.2, ls="-", label="H")
        ax.plot(sub["resid"], sub["sheet_fraction"], lw=1.2, ls="-", label="S")
        ax.plot(sub["resid"], sub["coil_fraction"], lw=1.2, ls="-", label="C")
        ax.set_xlabel("resid")
        ax.set_ylabel("fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(str(traj_name))
        ax.legend(loc="best")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_fraction_per_residue_{traj_name}")
        plt.close(fig)

    state_to_idx = {cfg.dssp.coil_code: 0, "S": 1, "H": 2}
    cmap = ListedColormap(["#d9d9d9", "#4c78a8", "#e45756"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for traj_name, states, resids in heatmap_payloads:
        if states.size == 0:
            continue
        # imshow expects [rows, cols] = [residue, time]
        mat = np.vectorize(lambda x: state_to_idx.get(str(x), 0))(states).T
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xlabel("frame index")
        ax.set_ylabel("resid")
        ax.set_title(str(traj_name))

        n_res = len(resids)
        if n_res > 1:
            tick_positions = np.linspace(0, n_res - 1, num=min(8, n_res), dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(resids[i]) for i in tick_positions])
        elif n_res == 1:
            ax.set_yticks([0])
            ax.set_yticklabels([str(resids[0])])

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels([cfg.dssp.coil_code, "S", "H"])
        cbar.set_label("DSSP class")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_heatmap_residue_time_{traj_name}")
        plt.close(fig)


def run_sasa(ctx: RunContext) -> None:
    _, _, pd, _ = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

    try:
        from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
    except Exception:
        marker = pd.DataFrame([{"status": "skipped", "reason": "mdakit_sasa is not installed"}])
        marker.to_csv(dirs["tables"] / "sasa_status.csv", index=False)
        return

    rows = []
    for traj_name, u in _load_universes(cfg):
        sasa = SASAAnalysis(
            u,
            selection=cfg.sasa.selection,
            probe_radius=cfg.sasa.probe_radius,
            n_sphere_points=cfg.sasa.n_sphere_points,
        )
        sasa.run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        values = getattr(sasa.results, "total_area", None)
        if values is None:
            continue
        for frame_i, value in enumerate(values):
            rows.append({"trajectory": traj_name, "frame_index": frame_i, "sasa": float(value)})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        pd.DataFrame([{"status": "skipped", "reason": "No SASA values produced by backend"}]).to_csv(
            dirs["tables"] / "sasa_status.csv", index=False
        )
        return

    df.to_csv(dirs["tables"] / "sasa_timeseries.csv", index=False)
    _plot_timeseries_and_distribution(cfg, df, "frame_index", "sasa", "trajectory", "sasa", "SASA")


def run_placeholder(ctx: RunContext, step_name: str) -> None:
    dirs = ensure_dirs(ctx.outdir)
    (dirs["tables"] / f"{step_name}_status.csv").write_text("status,reason\nskipped,not implemented in v0.1.0\n")


STEP_HANDLERS = {
    "rmsd": run_rmsd,
    "rmsf": run_rmsf,
    "dssp": run_dssp,
    "pca": run_pca,
    "cluster": run_cluster,
    "representative": run_representative,
    "rg": run_rg,
    "sasa": run_sasa,
    "ligand_site": lambda ctx: run_placeholder(ctx, "ligand_site"),
    "distance": run_distance,
    "ramachandran": run_ramachandran,
}
