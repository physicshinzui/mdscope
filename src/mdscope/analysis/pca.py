from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._common import RunContext, _auto_hist_bins, _frame_slice, _imports, _load_universes, _save_plot, _trajectory_names, ensure_dirs
from .mapping import build_residue_mapping, filter_mapping_to_reference_resindices, ligand_site_resindices


REFERENCE_COLORS = [
    "#111111",
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#9467bd",
]


def _pairwise_distance_vector(positions: Any) -> Any:
    _, np, _, _ = _imports()
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (n_atoms, 3), got {pos.shape}")
    if pos.shape[0] < 2:
        raise ValueError("distance-based PCA requires at least 2 atoms")
    iu = np.triu_indices(pos.shape[0], k=1)
    deltas = pos[iu[0]] - pos[iu[1]]
    return np.linalg.norm(deltas, axis=1)


def _collect_matrix(u: Any, selection: str, frame_slice: slice) -> tuple[Any, list[int]]:
    _, np, _, _ = _imports()
    ag = u.select_atoms(selection)
    rows = []
    frames = []
    for ts in u.trajectory[frame_slice]:
        rows.append(ag.positions.reshape(-1).copy())
        frames.append(int(ts.frame))
    return np.vstack(rows), frames


def _collect_distance_matrix(u: Any, selection: str, frame_slice: slice) -> tuple[Any, list[int]]:
    _, np, _, _ = _imports()
    ag = u.select_atoms(selection)
    if len(ag) < 2:
        raise RuntimeError(f"PCA distance mode selection must contain at least 2 atoms: '{selection}'")
    rows = []
    frames = []
    for ts in u.trajectory[frame_slice]:
        rows.append(_pairwise_distance_vector(ag.positions))
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


def _collect_distance_matrix_from_atom_indices(u: Any, atom_indices: list[int], frame_slice: slice) -> tuple[Any, list[int]]:
    _, np, _, _ = _imports()
    ag = u.atoms[atom_indices]
    if len(ag) < 2:
        raise RuntimeError("PCA distance mode site selection must contain at least 2 mapped atoms")
    rows = []
    frames = []
    for ts in u.trajectory[frame_slice]:
        rows.append(_pairwise_distance_vector(ag.positions))
        frames.append(int(ts.frame))
    return np.vstack(rows), frames


def _build_site_atom_map(
    mobile_universe: Any,
    reference_universe: Any,
    source_label: str,
    align_selection: str,
    map_mode: str,
    atom_selection: str,
    reference_resindices: set[int],
    map_file: Path | None = None,
) -> tuple[dict[tuple[str, int, str], int], str, int, list[dict[str, Any]]]:
    mob_res, ref_res, mapping, strategy = build_residue_mapping(
        mobile_universe=mobile_universe,
        reference_universe=reference_universe,
        align_selection=align_selection,
        map_mode=map_mode,
        map_file=map_file,
    )
    mapping = filter_mapping_to_reference_resindices(mapping, ref_res, reference_resindices)
    atom_map: dict[tuple[str, int, str], int] = {}
    residue_pairs: list[dict[str, Any]] = []
    for mob_i, ref_i in mapping:
        ref_chain = str(ref_res[ref_i].segid)
        ref_resid = int(ref_res[ref_i].resid)
        ref_resname = str(ref_res[ref_i].resname)
        mob_chain = str(mob_res[mob_i].segid)
        mob_resid = int(mob_res[mob_i].resid)
        mob_resname = str(mob_res[mob_i].resname)
        if mob_chain != ref_chain:
            print(
                f"[warn] PCA site mapping chain mismatch for {source_label}: "
                f"{ref_chain}:{ref_resid} {ref_resname} -> {mob_chain}:{mob_resid} {mob_resname}"
            )
        if mob_resname != ref_resname:
            print(
                f"[warn] PCA site mapping residue-name mismatch for {source_label}: "
                f"{ref_chain}:{ref_resid} {ref_resname} -> {mob_chain}:{mob_resid} {mob_resname}"
            )
        residue_pairs.append(
            {
                "reference_chain": ref_chain,
                "reference_resid": ref_resid,
                "reference_resname": ref_resname,
                "mobile_chain": mob_chain,
                "mobile_resid": mob_resid,
                "mobile_resname": mob_resname,
            }
        )
        mob_atoms = mob_res[mob_i].atoms.select_atoms(atom_selection)
        ref_atoms = ref_res[ref_i].atoms.select_atoms(atom_selection)
        if len(mob_atoms) == 0 or len(ref_atoms) == 0:
            continue
        ref_name_to_atom = {str(a.name): a for a in ref_atoms}
        for ma in mob_atoms:
            ra = ref_name_to_atom.get(str(ma.name))
            if ra is None:
                continue
            key = (str(ra.segid), int(ra.resid), str(ra.name))
            atom_map[key] = int(ma.index)
    return atom_map, strategy, len(mapping), residue_pairs


def _append_unique_atom_pairs(
    fit_atoms: Any,
    ref_atoms: Any,
    fit_indices: list[int],
    ref_indices: list[int],
    seen_fit: set[int],
    seen_ref: set[int],
) -> None:
    fit_by_name = {str(a.name): int(a.index) for a in fit_atoms}
    for a in ref_atoms:
        fit_idx = fit_by_name.get(str(a.name))
        ref_idx = int(a.index)
        if fit_idx is None or fit_idx in seen_fit or ref_idx in seen_ref:
            continue
        seen_fit.add(fit_idx)
        seen_ref.add(ref_idx)
        fit_indices.append(fit_idx)
        ref_indices.append(ref_idx)


def _plot_pca_free_energy_rt(
    cfg: Any,
    scores: Any,
    out_prefix: Path,
    title: str,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    reference_colors: dict[str, Any] | None = None,
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
        for ref_name, sub in refs.groupby("trajectory"):
            ref_color = reference_colors.get(str(ref_name)) if reference_colors else "red"
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                marker="x",
                s=48,
                linewidths=1.6,
                color=ref_color,
                label=str(ref_name),
                zorder=10,
            )
        ax.legend(loc="best")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Free energy / RT")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if axis_limits is not None:
        (xmin, xmax), (ymin, ymax) = axis_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    _save_plot(cfg, fig, out_prefix)
    plt.close(fig)


def _pc12_axis_limits(scores: Any, cfg: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    _, np, _, _ = _imports()
    mode = getattr(cfg.pca, "plot_axis_mode", "full")
    if mode == "manual":
        if cfg.pca.plot_xlim is None or cfg.pca.plot_ylim is None:
            return None
        return ((float(cfg.pca.plot_xlim[0]), float(cfg.pca.plot_xlim[1])), (float(cfg.pca.plot_ylim[0]), float(cfg.pca.plot_ylim[1])))
    if "PC1" not in scores.columns or "PC2" not in scores.columns:
        return None
    data = scores[scores["PC1"].notna() & scores["PC2"].notna()]
    if len(data) == 0:
        return None
    x = data["PC1"].to_numpy(dtype=float)
    y = data["PC2"].to_numpy(dtype=float)
    if mode == "percentile":
        plo, phi = [float(v) for v in cfg.pca.plot_axis_percentile]
        xmin = float(np.nanpercentile(x, plo))
        xmax = float(np.nanpercentile(x, phi))
        ymin = float(np.nanpercentile(y, plo))
        ymax = float(np.nanpercentile(y, phi))
    else:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
    if not all(np.isfinite(v) for v in [xmin, xmax, ymin, ymax]):
        return None
    xspan = xmax - xmin
    yspan = ymax - ymin
    xpad = max(0.05 * xspan, 1e-6 if xspan <= 0 else 0.0)
    ypad = max(0.05 * yspan, 1e-6 if yspan <= 0 else 0.0)
    if xspan == 0:
        xpad = 0.5
    if yspan == 0:
        ypad = 0.5
    return ((xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))


def _reference_color_map(ref_names: list[str]) -> dict[str, str]:
    return {name: REFERENCE_COLORS[i % len(REFERENCE_COLORS)] for i, name in enumerate(sorted(ref_names))}


def plot_pca_from_scores(cfg: Any, scores: Any, outdir: Path) -> None:
    _, _, pd, plt = _imports()
    dirs = ensure_dirs(outdir)
    if "PC1" not in scores.columns or "PC2" not in scores.columns:
        raise RuntimeError("pca_scores.csv must contain PC1 and PC2 columns to regenerate PCA plots")

    axis_limits = _pc12_axis_limits(scores, cfg)
    refs_only = scores[(scores["frame"] == -1) & scores["PC1"].notna() & scores["PC2"].notna()]
    reference_colors: dict[str, Any] = {}
    if len(refs_only) > 0:
        ref_names = sorted(set(str(v) for v in refs_only["trajectory"].tolist()))
        reference_colors = _reference_color_map(ref_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    for tr, sub in scores.groupby("trajectory"):
        if "PC1" in sub and "PC2" in sub:
            marker = "x" if (sub["frame"] == -1).all() else "o"
            size = 42 if marker == "x" else 8
            alpha = 0.9 if marker == "x" else 0.6
            lw = 1.5 if marker == "x" else 0.0
            zorder = 10 if marker == "x" else 2
            color = reference_colors.get(str(tr)) if marker == "x" else None
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=size,
                alpha=alpha,
                marker=marker,
                linewidths=lw,
                label=str(tr),
                zorder=zorder,
                color=color if color is not None else None,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if axis_limits is not None:
        (xmin, xmax), (ymin, ymax) = axis_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="best")
    _save_plot(cfg, fig, dirs["figures"] / "pca_scatter")
    plt.close(fig)

    for traj_name, sub in scores.groupby("trajectory"):
        if str(traj_name) in set(refs_only["trajectory"].astype(str).tolist()):
            continue
        sub = sub[sub["PC1"].notna() & sub["PC2"].notna()]
        if len(sub) == 0:
            continue
        fig_t, ax_t = plt.subplots(figsize=(6, 6))
        normal = sub[sub["frame"] != -1]
        if len(normal) > 0:
            ax_t.scatter(normal["PC1"], normal["PC2"], s=8, alpha=0.6, marker="o", linewidths=0.0, label=str(traj_name), zorder=2)
        if len(refs_only) > 0:
            for ref_name, sub_ref in refs_only.groupby("trajectory"):
                ref_color = reference_colors.get(str(ref_name), "red")
                ax_t.scatter(
                    sub_ref["PC1"],
                    sub_ref["PC2"],
                    s=42,
                    alpha=0.9,
                    marker="x",
                    linewidths=1.5,
                    color=ref_color,
                    label=str(ref_name),
                    zorder=10,
                )
        ax_t.set_xlabel("PC1")
        ax_t.set_ylabel("PC2")
        if axis_limits is not None:
            (xmin, xmax), (ymin, ymax) = axis_limits
            ax_t.set_xlim(xmin, xmax)
            ax_t.set_ylim(ymin, ymax)
        ax_t.legend(loc="best")
        _save_plot(cfg, fig_t, dirs["figures"] / f"pca_scatter_{traj_name}")
        plt.close(fig_t)

    if cfg.pca.free_energy_enabled:
        _plot_pca_free_energy_rt(
            cfg,
            scores,
            dirs["figures"] / "pca_free_energy_rt",
            "PCA Free Energy Landscape (RT)",
            axis_limits=axis_limits,
            reference_colors=reference_colors,
        )
        if cfg.pca.free_energy_per_trajectory:
            for traj_name, sub in scores.groupby("trajectory"):
                if (sub["frame"] == -1).all():
                    continue
                sub_with_refs = pd.concat([sub, refs_only], ignore_index=True) if len(refs_only) > 0 else sub
                _plot_pca_free_energy_rt(
                    cfg,
                    sub_with_refs,
                    dirs["figures"] / f"pca_free_energy_rt_{traj_name}",
                    f"PCA Free Energy Landscape (RT): {traj_name}",
                    axis_limits=axis_limits,
                    reference_colors=reference_colors,
                )


def run_pca(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    from MDAnalysis.analysis.align import AlignTraj
    from sklearn.decomposition import PCA

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    feature_mode = cfg.pca.feature_mode

    names = _trajectory_names(cfg)
    matrices: dict[str, Any] = {}
    frames_map: dict[str, list[int]] = {}
    universes = _load_universes(cfg)
    universe_map = {name: u for name, u in universes}
    fit_name = cfg.pca.fit_trajectory or names[0]
    fit_u = universe_map[fit_name]
    fit_u.trajectory[0]

    if cfg.pca.align:
        align_sel = cfg.pca.site_align_selection if cfg.pca.site_from_reference_ligand else cfg.system.align_selection
        fit_ref = fit_u.copy()
        fit_ref.trajectory[0]
        for name, u in universes:
            AlignTraj(
                u,
                fit_ref,
                select=align_sel,
                in_memory=True,
            ).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)

    site_ref_u: Any | None = None
    site_key_order: list[tuple[str, int, str]] | None = None
    ref_site_res: set[int] | None = None
    if cfg.pca.site_from_reference_ligand:
        site_ref_path = cfg.pca.site_reference_pdb or cfg.rmsd.reference
        if site_ref_path is None:
            raise RuntimeError("pca.site_reference_pdb or rmsd.reference is required when pca.site_from_reference_ligand=true")
        if not cfg.pca.site_ligand_selection:
            raise RuntimeError("pca.site_ligand_selection is required when pca.site_from_reference_ligand=true")
        site_ref_u = _imports()[0].Universe(str(site_ref_path))
        ref_site_res = ligand_site_resindices(site_ref_u, cfg.pca.site_ligand_selection, cfg.pca.site_cutoff)
        if not ref_site_res:
            raise RuntimeError("No reference ligand-site residues for PCA site mode")

        atom_idx_by_traj: dict[str, dict[tuple[str, int, str], int]] = {}
        map_rows = []
        for name, u in universes:
            atom_map, strategy, mapped_residue_count, residue_pairs = _build_site_atom_map(
                mobile_universe=u,
                reference_universe=site_ref_u,
                source_label=name,
                align_selection=cfg.pca.site_align_selection,
                map_mode=cfg.pca.site_map_mode,
                atom_selection=cfg.pca.site_atom_selection,
                reference_resindices=ref_site_res,
                map_file=cfg.pca.site_map_file,
            )
            if mapped_residue_count < 1:
                raise RuntimeError(f"No mapped ligand-site residues for trajectory: {name}")

            atom_idx_by_traj[name] = atom_map
            map_rows.append(
                {
                    "trajectory": name,
                    "mapping_strategy": strategy,
                    "mapped_site_residues": mapped_residue_count,
                    "mapped_site_atoms": len(atom_map),
                    "used_site_residues": residue_pairs,
                    "status": "mapped" if atom_map else "no_atoms",
                }
            )

        common_keys = set.intersection(*[set(v.keys()) for v in atom_idx_by_traj.values()]) if atom_idx_by_traj else set()
        if len(common_keys) < 3:
            raise RuntimeError(f"PCA site mode found too few common mapped atoms: {len(common_keys)}")
        key_order = sorted(common_keys)
        site_key_order = key_order

        for name, u in universes:
            atom_indices = [atom_idx_by_traj[name][k] for k in key_order]
            if feature_mode == "distance":
                mat, frames = _collect_distance_matrix_from_atom_indices(u, atom_indices, _frame_slice(cfg))
            else:
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
            if feature_mode == "distance":
                mat, frames = _collect_distance_matrix(u, cfg.pca.selection, _frame_slice(cfg))
            else:
                mat, frames = _collect_matrix(u, cfg.pca.selection, _frame_slice(cfg))
            matrices[name] = mat
            frames_map[name] = frames

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

    reference_projection_rows = []
    for idx, pdb_path in enumerate(cfg.pca.reference_pdbs):
        mda, _, _, _ = _imports()
        ref_u = mda.Universe(str(pdb_path))
        ref_name = cfg.pca.reference_names[idx] if idx < len(cfg.pca.reference_names) else pdb_path.stem

        if cfg.pca.align:
            from MDAnalysis.analysis.align import alignto

            align_sel = cfg.pca.site_align_selection if cfg.pca.site_from_reference_ligand else cfg.system.align_selection
            fit_res, ref_res, mapping, _ = build_residue_mapping(
                mobile_universe=fit_u,
                reference_universe=ref_u,
                align_selection=align_sel,
                map_mode="align",
                map_file=None,
            )
            fit_align_idx: list[int] = []
            ref_align_idx: list[int] = []
            seen_fit_idx: set[int] = set()
            seen_ref_idx: set[int] = set()
            for fit_i, ref_i in mapping:
                fit_atoms = fit_res[fit_i].atoms.select_atoms(align_sel)
                ref_atoms = ref_res[ref_i].atoms.select_atoms(align_sel)
                if len(fit_atoms) == 0 or len(ref_atoms) == 0:
                    continue
                _append_unique_atom_pairs(
                    fit_atoms=fit_atoms,
                    ref_atoms=ref_atoms,
                    fit_indices=fit_align_idx,
                    ref_indices=ref_align_idx,
                    seen_fit=seen_fit_idx,
                    seen_ref=seen_ref_idx,
                )
            if len(fit_align_idx) >= 3:
                fit_align_ag = fit_u.atoms[fit_align_idx]
                ref_align_ag = ref_u.atoms[ref_align_idx]
                # Align the single reference structure to the PCA fit trajectory reference frame
                # using the same mapped atom pairs used for correspondence.
                alignto(
                    ref_align_ag,
                    fit_align_ag,
                    select=None,
                    subselection=ref_u.atoms,
                    match_atoms=False,
                )
            else:
                fit_align_ag = None
                ref_align_ag = None
        else:
            fit_align_ag = None
            ref_align_ag = None

        if cfg.pca.site_from_reference_ligand:
            if site_key_order is None:
                raise RuntimeError("Internal error: missing site key order for PCA site mode")
            if ref_site_res is None:
                raise RuntimeError("Internal error: missing site residue set for PCA site mode")
            if site_ref_u is None:
                raise RuntimeError("Internal error: missing site reference universe for PCA site mode")
            ref_atom_map, strategy, mapped_residue_count, residue_pairs = _build_site_atom_map(
                mobile_universe=ref_u,
                reference_universe=site_ref_u,
                source_label=ref_name,
                align_selection=cfg.pca.site_align_selection,
                map_mode=cfg.pca.site_map_mode,
                atom_selection=cfg.pca.site_atom_selection,
                reference_resindices=ref_site_res,
                map_file=cfg.pca.site_map_file,
            )
            missing = [k for k in site_key_order if k not in ref_atom_map]
            if missing:
                print(
                    f"[warn] skipping PCA reference projection for {pdb_path}: "
                    f"could not map {len(missing)} site atoms from the site reference "
                    f"using mode '{cfg.pca.site_map_mode}'."
                )
                reference_projection_rows.append(
                    {
                        "reference_name": ref_name,
                        "reference_pdb": str(pdb_path),
                        "mapping_strategy": strategy,
                        "mapped_site_residues": mapped_residue_count,
                        "mapped_site_atoms": len(ref_atom_map),
                        "missing_site_atoms": len(missing),
                        "used_site_residues": residue_pairs,
                        "status": "skipped",
                    }
                )
                continue
            coords = [ref_u.atoms[ref_atom_map[k]].position.copy() for k in site_key_order]
            ref_coords = np.array(coords)
            if feature_mode == "distance":
                ref_vec = _pairwise_distance_vector(ref_coords).reshape(1, -1)
            else:
                ref_vec = ref_coords.reshape(1, -1)
        else:
            ref_sel = cfg.pca.selection
            ref_atoms = ref_u.select_atoms(ref_sel)
            if feature_mode == "distance":
                if len(ref_atoms) < 2:
                    raise RuntimeError(
                        f"PCA reference selection for distance mode must contain at least 2 atoms: '{ref_sel}'"
                    )
                expected_features = (len(ref_atoms) * (len(ref_atoms) - 1)) // 2
                if expected_features != pca.n_features_in_:
                    raise RuntimeError(
                        f"PCA reference feature mismatch for {pdb_path}: "
                        f"selection '{ref_sel}' gives {len(ref_atoms)} atoms -> {expected_features} pair distances, "
                        f"expected {pca.n_features_in_}. Use a selection matching the PCA fit atoms exactly."
                    )
                ref_vec = _pairwise_distance_vector(ref_atoms.positions).reshape(1, -1)
            else:
                expected_atoms = pca.n_features_in_ // 3
                if len(ref_atoms) != expected_atoms:
                    raise RuntimeError(
                        f"PCA reference atom count mismatch for {pdb_path}: "
                        f"selection '{ref_sel}' gives {len(ref_atoms)} atoms, expected {expected_atoms}. "
                        "Use a selection that matches the PCA fit atoms exactly."
                    )
                ref_vec = ref_atoms.positions.reshape(1, -1)
        ref_sc = pca.transform(ref_vec)
        row = {"trajectory": ref_name, "frame": -1}
        for pc_i in range(ref_sc.shape[1]):
            row[f"PC{pc_i + 1}"] = float(ref_sc[0, pc_i])
        scores_rows.append(row)
        reference_projection_rows.append(
            {
                "reference_name": ref_name,
                "reference_pdb": str(pdb_path),
                "mapping_strategy": strategy if cfg.pca.site_from_reference_ligand else "direct_selection",
                "mapped_site_residues": mapped_residue_count if cfg.pca.site_from_reference_ligand else None,
                "mapped_site_atoms": len(site_key_order) if cfg.pca.site_from_reference_ligand and site_key_order is not None else None,
                "missing_site_atoms": 0 if cfg.pca.site_from_reference_ligand else None,
                "used_site_residues": residue_pairs if cfg.pca.site_from_reference_ligand else None,
                "status": "projected",
            }
        )

    scores = pd.DataFrame(scores_rows)
    scores.to_csv(dirs["tables"] / "pca_scores.csv", index=False)
    eig = pd.DataFrame({"component": [i + 1 for i in range(len(pca.explained_variance_))], "eigenvalue": pca.explained_variance_})
    eig.to_csv(dirs["tables"] / "pca_eigenvalues.csv", index=False)
    (dirs["data"] / "pca_feature_space.json").write_text(
        json.dumps(
            {
                "feature_mode": feature_mode,
                "n_features": int(pca.n_features_in_),
                "n_components_computed": int(getattr(pca, "n_components_", pca.n_components)),
                "site_from_reference_ligand": bool(cfg.pca.site_from_reference_ligand),
            },
            indent=2,
        )
    )
    (dirs["data"] / "pca_reference_projection_report.json").write_text(json.dumps(reference_projection_rows, indent=2))
    if "PC1" not in scores.columns or "PC2" not in scores.columns:
        n_comp = int(getattr(pca, "n_components_", pca.n_components))
        raise RuntimeError(
            "PCA produced fewer than 2 components, so PC1-PC2 plots cannot be generated. "
            f"Computed components: {n_comp}. This often happens when too few frames are available "
            "(e.g., single-frame PDB trajectory) or the feature space rank is too low."
        )
    plot_pca_from_scores(cfg, scores, ctx.outdir)
