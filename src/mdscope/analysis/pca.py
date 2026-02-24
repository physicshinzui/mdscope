from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._common import RunContext, _auto_hist_bins, _frame_slice, _imports, _load_universes, _save_plot, _trajectory_names, ensure_dirs
from .mapping import build_residue_mapping, filter_mapping_to_reference_resindices, ligand_site_resindices


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
    cfg: Any,
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
        ax.scatter(
            refs["PC1"],
            refs["PC2"],
            marker="x",
            s=48,
            linewidths=1.6,
            c="red",
            label="reference",
            zorder=10,
        )
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
    from MDAnalysis.analysis.align import AlignTraj
    from sklearn.decomposition import PCA

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

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
            for fit_i, ref_i in mapping:
                fit_atoms = fit_res[fit_i].atoms.select_atoms(align_sel)
                ref_atoms = ref_res[ref_i].atoms.select_atoms(align_sel)
                if len(fit_atoms) == 0 or len(ref_atoms) == 0:
                    continue
                fit_by_name = {str(a.name): int(a.index) for a in fit_atoms}
                for a in ref_atoms:
                    fit_idx = fit_by_name.get(str(a.name))
                    if fit_idx is None:
                        continue
                    fit_align_idx.append(fit_idx)
                    ref_align_idx.append(int(a.index))
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
            zorder = 10 if marker == "x" else 2
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=size,
                alpha=alpha,
                marker=marker,
                linewidths=lw,
                label=str(tr),
                zorder=zorder,
            )
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
