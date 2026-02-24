from __future__ import annotations

from ._common import RunContext, _auto_hist_bins, _imports, _load_universes, _plot_timeseries_and_distribution, _save_plot, ensure_dirs, TIEN2013_MAX_ASA


def run_sasa(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

    try:
        from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
    except Exception:
        marker = pd.DataFrame([{"status": "skipped", "reason": "mdakit_sasa is not installed"}])
        marker.to_csv(dirs["tables"] / "sasa_status.csv", index=False)
        return

    rows = []
    residue_rows = []
    missing_maxasa: dict[str, set[str]] = {}
    maxasa_table = TIEN2013_MAX_ASA if cfg.sasa.reference_scale == "tien2013" else TIEN2013_MAX_ASA
    value_col = "rsasa" if cfg.sasa.relative else "sasa"
    value_label = "rSASA" if cfg.sasa.relative else "SASA"
    for traj_name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.sasa.selection)
        residues = list(ag.residues)
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

        residue_area = getattr(sasa.results, "residue_area", None)
        if residue_area is None:
            continue
        residue_area = np.asarray(residue_area)
        n_frames = residue_area.shape[0] if residue_area.ndim == 2 else 0
        if n_frames == 0:
            continue
        n_cols = residue_area.shape[1]
        if n_cols != len(residues):
            raise RuntimeError(
                f"SASA residue-axis mismatch for {traj_name}: residue_area has {n_cols} columns "
                f"but selection has {len(residues)} residues"
            )
        n_use = n_cols
        for frame_i in range(n_frames):
            for ri in range(n_use):
                residue = residues[ri]
                resname = str(residue.resname)
                max_asa = maxasa_table.get(resname)
                rsasa = np.nan
                if max_asa and max_asa > 0:
                    rsasa = float(residue_area[frame_i, ri]) / float(max_asa)
                    if cfg.sasa.rsasa_clip:
                        rsasa = float(np.clip(rsasa, 0.0, 1.0))
                else:
                    missing_maxasa.setdefault(traj_name, set()).add(resname)
                sasa_val = float(residue_area[frame_i, ri])
                residue_rows.append(
                    {
                        "trajectory": traj_name,
                        "frame_index": frame_i,
                        "chain": str(residue.segid).strip(),
                        "resid": int(residue.resid),
                        "resname": resname,
                        "sasa": sasa_val,
                        "rsasa": float(rsasa),
                        "value": float(rsasa) if cfg.sasa.relative else sasa_val,
                        "value_kind": value_col,
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        pd.DataFrame([{"status": "skipped", "reason": "No SASA values produced by backend"}]).to_csv(
            dirs["tables"] / "sasa_status.csv", index=False
        )
        return

    df.to_csv(dirs["tables"] / "sasa_timeseries.csv", index=False)
    _plot_timeseries_and_distribution(cfg, df, "frame_index", "sasa", "trajectory", "sasa", "SASA")

    if residue_rows:
        residue_df = pd.DataFrame(residue_rows)
        residue_df.to_csv(dirs["tables"] / "sasa_per_residue_timeseries.csv", index=False)
        summary = residue_df.groupby(["trajectory", "chain", "resid", "resname"], as_index=False).agg(
            sasa_mean=("sasa", "mean"),
            sasa_std=("sasa", "std"),
            rsasa_mean=("rsasa", "mean"),
            rsasa_std=("rsasa", "std"),
            value_mean=("value", "mean"),
            value_std=("value", "std"),
        )
        summary.to_csv(dirs["tables"] / "sasa_per_residue_summary.csv", index=False)
        if missing_maxasa:
            note_rows = []
            for traj_name in sorted(missing_maxasa):
                for resname in sorted(missing_maxasa[traj_name]):
                    note_rows.append(
                        {
                            "trajectory": traj_name,
                            "resname": resname,
                            "reference_scale": cfg.sasa.reference_scale,
                            "note": "maxASA not found; rsasa set to NaN",
                        }
                    )
            pd.DataFrame(note_rows).to_csv(dirs["tables"] / "sasa_rsasa_notes.csv", index=False)

        per_res_fig_dir = dirs["figures"] / "sasa_per_residue"
        per_res_fig_dir.mkdir(parents=True, exist_ok=True)

        for (traj_name, chain, resid, resname), sub in residue_df.groupby(
            ["trajectory", "chain", "resid", "resname"], sort=True
        ):
            chain_label = chain if str(chain).strip() else "NA"
            stem = f"{traj_name}_{chain_label}_{int(resid)}_{resname}"
            sub = sub.sort_values("frame_index")

            fig_ts, ax_ts = plt.subplots(figsize=(7.2, 4.5))
            ax_ts.plot(sub["frame_index"], sub["value"], lw=1.2, ls="-")
            ax_ts.set_xlabel("frame_index")
            ax_ts.set_ylabel(value_label)
            if cfg.sasa.relative:
                ax_ts.set_ylim(0.0, 1.0)
            ax_ts.set_title(f"{traj_name} {chain_label}:{int(resid)} {resname}")
            _save_plot(cfg, fig_ts, per_res_fig_dir / f"{stem}_timeseries")
            plt.close(fig_ts)

            values = sub["value"].dropna().to_numpy()
            if len(values) == 0:
                continue
            fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
            bins = _auto_hist_bins(values, method=cfg.plot.hist_bin_method)
            ax_dist.hist(values, bins=bins, density=True, alpha=0.35)
            if values.std() > 0:
                xs = np.linspace(values.min(), values.max(), 200)
                bw = 1.06 * values.std() * (len(values) ** (-1.0 / 5.0))
                bw = max(bw, 1e-6)
                kernel = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2)
                density = kernel.sum(axis=1) / (len(values) * bw * np.sqrt(2.0 * np.pi))
                ax_dist.plot(xs, density, lw=1.2)
            ax_dist.set_xlabel(value_label)
            ax_dist.set_ylabel("density")
            if cfg.sasa.relative:
                ax_dist.set_xlim(0.0, 1.0)
            ax_dist.set_title(f"{traj_name} {chain_label}:{int(resid)} {resname}")
            _save_plot(cfg, fig_dist, per_res_fig_dir / f"{stem}_distribution")
            plt.close(fig_dist)
