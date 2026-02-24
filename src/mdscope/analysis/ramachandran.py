from __future__ import annotations

from ._common import RunContext, _imports, _load_universes, _save_plot, ensure_dirs


def run_ramachandran(ctx: RunContext) -> None:
    _, _, pd, plt = _imports()
    from MDAnalysis.analysis.dihedrals import Ramachandran

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    rows = []
    note_rows = []

    residue_filter: set[tuple[str, int]] | None = None
    if cfg.ramachandran.residues:
        residue_filter = set()
        for token in cfg.ramachandran.residues:
            text = str(token).strip()
            if ":" in text:
                chain, resid_text = text.split(":", 1)
                residue_filter.add((chain.strip(), int(resid_text.strip())))
            else:
                residue_filter.add(("", int(text)))

    for traj_name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.ramachandran.selection)
        input_residues = list(ag.residues)
        rama = Ramachandran(ag).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        angles = rama.results.angles
        # Use Ramachandran internal residue list (ag2) to keep exact index correspondence
        # with results.angles; this excludes residues that cannot define phi/psi.
        residues = list(rama.ag2.residues)
        input_keys = {(str(r.segid).strip(), int(r.resid), str(r.resname)) for r in input_residues}
        used_keys = {(str(r.segid).strip(), int(r.resid), str(r.resname)) for r in residues}
        excluded = sorted(input_keys - used_keys, key=lambda x: (x[0], x[1], x[2]))
        excluded_preview = ";".join([f"{c}:{rid}:{rn}" for c, rid, rn in excluded[:25]])
        note_rows.append(
            {
                "trajectory": traj_name,
                "input_residue_count": len(input_residues),
                "used_residue_count": len(residues),
                "excluded_residue_count": len(excluded),
                "exclusion_note": (
                    "Terminal residues and residues lacking required backbone atoms "
                    "(for phi/psi) are excluded by Ramachandran analysis."
                ),
                "excluded_residues_preview": excluded_preview,
            }
        )
        for frame_i, frame_vals in enumerate(angles):
            for res_i, (phi, psi) in enumerate(frame_vals):
                if res_i >= len(residues):
                    continue
                residue = residues[res_i]
                segid = str(residue.segid).strip()
                resid = int(residue.resid)
                if residue_filter is not None and (segid, resid) not in residue_filter and ("", resid) not in residue_filter:
                    continue
                rows.append(
                    {
                        "trajectory": traj_name,
                        "frame_index": frame_i,
                        "chain": segid,
                        "resid": resid,
                        "resname": residue.resname,
                        "phi": float(phi),
                        "psi": float(psi),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "phi_psi_timeseries.csv", index=False)
    pd.DataFrame(note_rows).to_csv(dirs["tables"] / "ramachandran_notes.csv", index=False)
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

    per_residue_tables_dir = dirs["tables"] / "ramachandran_per_residue"
    per_residue_figures_dir = dirs["figures"] / "ramachandran_per_residue"
    per_residue_tables_dir.mkdir(parents=True, exist_ok=True)
    per_residue_figures_dir.mkdir(parents=True, exist_ok=True)

    for (traj_name, chain, resid, resname), sub in df.groupby(["trajectory", "chain", "resid", "resname"]):
        chain_label = chain if str(chain).strip() else "NA"
        stem = f"{traj_name}_{chain_label}_{int(resid)}_{resname}"
        sub.sort_values("frame_index").to_csv(per_residue_tables_dir / f"{stem}.csv", index=False)

        fig_r, ax_r = plt.subplots(figsize=(6, 6))
        ax_r.scatter(sub["phi"], sub["psi"], s=6, alpha=0.4)
        ax_r.set_xlim(-180, 180)
        ax_r.set_ylim(-180, 180)
        ax_r.set_xlabel("phi")
        ax_r.set_ylabel("psi")
        ax_r.set_title(f"{traj_name} {chain_label}:{int(resid)} {resname}")
        _save_plot(cfg, fig_r, per_residue_figures_dir / stem)
        plt.close(fig_r)
