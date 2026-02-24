from __future__ import annotations

from ._common import RunContext, _imports, _load_universes, _save_plot, ensure_dirs


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
