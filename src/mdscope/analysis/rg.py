from __future__ import annotations

from ._common import RunContext, _frame_slice, _imports, _load_universes, _plot_timeseries_and_distribution, ensure_dirs


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
