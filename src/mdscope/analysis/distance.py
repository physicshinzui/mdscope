from __future__ import annotations

from ._common import RunContext, _frame_slice, _imports, _load_universes, _plot_timeseries_and_distribution, ensure_dirs


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
