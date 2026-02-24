from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ._common import (
    RunContext,
    _auto_hist_bins,
    _block_slices,
    _frame_slice,
    _imports,
    _load_universes,
    _plot_timeseries_and_distribution,
    _save_plot,
    ensure_dirs,
)


def _water_region_atoms(u: Any, cfg: AppConfig) -> Any:
    if cfg.water.region_mode != "selection":
        raise RuntimeError(
            "water.region_mode=ligand_site is not implemented in phase 1; use water.region_mode=selection"
        )
    ag = u.select_atoms(cfg.water.region_selection)
    if len(ag) == 0:
        raise RuntimeError(f"Water region selection is empty: {cfg.water.region_selection!r}")
    return ag


def _water_oxygen_atoms(u: Any, cfg: AppConfig) -> tuple[Any, dict[str, int]]:
    water_ag = u.select_atoms(cfg.water.water_selection)
    if len(water_ag) == 0:
        raise RuntimeError(
            "No atoms matched water.water_selection. "
            f"selection={cfg.water.water_selection!r}. "
            "Check water residue names (e.g., HOH/WAT/SOL/TIP3)."
        )
    oxygen_ag = water_ag.select_atoms(cfg.water.oxygen_selection)
    if len(oxygen_ag) == 0:
        name_counts: dict[str, int] = {}
        for atom in water_ag:
            key = str(atom.name)
            name_counts[key] = name_counts.get(key, 0) + 1
        preview = ", ".join([f"{k}:{v}" for k, v in sorted(name_counts.items())[:12]])
        raise RuntimeError(
            "No atoms matched water.oxygen_selection within selected waters. "
            f"water.oxygen_selection={cfg.water.oxygen_selection!r}; "
            f"example water atom names: {preview}"
        )
    oxygen_name_counts: dict[str, int] = {}
    for atom in oxygen_ag:
        key = str(atom.name)
        oxygen_name_counts[key] = oxygen_name_counts.get(key, 0) + 1
    return oxygen_ag, oxygen_name_counts


def _nearby_water_oxygen_indices(water_ox_ag: Any, region_ag: Any, cutoff: float) -> list[int]:
    from MDAnalysis.lib.distances import capped_distance

    if len(water_ox_ag) == 0 or len(region_ag) == 0:
        return []
    pairs = capped_distance(
        water_ox_ag.positions,
        region_ag.positions,
        max_cutoff=float(cutoff),
        box=getattr(region_ag, "dimensions", None),
        return_distances=False,
    )
    if pairs is None or len(pairs) == 0:
        return []
    return sorted({int(i) for i in pairs[:, 0]})


def _plot_water_rdf(cfg: AppConfig, df: Any, out_prefix: Path) -> None:
    _, _, _, plt = _imports()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for traj_name, sub in df.groupby("trajectory"):
        ax.plot(sub["r_angstrom"], sub["g_r"], lw=1.4, label=str(traj_name))
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.legend(loc="best")
    _save_plot(cfg, fig, out_prefix)
    plt.close(fig)


def _plot_water_residence_distribution(cfg: AppConfig, df: Any, out_prefix: Path) -> None:
    _, np, _, plt = _imports()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for traj_name, sub in df.groupby("trajectory"):
        values = sub["duration_frames"].dropna().to_numpy()
        if len(values) == 0:
            continue
        bins = _auto_hist_bins(values, method="fd")
        ax.hist(values, bins=bins, alpha=0.35, density=True, label=str(traj_name))
        if len(values) > 1 and np.std(values) > 0:
            xs = np.linspace(values.min(), values.max(), 200)
            bw = max(1.06 * np.std(values) * (len(values) ** (-1.0 / 5.0)), 1e-6)
            kernel = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2)
            density = kernel.sum(axis=1) / (len(values) * bw * np.sqrt(2.0 * np.pi))
            ax.plot(xs, density, lw=1.2)
    ax.set_xlabel("Residence duration (frames)")
    ax.set_ylabel("density")
    ax.legend(loc="best")
    _save_plot(cfg, fig, out_prefix)
    plt.close(fig)


def run_water(ctx: RunContext) -> None:
    _, np, pd, _ = _imports()
    from MDAnalysis.analysis.rdf import InterRDF

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    wcfg = cfg.water
    enabled = set(wcfg.enabled_metrics)
    report: dict[str, Any] = {
        "enabled_metrics": sorted(enabled),
        "region_mode": wcfg.region_mode,
        "region_selection": wcfg.region_selection,
        "water_selection": wcfg.water_selection,
        "oxygen_selection": wcfg.oxygen_selection,
        "cutoff_angstrom": float(wcfg.cutoff_angstrom),
        "trajectories": [],
        "notes": [],
    }

    occupancy_rows: list[dict[str, Any]] = []
    rdf_rows: list[dict[str, Any]] = []
    residence_event_rows: list[dict[str, Any]] = []
    for traj_name, u in _load_universes(cfg):
        traj_info: dict[str, Any] = {"trajectory": traj_name}
        water_ox_ag, oxygen_name_counts = _water_oxygen_atoms(u, cfg)
        traj_info["water_oxygen_atoms"] = int(len(water_ox_ag))
        traj_info["water_oxygen_names"] = oxygen_name_counts

        if "occupancy" in enabled or "residence" in enabled:
            region_ag = _water_region_atoms(u, cfg)
            traj_info["region_atoms"] = int(len(region_ag))
            active: dict[int, int] = {}
            total_frames = 0
            last_frame_seen: int | None = None
            for ts in u.trajectory[_frame_slice(cfg)]:
                near_idx = _nearby_water_oxygen_indices(water_ox_ag, region_ag, wcfg.cutoff_angstrom)
                present_resids = {
                    int(water_ox_ag[int(i)].resid)
                    for i in near_idx
                }
                if wcfg.occupancy.count_mode == "oxygen_atoms":
                    water_count = int(len(near_idx))
                else:
                    water_count = int(len(present_resids))
                frame = int(ts.frame)
                last_frame_seen = frame
                total_frames += 1
                if "occupancy" in enabled:
                    occupancy_rows.append(
                        {
                            "trajectory": traj_name,
                            "frame": frame,
                            "water_count": water_count,
                        }
                    )
                if "residence" in enabled:
                    if int(wcfg.residence.gap_tolerance_frames) != 0:
                        raise RuntimeError("water.residence.gap_tolerance_frames > 0 is not implemented in phase 1")
                    current_keys = set(active.keys())
                    for wid in present_resids - current_keys:
                        active[int(wid)] = frame
                    for wid in current_keys - present_resids:
                        start_frame = int(active.pop(int(wid)))
                        duration_frame_span = frame - start_frame
                        # Number of sampled frames in the contiguous event, consistent with frame slicing.
                        duration_sampled = int(max(duration_frame_span // max(cfg.frames.step, 1), 0) + 1)
                        if duration_sampled >= int(wcfg.residence.min_stay_frames):
                            residence_event_rows.append(
                                {
                                    "trajectory": traj_name,
                                    "water_id": int(wid),
                                    "start_frame": start_frame,
                                    "end_frame": frame - 1,
                                    "duration_frames": int(duration_sampled),
                                    "duration_frame_span": int(max((frame - 1) - start_frame, 0)),
                                }
                            )
            if "residence" in enabled:
                # Close events at end of slice using inclusive end frame for reporting.
                final_frame = last_frame_seen
                if final_frame is not None:
                    for wid, start_frame in list(active.items()):
                        duration_frame_span = int(final_frame) - int(start_frame)
                        duration_sampled = int(max(duration_frame_span // max(cfg.frames.step, 1), 0) + 1)
                        if duration_sampled >= int(wcfg.residence.min_stay_frames):
                            residence_event_rows.append(
                                {
                                    "trajectory": traj_name,
                                    "water_id": int(wid),
                                    "start_frame": int(start_frame),
                                    "end_frame": int(final_frame),
                                    "duration_frames": int(duration_sampled),
                                    "duration_frame_span": int(duration_frame_span),
                                }
                            )
                traj_info["occupancy_residence_frames"] = int(total_frames)

        if "rdf" in enabled:
            solute_ag = u.select_atoms(wcfg.rdf.solute_selection)
            if len(solute_ag) == 0:
                raise RuntimeError(f"water.rdf.solute_selection is empty for {traj_name}: {wcfg.rdf.solute_selection!r}")
            rmin, rmax = [float(x) for x in wcfg.rdf.range_angstrom]
            rdf_obj = InterRDF(solute_ag, water_ox_ag, range=(rmin, rmax), nbins=int(wcfg.rdf.nbins)).run(
                start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step
            )
            bins = getattr(rdf_obj.results, "bins", None)
            rdf_vals = getattr(rdf_obj.results, "rdf", None)
            if bins is None:
                bins = getattr(rdf_obj, "bins", None)
            if rdf_vals is None:
                rdf_vals = getattr(rdf_obj, "rdf", None)
            if bins is not None and rdf_vals is not None:
                for r, g in zip(bins, rdf_vals):
                    rdf_rows.append({"trajectory": traj_name, "r_angstrom": float(r), "g_r": float(g)})
            traj_info["rdf_solute_atoms"] = int(len(solute_ag))

        report["trajectories"].append(traj_info)

    if "occupancy" in enabled:
        occ_df = pd.DataFrame(occupancy_rows)
        occ_df.to_csv(dirs["tables"] / "water_occupancy_timeseries.csv", index=False)
        if len(occ_df) > 0:
            summary = (
                occ_df.groupby("trajectory")["water_count"]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .reset_index()
                .rename(columns={"count": "n_frames", "std": "std_count"})
            )
            for q_label, q_val in [("p10", 0.10), ("p90", 0.90)]:
                q_series = occ_df.groupby("trajectory")["water_count"].quantile(q_val)
                summary[q_label] = summary["trajectory"].map(q_series.to_dict())
            summary = summary.rename(columns={"mean": "mean_count", "median": "median_count", "min": "min_count", "max": "max_count"})
            summary.to_csv(dirs["tables"] / "water_occupancy_summary.csv", index=False)

            if bool(wcfg.occupancy.block_summary):
                block_rows: list[dict[str, Any]] = []
                n_blocks = max(int(cfg.convergence.n_blocks), 1)
                for traj_name, sub in occ_df.groupby("trajectory"):
                    vals = sub.sort_values("frame").reset_index(drop=True)
                    spans = _block_slices(len(vals), n_blocks)
                    for i, (lo, hi) in enumerate(spans, start=1):
                        chunk = vals.iloc[lo:hi]
                        if len(chunk) == 0:
                            continue
                        block_rows.append(
                            {
                                "trajectory": traj_name,
                                "block": int(i),
                                "n_frames": int(len(chunk)),
                                "frame_start": int(chunk["frame"].iloc[0]),
                                "frame_end": int(chunk["frame"].iloc[-1]),
                                "mean_count": float(chunk["water_count"].mean()),
                                "std_count": float(chunk["water_count"].std(ddof=1)) if len(chunk) > 1 else 0.0,
                                "median_count": float(chunk["water_count"].median()),
                            }
                        )
                pd.DataFrame(block_rows).to_csv(dirs["tables"] / "water_occupancy_blocks.csv", index=False)

            _plot_timeseries_and_distribution(
                cfg,
                occ_df,
                "frame",
                "water_count",
                "trajectory",
                "water_occupancy",
                "Nearby water count",
            )
        else:
            pd.DataFrame(
                columns=["trajectory", "n_frames", "mean_count", "std_count", "median_count", "min_count", "max_count", "p10", "p90"]
            ).to_csv(dirs["tables"] / "water_occupancy_summary.csv", index=False)

    if "rdf" in enabled:
        rdf_df = pd.DataFrame(rdf_rows)
        rdf_df.to_csv(dirs["tables"] / "water_rdf.csv", index=False)
        if len(rdf_df) > 0:
            _plot_water_rdf(cfg, rdf_df, dirs["figures"] / "water_rdf")

    if "residence" in enabled:
        events_df = pd.DataFrame(
            residence_event_rows,
            columns=[
                "trajectory",
                "water_id",
                "start_frame",
                "end_frame",
                "duration_frames",
                "duration_frame_span",
            ],
        )
        events_df.to_csv(dirs["tables"] / "water_residence_events.csv", index=False)
        if len(events_df) > 0:
            summary = (
                events_df.groupby("trajectory")["duration_frames"]
                .agg(["count", "mean", "median", "max"])
                .reset_index()
                .rename(columns={"count": "n_events", "mean": "mean_duration", "median": "median_duration", "max": "max_duration"})
            )
            summary.to_csv(dirs["tables"] / "water_residence_summary.csv", index=False)

            totals = (
                events_df.groupby(["trajectory", "water_id"])["duration_frames"]
                .sum()
                .reset_index(name="total_duration_frames")
            )
            top_n = max(int(wcfg.residence.report_top_waters), 0)
            if top_n > 0:
                top_totals = (
                    totals.sort_values(["trajectory", "total_duration_frames"], ascending=[True, False])
                    .groupby("trajectory", group_keys=False)
                    .head(top_n)
                )
            else:
                top_totals = totals.iloc[0:0]
            top_totals.to_csv(dirs["tables"] / "water_residence_top_waters.csv", index=False)
            _plot_water_residence_distribution(cfg, events_df, dirs["figures"] / "water_residence_duration_distribution")
        else:
            pd.DataFrame(columns=["trajectory", "n_events", "mean_duration", "median_duration", "max_duration"]).to_csv(
                dirs["tables"] / "water_residence_summary.csv", index=False
            )
            pd.DataFrame(columns=["trajectory", "water_id", "total_duration_frames"]).to_csv(
                dirs["tables"] / "water_residence_top_waters.csv", index=False
            )

    (dirs["data"] / "water_report.json").write_text(json.dumps(report, indent=2))
