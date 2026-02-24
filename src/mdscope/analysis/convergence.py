from __future__ import annotations

import json
from typing import Any

from ._common import RunContext, _auto_hist_bins, _block_slices, _imports, _save_plot, ensure_dirs


def _jsd_from_prob(p: Any, q: Any) -> float:
    _, np, _, _ = _imports()
    from scipy.spatial.distance import jensenshannon

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    # Use base=2 so JSD divergence is naturally bounded in [0, 1].
    # scipy.spatial.distance.jensenshannon returns sqrt(JSD); square it to keep
    # the historical "JSD divergence" scale used by existing thresholds.
    js_distance = float(jensenshannon(p, q, base=2.0))
    return float(js_distance * js_distance)


def _jsd_1d(a: Any, b: Any) -> float:
    _, np, _, _ = _imports()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    bins = _auto_hist_bins(pooled, method="fd")
    edges = np.histogram_bin_edges(pooled, bins=bins)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    return _jsd_from_prob(pa, pb)


def _jsd_2d(xa: Any, xb: Any) -> float:
    _, np, _, _ = _imports()
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    if len(xa) == 0 or len(xb) == 0:
        return float("nan")
    x_all = np.concatenate([xa[:, 0], xb[:, 0]])
    y_all = np.concatenate([xa[:, 1], xb[:, 1]])
    xbins = _auto_hist_bins(x_all, method="fd")
    ybins = _auto_hist_bins(y_all, method="fd")
    xedges = np.histogram_bin_edges(x_all, bins=xbins)
    yedges = np.histogram_bin_edges(y_all, bins=ybins)
    ha, _, _ = np.histogram2d(xa[:, 0], xa[:, 1], bins=(xedges, yedges))
    hb, _, _ = np.histogram2d(xb[:, 0], xb[:, 1], bins=(xedges, yedges))
    return _jsd_from_prob(ha.ravel(), hb.ravel())


def run_convergence(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    from itertools import combinations

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    c = cfg.convergence
    summary_rows: list[dict[str, Any]] = []
    metric_status: list[dict[str, Any]] = []
    enabled = set(c.enabled_metrics)

    def add_metric_status(metric: str, within_ok: bool, between_ok: bool, note: str = "") -> None:
        metric_status.append(
            {
                "metric": metric,
                "within_pass": bool(within_ok),
                "between_pass": bool(between_ok),
                "pass": bool(within_ok and between_ok),
                "note": note,
            }
        )

    # 1D metrics: RMSD / Rg
    for metric, table_name, value_col, thr in [
        ("rmsd", "rmsd_vs_reference.csv", "rmsd", c.rmsd),
        ("rg", "rg_timeseries.csv", "rg", c.rg),
    ]:
        if metric not in enabled:
            continue
        path = dirs["tables"] / table_name
        if not path.exists():
            add_metric_status(metric, False, False, f"missing table: {table_name}")
            continue
        df = pd.read_csv(path)
        within_passes: list[bool] = []
        by_traj_values: dict[str, Any] = {}
        for traj, sub in df.groupby("trajectory"):
            sub = sub.sort_values("frame")
            vals = sub[value_col].dropna().to_numpy()
            by_traj_values[str(traj)] = vals
            if len(vals) < c.min_frames:
                within_passes.append(False)
                summary_rows.append(
                    {
                        "metric": metric,
                        "scope": "within",
                        "entity": str(traj),
                        "pass": False,
                        "reason": f"insufficient_frames:{len(vals)}<{c.min_frames}",
                    }
                )
                continue
            spans = _block_slices(len(vals), c.n_blocks)
            blocks = [vals[s:e] for s, e in spans if e > s]
            jsd_vals = [_jsd_1d(blocks[i], blocks[i + 1]) for i in range(max(len(blocks) - 1, 0))]
            jsd_max = float(np.nanmax(jsd_vals)) if len(jsd_vals) > 0 else 0.0
            ok = jsd_max <= thr.jsd_max
            within_passes.append(bool(ok))
            summary_rows.append(
                {
                    "metric": metric,
                    "scope": "within",
                    "entity": str(traj),
                    "pass": bool(ok),
                    "jsd_max_consecutive": jsd_max,
                    "jsd_max": thr.jsd_max,
                }
            )

        between_pass = True
        pair_jsd = []
        traj_names = sorted(by_traj_values.keys())
        if len(traj_names) >= 2:
            for a, b in combinations(traj_names, 2):
                jsd = _jsd_1d(by_traj_values[a], by_traj_values[b])
                pair_jsd.append(jsd)
                summary_rows.append(
                    {
                        "metric": metric,
                        "scope": "between",
                        "entity": f"{a}|{b}",
                        "pass": bool(jsd <= thr.jsd_max),
                        "jsd": float(jsd),
                        "jsd_max": thr.jsd_max,
                    }
                )
            between_pass = bool(np.nanmax(pair_jsd) <= thr.jsd_max) if pair_jsd else True

        add_metric_status(metric, all(within_passes) if within_passes else False, between_pass)

    # PCA metric
    if "pca" in enabled:
        path = dirs["tables"] / "pca_scores.csv"
        if not path.exists():
            add_metric_status("pca", False, False, "missing table: pca_scores.csv")
        else:
            pcs = [f"PC{i}" for i in c.pca.pcs]
            df = pd.read_csv(path)
            df = df[df["frame"] >= 0]
            if not all(col in df.columns for col in pcs[:2]):
                add_metric_status("pca", False, False, f"missing PCA columns: {pcs[:2]}")
            else:
                within_passes = []
                by_traj_xy: dict[str, Any] = {}
                for traj, sub in df.groupby("trajectory"):
                    sub = sub.sort_values("frame")
                    xy = sub[pcs[:2]].to_numpy()
                    by_traj_xy[str(traj)] = xy
                    if len(xy) < c.min_frames:
                        within_passes.append(False)
                        summary_rows.append(
                            {
                                "metric": "pca",
                                "scope": "within",
                                "entity": str(traj),
                                "pass": False,
                                "reason": f"insufficient_frames:{len(xy)}<{c.min_frames}",
                            }
                        )
                        continue
                    spans = _block_slices(len(xy), c.n_blocks)
                    blocks = [xy[s:e] for s, e in spans if e > s]
                    jsd_vals = [_jsd_2d(blocks[i], blocks[i + 1]) for i in range(max(len(blocks) - 1, 0))]
                    jsd_max = float(np.nanmax(jsd_vals)) if len(jsd_vals) > 0 else 0.0
                    ok = jsd_max <= c.pca.jsd_max
                    within_passes.append(bool(ok))
                    summary_rows.append(
                        {
                            "metric": "pca",
                            "scope": "within",
                            "entity": str(traj),
                            "pass": bool(ok),
                            "jsd_max_consecutive": jsd_max,
                            "jsd_max": c.pca.jsd_max,
                        }
                    )
                between_pass = True
                pair_jsd = []
                traj_names = sorted(by_traj_xy.keys())
                if len(traj_names) >= 2:
                    for a, b in combinations(traj_names, 2):
                        jsd = _jsd_2d(by_traj_xy[a], by_traj_xy[b])
                        pair_jsd.append(jsd)
                        summary_rows.append(
                            {
                                "metric": "pca",
                                "scope": "between",
                                "entity": f"{a}|{b}",
                                "pass": bool(jsd <= c.pca.jsd_max),
                                "jsd": float(jsd),
                                "jsd_max": c.pca.jsd_max,
                            }
                        )
                    between_pass = bool(np.nanmax(pair_jsd) <= c.pca.jsd_max) if pair_jsd else True
                add_metric_status("pca", all(within_passes) if within_passes else False, between_pass)

    # Cluster occupancy metric + stacked bars
    if "cluster_occupancy" in enabled:
        path = dirs["tables"] / "hdbscan_labels.csv"
        if not path.exists():
            add_metric_status("cluster_occupancy", False, False, "missing table: hdbscan_labels.csv")
        else:
            df = pd.read_csv(path)
            df = df[df["frame"] >= 0]
            labels_all = sorted(int(v) for v in df["label"].dropna().unique())
            within_passes = []
            traj_vectors: dict[str, Any] = {}
            for traj, sub in df.groupby("trajectory"):
                sub = sub.sort_values("frame")
                labs = sub["label"].to_numpy(dtype=int)
                if len(labs) < c.min_frames:
                    within_passes.append(False)
                    summary_rows.append(
                        {
                            "metric": "cluster_occupancy",
                            "scope": "within",
                            "entity": str(traj),
                            "pass": False,
                            "reason": f"insufficient_frames:{len(labs)}<{c.min_frames}",
                        }
                    )
                    continue
                spans = _block_slices(len(labs), c.n_blocks)
                block_vectors = []
                for s, e in spans:
                    block = labs[s:e]
                    vec = np.array([(block == lab).mean() for lab in labels_all], dtype=float)
                    block_vectors.append(vec)
                traj_vectors[str(traj)] = np.array([(labs == lab).mean() for lab in labels_all], dtype=float)
                jsd_vals = [
                    _jsd_from_prob(block_vectors[i], block_vectors[i + 1])
                    for i in range(max(len(block_vectors) - 1, 0))
                ]
                jsd_max = float(np.nanmax(jsd_vals)) if len(jsd_vals) > 0 else 0.0
                ok = jsd_max <= c.cluster_occupancy.jsd_max
                within_passes.append(bool(ok))
                summary_rows.append(
                    {
                        "metric": "cluster_occupancy",
                        "scope": "within",
                        "entity": str(traj),
                        "pass": bool(ok),
                        "jsd_max_consecutive": jsd_max,
                        "jsd_max": c.cluster_occupancy.jsd_max,
                    }
                )

                # Stacked bar figure per trajectory
                fig, ax = plt.subplots(figsize=(6, 6))
                bottoms = np.zeros(len(block_vectors))
                xs = np.arange(1, len(block_vectors) + 1)
                cmap = plt.get_cmap("tab20")
                block_mat = np.array(block_vectors)
                for li, lab in enumerate(labels_all):
                    vals = block_mat[:, li]
                    label_name = "noise" if lab == -1 else f"cluster_{lab}"
                    ax.bar(xs, vals, bottom=bottoms, color=cmap(li % 20), label=label_name, width=0.7)
                    bottoms += vals
                ax.set_xlabel("block")
                ax.set_ylabel("occupancy")
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f"Cluster Occupancy by Block: {traj}")
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    fontsize=7,
                    ncol=1,
                )
                _save_plot(cfg, fig, dirs["figures"] / f"convergence_cluster_occupancy_blocks_{traj}")
                plt.close(fig)

            between_pass = True
            jsd_pairs = []
            traj_names = sorted(traj_vectors.keys())
            if len(traj_names) >= 2:
                for a, b in combinations(traj_names, 2):
                    jsd = _jsd_from_prob(traj_vectors[a], traj_vectors[b])
                    jsd_pairs.append(jsd)
                    summary_rows.append(
                        {
                            "metric": "cluster_occupancy",
                            "scope": "between",
                            "entity": f"{a}|{b}",
                            "pass": bool(jsd <= c.cluster_occupancy.jsd_max),
                            "jsd": jsd,
                            "jsd_max": c.cluster_occupancy.jsd_max,
                        }
                    )
                between_pass = bool(np.nanmax(jsd_pairs) <= c.cluster_occupancy.jsd_max) if jsd_pairs else True
            add_metric_status("cluster_occupancy", all(within_passes) if within_passes else False, between_pass)

    status_df = pd.DataFrame(metric_status)
    status_df.to_csv(dirs["tables"] / "convergence_metric_status.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(dirs["tables"] / "convergence_summary.csv", index=False)

    evaluated = [row for row in metric_status if row.get("metric")]
    n_pass = sum(1 for row in evaluated if row.get("pass"))
    n_total = len(evaluated)
    if n_total == 0:
        overall = False
    elif c.rule == "all_of":
        overall = n_pass == n_total
    else:
        k = min(max(c.k_required, 1), n_total)
        overall = n_pass >= k

    report = {
        "overall_pass": bool(overall),
        "rule": c.rule,
        "k_required": c.k_required,
        "n_metrics_total": n_total,
        "n_metrics_pass": n_pass,
        "metrics": metric_status,
    }
    (dirs["data"] / "convergence_report.json").write_text(json.dumps(report, indent=2))
