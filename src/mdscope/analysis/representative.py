from __future__ import annotations

from ._common import RunContext, _imports, _load_universes, ensure_dirs


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

    seed = cfg.cluster.representative_random_seed
    rng = np.random.default_rng(seed if seed is not None else cfg.runtime.seed)

    for label, sub in merged.groupby("label"):
        if int(label) < 0:
            continue
        sub = sub[sub["frame"] >= 0].copy()
        if len(sub) == 0:
            continue
        X = sub[pc_cols].to_numpy()
        if len(X) == 0:
            continue
        if cfg.cluster.representative_method == "random":
            n_pick = max(int(cfg.cluster.representative_random_n), 1)
            n_pick = min(n_pick, len(sub))
            picked_idx = rng.choice(len(sub), size=n_pick, replace=False)
            for rank, i in enumerate(np.atleast_1d(picked_idx), start=1):
                row = sub.iloc[int(i)]
                traj_name = str(row["trajectory"])
                frame = int(row["frame"])
                cluster_name = str(row.get("cluster_name", f"cluster_{int(label)}"))
                u = universes[traj_name]
                u.trajectory[frame]
                out_pdb = dirs["representatives"] / f"cluster_{int(label)}_rand_{rank}.pdb"
                u.atoms.write(str(out_pdb))
                representatives.append(
                    {
                        "label": int(label),
                        "cluster_name": cluster_name,
                        "trajectory": traj_name,
                        "frame": frame,
                        "pdb": str(out_pdb),
                        "method": "random",
                        "sample_rank": int(rank),
                        "random_seed": (seed if seed is not None else cfg.runtime.seed),
                    }
                )
        else:
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
                    "method": str(cfg.cluster.representative_method),
                    "sample_rank": 1,
                    "random_seed": "",
                }
            )

    pd.DataFrame(representatives).to_csv(dirs["tables"] / "representatives.csv", index=False)
