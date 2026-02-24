from __future__ import annotations

from ._common import RunContext, _imports, _save_plot, ensure_dirs


def run_cluster(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    import hdbscan

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    scores_path = dirs["tables"] / "pca_scores.csv"
    if not scores_path.exists():
        raise RuntimeError("pca_scores.csv not found; run pca step first")

    scores = pd.read_csv(scores_path)
    pcs = [f"PC{i}" for i in cfg.pca.use_pcs]
    pcs = [c for c in pcs if c in scores.columns]
    X = scores[pcs].to_numpy()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.cluster.min_cluster_size,
        min_samples=cfg.cluster.min_samples,
        metric=cfg.cluster.metric,
        cluster_selection_method=cfg.cluster.selection_method,
        allow_single_cluster=cfg.cluster.allow_single_cluster,
    )
    labels = clusterer.fit_predict(X)

    out = scores[["trajectory", "frame"]].copy()
    out["label"] = labels
    out["probability"] = getattr(clusterer, "probabilities_", np.zeros(len(out)))

    label_to_name: dict[int, str] = {-1: "noise"}
    used_names: set[str] = {"noise"}
    for label in sorted(int(v) for v in out["label"].unique() if int(v) >= 0):
        members = out[out["label"] == label]
        if len(members) == 0:
            label_to_name[label] = f"cluster_{label}"
            continue
        dominant = str(members["trajectory"].value_counts().idxmax())
        name = dominant
        if name in used_names:
            name = f"{dominant}_{label}"
        used_names.add(name)
        label_to_name[label] = name
    out["cluster_name"] = out["label"].map(lambda x: label_to_name.get(int(x), f"cluster_{int(x)}"))
    out.to_csv(dirs["tables"] / "hdbscan_labels.csv", index=False)

    summary = (
        out[out["label"] >= 0]
        .groupby(["label", "cluster_name"])
        .size()
        .reset_index(name="size")
        .sort_values("label")
    )
    summary.to_csv(dirs["tables"] / "cluster_summary.csv", index=False)

    merged = scores.merge(out[["trajectory", "frame", "label", "cluster_name"]], on=["trajectory", "frame"], how="left")
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    labels_sorted = sorted(int(v) for v in merged["label"].dropna().unique())
    color_by_label = {label: cmap(i % 10) for i, label in enumerate(labels_sorted)}

    normal = merged[merged["frame"] != -1]
    for label, sub in normal.groupby("label"):
        label_i = int(label)
        cluster_name = str(sub["cluster_name"].iloc[0]) if "cluster_name" in sub else ("noise" if label_i == -1 else f"cluster_{label_i}")
        point_size = 6 if label_i == -1 else 10
        point_alpha = 0.35 if label_i == -1 else 0.7
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=point_size,
            alpha=point_alpha,
            marker="o",
            linewidths=0.0,
            color=color_by_label.get(label_i),
            label=cluster_name,
        )

    refs = merged[merged["frame"] == -1]
    for ref_name, sub in refs.groupby("trajectory"):
        label_i = int(sub["label"].iloc[0])
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=48,
            alpha=0.95,
            marker="x",
            linewidths=1.6,
            color=color_by_label.get(label_i),
            label=str(ref_name),
            zorder=10,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    _save_plot(cfg, fig, dirs["figures"] / "pca_scatter_clustered")
    plt.close(fig)
