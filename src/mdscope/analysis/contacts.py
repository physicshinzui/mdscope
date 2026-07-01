from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ._common import RunContext, _frame_slice, _imports, _load_universes, _save_plot, ensure_dirs


def _bh_fdr(p_values: Any) -> Any:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    _, np, _, _ = _imports()
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _welch_p_and_hedges_g(values_a: Any, values_b: Any) -> tuple[float, float]:
    _, np, _, _ = _imports()
    from scipy.stats import ttest_ind

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    if np.allclose(a, a[0]) and np.allclose(b, b[0]):
        p_value = 1.0 if np.isclose(a[0], b[0]) else 0.0
    else:
        p_value = float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return p_value, float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    if pooled_var <= 0:
        effect = 0.0 if np.isclose(a.mean(), b.mean()) else float(np.sign(b.mean() - a.mean()) * np.inf)
    else:
        cohens_d = float((b.mean() - a.mean()) / np.sqrt(pooled_var))
        correction = 1.0 - 3.0 / max(4.0 * (n_a + n_b) - 9.0, 1.0)
        effect = cohens_d * correction
    return p_value, effect


def _residue_metadata(universe: Any, residue_selection: str) -> tuple[list[dict[str, Any]], Any, Any]:
    _, np, _, _ = _imports()
    residues = universe.select_atoms(residue_selection).residues
    if len(residues) < 2:
        raise RuntimeError(f"Contact residue selection contains fewer than two residues: {residue_selection}")

    fragment_ids = [int(res.atoms[0].fragindex) for res in residues]
    fragment_order: dict[int, int] = {}
    chain_positions: dict[int, int] = defaultdict(int)
    metadata: list[dict[str, Any]] = []
    for index, (residue, fragment_id) in enumerate(zip(residues, fragment_ids)):
        if fragment_id not in fragment_order:
            fragment_order[fragment_id] = len(fragment_order) + 1
        chain = fragment_order[fragment_id]
        chain_positions[chain] += 1
        metadata.append(
            {
                "index": index,
                "chain": chain,
                "chain_position": chain_positions[chain],
                "resid": int(residue.resid),
                "resname": str(residue.resname),
                "resindex": int(residue.resindex),
            }
        )
    return metadata, residues, np.asarray(fragment_ids, dtype=int)


def _collect_trajectory_contacts(universe: Any, config: Any) -> tuple[dict[int, int], int, list[dict[str, Any]]]:
    _, np, _, _ = _imports()
    from MDAnalysis.lib.distances import self_capped_distance

    metadata, residues, _ = _residue_metadata(universe, config.contacts.selection)
    residue_atoms = residues.atoms
    atoms = residue_atoms.select_atoms(config.contacts.atom_selection)
    if len(atoms) == 0:
        raise RuntimeError(f"Contact atom selection is empty: {config.contacts.atom_selection}")

    residue_position = {item["resindex"]: item["index"] for item in metadata}
    atom_to_residue = np.asarray([residue_position[int(atom.resindex)] for atom in atoms], dtype=int)
    chain = np.asarray([item["chain"] for item in metadata], dtype=int)
    chain_position = np.asarray([item["chain_position"] for item in metadata], dtype=int)
    n_residues = len(metadata)
    counts: dict[int, int] = defaultdict(int)
    n_frames = 0

    for ts in universe.trajectory[_frame_slice(config)]:
        box = None
        if config.contacts.use_pbc and ts.dimensions is not None and np.all(np.asarray(ts.dimensions[:3]) > 0):
            box = ts.dimensions
        atom_pairs = self_capped_distance(
            atoms.positions,
            max_cutoff=config.contacts.cutoff_angstrom,
            box=box,
            return_distances=False,
        )
        n_frames += 1
        if len(atom_pairs) == 0:
            continue
        left = atom_to_residue[atom_pairs[:, 0]]
        right = atom_to_residue[atom_pairs[:, 1]]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        mask = low != high
        local_neighbors = (chain[low] == chain[high]) & (
            np.abs(chain_position[low] - chain_position[high]) <= config.contacts.min_sequence_separation
        )
        mask &= ~local_neighbors
        if not np.any(mask):
            continue
        codes = np.unique(low[mask] * n_residues + high[mask])
        for code in codes:
            counts[int(code)] += 1

    if n_frames == 0:
        raise RuntimeError("No frames selected for contact analysis")
    return dict(counts), n_frames, metadata


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "group"


def _residue_label(item: dict[str, Any]) -> str:
    return f"C{item['chain']}:{item['resid']} {item['resname']}"


def _plot_contact_matrix(config: Any, matrix: Any, metadata: list[dict[str, Any]], title: str, out_prefix: Path, difference: bool) -> None:
    _, np, _, plt = _imports()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if difference:
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
        vmax = max(vmax, 1e-6)
        image = ax.imshow(matrix, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
        cbar_label = "occupancy difference"
    else:
        image = ax.imshow(matrix, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        cbar_label = "mean contact occupancy"
    fig.colorbar(image, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    tick_indices = sorted(set([0, len(metadata) - 1] + list(range(24, len(metadata), 25))))
    tick_labels = [f"C{metadata[i]['chain']}:{metadata[i]['resid']}" for i in tick_indices]
    ax.set_xticks(tick_indices, tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_indices, tick_labels, fontsize=7)
    boundaries = [i - 0.5 for i in range(1, len(metadata)) if metadata[i]["chain"] != metadata[i - 1]["chain"]]
    for boundary in boundaries:
        ax.axhline(boundary, color="white", lw=0.8, alpha=0.8)
        ax.axvline(boundary, color="white", lw=0.8, alpha=0.8)
    ax.set_xlabel("residue")
    ax.set_ylabel("residue")
    ax.set_title(title)
    _save_plot(config, fig, out_prefix)
    plt.close(fig)


def plot_contacts_from_tables(config: Any, outdir: Path) -> list[str]:
    _, np, pd, plt = _imports()
    dirs = ensure_dirs(outdir)
    comparison = config.contacts.comparison
    if comparison is None:
        raise RuntimeError("contacts.comparison is required")
    table = pd.read_csv(dirs["tables"] / "contact_comparison.csv")
    ranking = pd.read_csv(dirs["tables"] / "contact_ranking.csv")
    hotspots = pd.read_csv(dirs["tables"] / "contact_residue_hotspots.csv")
    if len(table) == 0:
        return []

    n_residues = int(max(table["residue_i_index"].max(), table["residue_j_index"].max()))
    metadata_map: dict[int, dict[str, Any]] = {}
    for _, row in table.iterrows():
        for side in ("i", "j"):
            index = int(row[f"residue_{side}_index"]) - 1
            metadata_map[index] = {
                "index": index,
                "chain": int(row[f"residue_{side}_chain"]),
                "resid": int(row[f"residue_{side}_resid"]),
                "resname": str(row[f"residue_{side}_resname_a"]),
            }
    metadata = [metadata_map.get(i, {"index": i, "chain": 0, "resid": i + 1, "resname": "UNK"}) for i in range(n_residues)]

    matrices = {
        comparison.group_a: np.zeros((n_residues, n_residues), dtype=float),
        comparison.group_b: np.zeros((n_residues, n_residues), dtype=float),
        "difference": np.zeros((n_residues, n_residues), dtype=float),
    }
    for _, row in table.iterrows():
        i = int(row["residue_i_index"]) - 1
        j = int(row["residue_j_index"]) - 1
        values = {
            comparison.group_a: float(row["mean_occupancy_a"]),
            comparison.group_b: float(row["mean_occupancy_b"]),
            "difference": float(row["difference_b_minus_a"]),
        }
        for key, value in values.items():
            matrices[key][i, j] = value
            matrices[key][j, i] = value

    rendered: list[str] = []
    for group in (comparison.group_a, comparison.group_b):
        stem = f"contact_map_mean_{_safe_stem(group)}"
        _plot_contact_matrix(config, matrices[group], metadata, f"Mean residue contacts: {group}", dirs["figures"] / stem, False)
        rendered.append(stem)
    diff_stem = f"contact_map_difference_{_safe_stem(comparison.group_b)}_minus_{_safe_stem(comparison.group_a)}"
    _plot_contact_matrix(
        config,
        matrices["difference"],
        metadata,
        f"Contact difference: {comparison.group_b} - {comparison.group_a}",
        dirs["figures"] / diff_stem,
        True,
    )
    rendered.append(diff_stem)

    top = ranking.head(config.contacts.top_n).iloc[::-1]
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.28 * len(top))))
        colors = ["#b2182b" if value > 0 else "#2166ac" for value in top["difference_b_minus_a"]]
        ax.barh(top["contact_label_a"], top["difference_b_minus_a"], color=colors)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_xlabel(f"occupancy difference ({comparison.group_b} - {comparison.group_a})")
        ax.set_ylabel("residue contact")
        ax.set_title(f"Top {len(top)} contact differences")
        _save_plot(config, fig, dirs["figures"] / "contact_top_differences")
        plt.close(fig)
        rendered.append("contact_top_differences")

    top_hotspots = hotspots.head(config.contacts.top_n).iloc[::-1]
    if len(top_hotspots) > 0:
        fig, ax = plt.subplots(figsize=(7.2, max(4.5, 0.25 * len(top_hotspots))))
        ax.barh(top_hotspots["residue_label_a"], top_hotspots["sum_abs_difference"], color="#6a51a3")
        ax.set_xlabel("sum of absolute incident contact differences")
        ax.set_ylabel("residue")
        ax.set_title("Residue contact-change hotspots")
        _save_plot(config, fig, dirs["figures"] / "contact_residue_hotspots")
        plt.close(fig)
        rendered.append("contact_residue_hotspots")
    return rendered


def run_contacts(ctx: RunContext) -> None:
    _, np, pd, _ = _imports()
    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    comparison = cfg.contacts.comparison
    if comparison is None:
        raise RuntimeError("contacts.comparison is required")

    requested_names = set(comparison.trajectories_a) | set(comparison.trajectories_b)
    trajectory_counts: dict[str, dict[int, int]] = {}
    trajectory_frames: dict[str, int] = {}
    trajectory_metadata: dict[str, list[dict[str, Any]]] = {}
    n_residues: int | None = None
    position_signature: list[tuple[int, int]] | None = None

    for trajectory_name, universe in _load_universes(cfg):
        if trajectory_name not in requested_names:
            continue
        counts, n_frames, metadata = _collect_trajectory_contacts(universe, cfg)
        signature = [(item["chain"], item["resid"]) for item in metadata]
        if position_signature is None:
            position_signature = signature
            n_residues = len(metadata)
        elif signature != position_signature:
            raise RuntimeError(f"Contact residue positions do not match reference trajectory: {trajectory_name}")
        trajectory_counts[trajectory_name] = counts
        trajectory_frames[trajectory_name] = n_frames
        trajectory_metadata[trajectory_name] = metadata

    missing = sorted(requested_names - set(trajectory_counts))
    if missing:
        raise RuntimeError(f"Contact trajectories were not loaded: {', '.join(missing)}")
    if n_residues is None:
        raise RuntimeError("No trajectories selected for contact analysis")

    all_codes = sorted(set().union(*[set(counts) for counts in trajectory_counts.values()]))
    occupancy_rows: list[dict[str, Any]] = []
    group_by_trajectory = {name: comparison.group_a for name in comparison.trajectories_a}
    group_by_trajectory.update({name: comparison.group_b for name in comparison.trajectories_b})
    for trajectory_name in list(comparison.trajectories_a) + list(comparison.trajectories_b):
        metadata = trajectory_metadata[trajectory_name]
        frames = trajectory_frames[trajectory_name]
        counts = trajectory_counts[trajectory_name]
        for code in all_codes:
            i, j = divmod(code, n_residues)
            occupancy_rows.append(
                {
                    "trajectory": trajectory_name,
                    "group": group_by_trajectory[trajectory_name],
                    "residue_i_index": i + 1,
                    "residue_j_index": j + 1,
                    "residue_i_chain": metadata[i]["chain"],
                    "residue_j_chain": metadata[j]["chain"],
                    "residue_i_resid": metadata[i]["resid"],
                    "residue_j_resid": metadata[j]["resid"],
                    "residue_i_resname": metadata[i]["resname"],
                    "residue_j_resname": metadata[j]["resname"],
                    "frames_in_contact": int(counts.get(code, 0)),
                    "frames_analyzed": frames,
                    "occupancy": float(counts.get(code, 0) / frames),
                }
            )
    occupancy_df = pd.DataFrame(occupancy_rows)
    occupancy_df.to_csv(dirs["tables"] / "contact_occupancy_per_trajectory.csv", index=False)

    metadata_a = trajectory_metadata[comparison.trajectories_a[0]]
    metadata_b = trajectory_metadata[comparison.trajectories_b[0]]
    comparison_rows: list[dict[str, Any]] = []
    for code in all_codes:
        i, j = divmod(code, n_residues)
        values_a = np.asarray([trajectory_counts[name].get(code, 0) / trajectory_frames[name] for name in comparison.trajectories_a])
        values_b = np.asarray([trajectory_counts[name].get(code, 0) / trajectory_frames[name] for name in comparison.trajectories_b])
        p_value, hedges_g = _welch_p_and_hedges_g(values_a, values_b)
        mean_a = float(values_a.mean())
        mean_b = float(values_b.mean())
        difference = mean_b - mean_a
        comparison_rows.append(
            {
                "residue_i_index": i + 1,
                "residue_j_index": j + 1,
                "residue_i_chain": metadata_a[i]["chain"],
                "residue_j_chain": metadata_a[j]["chain"],
                "residue_i_resid": metadata_a[i]["resid"],
                "residue_j_resid": metadata_a[j]["resid"],
                "residue_i_resname_a": metadata_a[i]["resname"],
                "residue_j_resname_a": metadata_a[j]["resname"],
                "residue_i_resname_b": metadata_b[i]["resname"],
                "residue_j_resname_b": metadata_b[j]["resname"],
                "contact_label_a": f"{_residue_label(metadata_a[i])} -- {_residue_label(metadata_a[j])}",
                "contact_label_b": f"{_residue_label(metadata_b[i])} -- {_residue_label(metadata_b[j])}",
                "mean_occupancy_a": mean_a,
                "sd_occupancy_a": float(values_a.std(ddof=1)) if len(values_a) > 1 else float("nan"),
                "mean_occupancy_b": mean_b,
                "sd_occupancy_b": float(values_b.std(ddof=1)) if len(values_b) > 1 else float("nan"),
                "difference_b_minus_a": difference,
                "abs_difference": abs(difference),
                "welch_p_value": p_value,
                "hedges_g": hedges_g,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["fdr_q_value"] = _bh_fdr(comparison_df["welch_p_value"].to_numpy())
    comparison_df = comparison_df.sort_values(["abs_difference", "fdr_q_value"], ascending=[False, True]).reset_index(drop=True)
    comparison_df["rank"] = np.arange(1, len(comparison_df) + 1)
    comparison_df.to_csv(dirs["tables"] / "contact_comparison.csv", index=False)

    eligible = comparison_df[
        (comparison_df[["mean_occupancy_a", "mean_occupancy_b"]].max(axis=1) >= cfg.contacts.minimum_mean_occupancy)
        & (comparison_df["abs_difference"] >= cfg.contacts.minimum_abs_difference)
    ].copy()
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    eligible.to_csv(dirs["tables"] / "contact_ranking.csv", index=False)

    hotspot_rows: list[dict[str, Any]] = []
    for residue_index in range(1, n_residues + 1):
        incident = eligible[(eligible["residue_i_index"] == residue_index) | (eligible["residue_j_index"] == residue_index)]
        item_a = metadata_a[residue_index - 1]
        item_b = metadata_b[residue_index - 1]
        hotspot_rows.append(
            {
                "residue_index": residue_index,
                "chain": item_a["chain"],
                "resid": item_a["resid"],
                "resname_a": item_a["resname"],
                "resname_b": item_b["resname"],
                "residue_label_a": _residue_label(item_a),
                "residue_label_b": _residue_label(item_b),
                "sum_abs_difference": float(incident["abs_difference"].sum()),
                "net_difference_b_minus_a": float(incident["difference_b_minus_a"].sum()),
                "max_abs_difference": float(incident["abs_difference"].max()) if len(incident) else 0.0,
                "n_ranked_contacts": int(len(incident)),
            }
        )
    hotspot_df = pd.DataFrame(hotspot_rows).sort_values("sum_abs_difference", ascending=False).reset_index(drop=True)
    hotspot_df["rank"] = np.arange(1, len(hotspot_df) + 1)
    hotspot_df.to_csv(dirs["tables"] / "contact_residue_hotspots.csv", index=False)

    metadata_payload = {
        "selection": cfg.contacts.selection,
        "atom_selection": cfg.contacts.atom_selection,
        "cutoff_angstrom": cfg.contacts.cutoff_angstrom,
        "min_sequence_separation": cfg.contacts.min_sequence_separation,
        "use_pbc": cfg.contacts.use_pbc,
        "comparison": comparison.model_dump(),
        "n_residues": n_residues,
        "n_contact_pairs_observed": len(all_codes),
        "n_contact_pairs_ranked": len(eligible),
        "frames_per_trajectory": trajectory_frames,
        "difference_definition": f"{comparison.group_b} minus {comparison.group_a}",
        "statistical_unit": "trajectory",
        "statistical_test": "Welch independent-samples t-test with Benjamini-Hochberg FDR",
    }
    (dirs["data"] / "contact_analysis_metadata.json").write_text(json.dumps(metadata_payload, indent=2))
    plot_contacts_from_tables(cfg, ctx.outdir)
