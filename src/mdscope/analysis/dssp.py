from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._common import RunContext, _frame_slice, _imports, _load_universes, _save_plot, ensure_dirs


def _dssp_precheck_diagnostics(ag: Any, selection: str) -> dict[str, Any]:
    from collections import Counter

    residue_issues: list[dict[str, Any]] = []
    total_counts = Counter(str(a.name) for a in ag)
    required = ["N", "CA", "C", "O"]

    for res in ag.residues:
        names = [str(a.name) for a in res.atoms]
        core_counts = {name: int(names.count(name)) for name in required}
        if tuple(core_counts[k] for k in required) == (1, 1, 1, 1):
            continue
        issue = {
            "resname": str(res.resname),
            "resid": int(res.resid),
            "chain": (str(getattr(res, "segid", "")).strip() or "A"),
            "required_counts": core_counts,
            "atom_names": names,
        }
        missing = [k for k in required if core_counts[k] == 0]
        if missing:
            issue["missing"] = missing
        extras = [n for n in names if n.startswith("OC")]
        if extras:
            issue["terminal_oxygen_like_atoms"] = sorted(set(extras))
        residue_issues.append(issue)

    return {
        "selection": selection,
        "selected_atoms": int(len(ag)),
        "selected_residues": int(len(ag.residues)),
        "required_name_counts": {k: int(total_counts.get(k, 0)) for k in required},
        "residue_issues": residue_issues,
    }


def _raise_dssp_precheck_error(traj_name: str, diag: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(diag, indent=2))
    counts = diag["required_name_counts"]
    issues = diag["residue_issues"]
    lines = [
        f"DSSP precheck failed for {traj_name}: unequal backbone atom counts in selection '{diag['selection']}'",
        f"Counts: N={counts['N']}, CA={counts['CA']}, C={counts['C']}, O={counts['O']}",
        f"Problem residues (showing up to 5 of {len(issues)}):",
    ]
    for item in issues[:5]:
        missing = ",".join(item.get("missing", [])) if item.get("missing") else "none"
        term = item.get("terminal_oxygen_like_atoms")
        suffix = f"; terminal-like oxygens={term}" if term else ""
        lines.append(
            f"- {item['resname']} {item['resid']} (chain {item['chain']}): "
            f"required={item['required_counts']}, missing={missing}{suffix}"
        )
    lines.extend(
        [
            "Likely cause: terminal or nonstandard residues in DSSP selection (e.g., OC1/OC2 instead of O).",
            "Suggestions:",
            "- Exclude terminal/problem residues in dssp.selection (e.g., 'backbone and not resid <Cterm>').",
            "- Or preprocess topology atom names to standard backbone naming (N, CA, C, O).",
            f"- Diagnostic JSON written to: {out_path}",
        ]
    )
    raise RuntimeError("\n".join(lines))


def run_dssp(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from MDAnalysis.analysis.dssp import DSSP

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config

    # Cleanup legacy combined DSSP plots to avoid mixing old/new naming schemes.
    for fmt in cfg.output.figure_formats:
        legacy_ts = dirs["figures"] / f"dssp_fraction_timeseries.{fmt}"
        if legacy_ts.exists():
            legacy_ts.unlink()
        legacy_res = dirs["figures"] / f"dssp_fraction_per_residue.{fmt}"
        if legacy_res.exists():
            legacy_res.unlink()

    frame_rows = []
    residue_rows = []
    heatmap_payloads: list[tuple[str, Any, list[int]]] = []

    for traj_name, u in _load_universes(cfg):
        ag = u.select_atoms(cfg.dssp.selection)
        if len(ag) == 0:
            continue
        diag = _dssp_precheck_diagnostics(ag, cfg.dssp.selection)
        counts = diag["required_name_counts"]
        if len(diag["residue_issues"]) > 0 or len({counts["N"], counts["CA"], counts["C"], counts["O"]}) != 1:
            _raise_dssp_precheck_error(traj_name, diag, dirs["data"] / f"dssp_selection_diagnostics_{traj_name}.json")

        frame_numbers = [int(ts.frame) for ts in u.trajectory[_frame_slice(cfg)]]
        dssp = DSSP(ag).run(start=cfg.frames.start, stop=cfg.frames.stop, step=cfg.frames.step)
        codes = np.asarray(dssp.results.dssp)
        if codes.ndim != 2 or codes.shape[0] == 0:
            continue
        resids = np.asarray(dssp.results.resids, dtype=int)
        if len(resids) != codes.shape[1]:
            raise RuntimeError(
                f"DSSP residue axis mismatch for {traj_name}: codes={codes.shape[1]} resids={len(resids)}"
            )

        n_frames = min(len(frame_numbers), codes.shape[0])
        frame_numbers = frame_numbers[:n_frames]
        codes = codes[:n_frames, :]

        # Normalize DSSP to 3-state classes used in output: H (helix), S (sheet), C (coil).
        codes_norm = np.full(codes.shape, cfg.dssp.coil_code, dtype="<U1")
        codes_norm[np.isin(codes, ["H", "G", "I"])] = "H"
        codes_norm[np.isin(codes, ["E", "B"])] = "S"
        heatmap_payloads.append((traj_name, codes_norm.copy(), list(resids)))

        for fi in range(n_frames):
            row_codes = codes_norm[fi]
            n_res = len(row_codes)
            if n_res == 0:
                continue
            n_h = int(np.sum(row_codes == "H"))
            n_s = int(np.sum(row_codes == "S"))
            n_c = int(np.sum(row_codes == cfg.dssp.coil_code))
            frame_rows.append(
                {
                    "trajectory": traj_name,
                    "frame": int(frame_numbers[fi]),
                    "frame_index": fi,
                    "helix_fraction": float(n_h / n_res),
                    "sheet_fraction": float(n_s / n_res),
                    "beta_fraction": float(n_s / n_res),
                    "coil_fraction": float(n_c / n_res),
                }
            )

        for ri, resid in enumerate(resids):
            col = codes_norm[:, ri]
            n_obs = len(col)
            if n_obs == 0:
                continue
            residue_rows.append(
                {
                    "trajectory": traj_name,
                    "resid": int(resid),
                    "helix_fraction": float(np.sum(col == "H") / n_obs),
                    "sheet_fraction": float(np.sum(col == "S") / n_obs),
                    "beta_fraction": float(np.sum(col == "S") / n_obs),
                    "coil_fraction": float(np.sum(col == cfg.dssp.coil_code) / n_obs),
                }
            )

    frame_df = pd.DataFrame(frame_rows)
    if len(frame_df) == 0:
        pd.DataFrame([{"status": "skipped", "reason": "No DSSP values produced"}]).to_csv(
            dirs["tables"] / "dssp_status.csv", index=False
        )
        return

    frame_df = frame_df.sort_values(["trajectory", "frame"]).reset_index(drop=True)
    residue_df = pd.DataFrame(residue_rows).sort_values(["trajectory", "resid"]).reset_index(drop=True)
    frame_df.to_csv(dirs["tables"] / "dssp_fraction_timeseries.csv", index=False)
    residue_df.to_csv(dirs["tables"] / "dssp_fraction_per_residue.csv", index=False)

    for traj_name, sub in frame_df.groupby("trajectory"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(sub["frame"], sub["helix_fraction"], lw=1.2, ls="-", label="H")
        ax.plot(sub["frame"], sub["sheet_fraction"], lw=1.2, ls="-", label="S")
        ax.plot(sub["frame"], sub["coil_fraction"], lw=1.2, ls="-", label="C")
        ax.set_xlabel("frame")
        ax.set_ylabel("fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(str(traj_name))
        ax.legend(loc="best")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_fraction_timeseries_{traj_name}")
        plt.close(fig)

    for traj_name, sub in residue_df.groupby("trajectory"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(sub["resid"], sub["helix_fraction"], lw=1.2, ls="-", label="H")
        ax.plot(sub["resid"], sub["sheet_fraction"], lw=1.2, ls="-", label="S")
        ax.plot(sub["resid"], sub["coil_fraction"], lw=1.2, ls="-", label="C")
        ax.set_xlabel("resid")
        ax.set_ylabel("fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(str(traj_name))
        ax.legend(loc="best")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_fraction_per_residue_{traj_name}")
        plt.close(fig)

    state_to_idx = {cfg.dssp.coil_code: 0, "S": 1, "H": 2}
    cmap = ListedColormap(["#d9d9d9", "#4c78a8", "#e45756"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for traj_name, states, resids in heatmap_payloads:
        if states.size == 0:
            continue
        # imshow expects [rows, cols] = [residue, time]
        mat = np.vectorize(lambda x: state_to_idx.get(str(x), 0))(states).T
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xlabel("frame index")
        ax.set_ylabel("resid")
        ax.set_title(str(traj_name))

        n_res = len(resids)
        if n_res > 1:
            tick_positions = np.linspace(0, n_res - 1, num=min(8, n_res), dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(resids[i]) for i in tick_positions])
        elif n_res == 1:
            ax.set_yticks([0])
            ax.set_yticklabels([str(resids[0])])

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels([cfg.dssp.coil_code, "S", "H"])
        cbar.set_label("DSSP class")
        _save_plot(cfg, fig, dirs["figures"] / f"dssp_heatmap_residue_time_{traj_name}")
        plt.close(fig)
