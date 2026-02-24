from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._common import RunContext, _auto_hist_bins, _imports, _save_plot, ensure_dirs


def _pocket_collect_inputs(ctx: RunContext) -> list[dict[str, Any]]:
    cfg = ctx.config
    dirs = ensure_dirs(ctx.outdir)
    pcfg = cfg.pocket
    rows: list[dict[str, Any]] = []
    if pcfg.input_mode == "representatives":
        for p in sorted(dirs["representatives"].glob("*.pdb")):
            rows.append({"structure_id": p.stem, "pdb_path": p, "source_type": "representatives"})
    else:
        if pcfg.source_dir is None:
            raise RuntimeError("pocket.source_dir is required when pocket.input_mode=explicit_pdbs")
        for p in sorted(pcfg.source_dir.glob(pcfg.pdb_glob)):
            if p.is_file():
                rows.append({"structure_id": p.stem, "pdb_path": p, "source_type": "explicit_pdbs"})
    if len(rows) == 0:
        raise RuntimeError(f"No pocket input PDBs found for pocket.input_mode={pcfg.input_mode}")
    return rows[: max(1, pcfg.max_structures)]


def _fpocket_available() -> bool:
    import shutil

    return shutil.which("fpocket") is not None


def _find_fpocket_info_file(run_dir: Path, staged_pdb: Path) -> Path | None:
    candidates = [
        run_dir / f"{staged_pdb.stem}_out" / f"{staged_pdb.stem}_info.txt",
        run_dir / f"{staged_pdb.stem}_out" / "pockets_info.txt",
        run_dir / f"{staged_pdb.stem}_out" / f"{staged_pdb.stem}_pockets_info.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    all_infos = sorted(run_dir.rglob("*info*.txt"))
    return all_infos[0] if all_infos else None


def _run_fpocket_command(pdb_path: Path, cwd: Path) -> tuple[int, str, str]:
    import subprocess

    proc = subprocess.run(
        ["fpocket", "-f", str(pdb_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _parse_fpocket_info(info_path: Path, structure_id: str) -> list[dict[str, Any]]:
    import re

    lines = info_path.read_text(errors="ignore").splitlines()
    parsed: list[dict[str, Any]] = []
    pocket: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal pocket
        if pocket is None:
            return
        parsed.append(pocket)
        pocket = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^Pocket\s+(\d+)", line, flags=re.IGNORECASE)
        if m:
            flush()
            pocket = {"structure_id": structure_id, "pocket_id": int(m.group(1))}
            continue
        if pocket is None or ":" not in line:
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        key_norm = key.lower().replace(" ", "_").replace("-", "_").replace(".", "")
        try:
            pocket[key_norm] = float(value.split()[0])
        except Exception:
            pocket[key_norm] = value
    flush()

    rows: list[dict[str, Any]] = []
    def _pick(d: dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None

    for p in parsed:
        rows.append(
            {
                "structure_id": structure_id,
                "pocket_id": int(p.get("pocket_id", -1)),
                "score": p.get("score", p.get("pocket_score")),
                "druggability_score": p.get("druggability_score", p.get("drug_score")),
                "volume": p.get("volume", p.get("volume_(approx)")),
                "alpha_sphere_count": p.get("number_of_alpha_spheres", p.get("nb_asph")),
                "total_sasa": _pick(p, "total_sasa"),
                "polar_sasa": _pick(p, "polar_sasa"),
                "apolar_sasa": _pick(p, "apolar_sasa"),
                "mean_local_hydrophobic_density": _pick(p, "mean_local_hydrophobic_density"),
                "mean_alpha_sphere_radius": _pick(p, "mean_alpha_sphere_radius", "mean_asph_ray"),
                "mean_alpha_sphere_solvent_access": _pick(
                    p,
                    "mean_alp_sph_solvent_access",
                    "mean_alpha_sphere_solvent_access",
                    "masph_sacc",
                ),
                "apolar_alpha_sphere_proportion": _pick(p, "apolar_alpha_sphere_proportion", "apolar_asphere_prop"),
                "hydrophobicity_score": _pick(p, "hydrophobicity_score"),
                "volume_score": _pick(p, "volume_score"),
                "polarity_score": _pick(p, "polarity_score"),
                "charge_score": _pick(p, "charge_score"),
                "proportion_of_polar_atoms": _pick(p, "proportion_of_polar_atoms", "prop_polar_atm"),
                "alpha_sphere_density": _pick(p, "alpha_sphere_density", "as_density"),
                "center_of_mass_alpha_sphere_max_dist": _pick(
                    p,
                    "cent_of_mass___alpha_sphere_max_dist",
                    "cent_of_mass_alpha_sphere_max_dist",
                ),
                "flexibility": _pick(p, "flexibility", "flex"),
            }
        )
    return rows


def run_pocket(ctx: RunContext) -> None:
    _, np, pd, plt = _imports()
    import shutil
    import time

    dirs = ensure_dirs(ctx.outdir)
    cfg = ctx.config
    pcfg = cfg.pocket
    pocket_root = ctx.outdir / pcfg.run_dir
    pocket_root.mkdir(parents=True, exist_ok=True)

    inputs = _pocket_collect_inputs(ctx)
    fpocket_ok = _fpocket_available()
    if not fpocket_ok and pcfg.fail_on_missing_fpocket:
        raise RuntimeError("fpocket executable not found in PATH")

    status_rows: list[dict[str, Any]] = []
    pocket_rows: list[dict[str, Any]] = []

    for item in inputs:
        structure_id = str(item["structure_id"])
        pdb_path = Path(item["pdb_path"])
        source_type = str(item["source_type"])
        run_dir = pocket_root / structure_id
        run_dir.mkdir(parents=True, exist_ok=True)
        staged_pdb = run_dir / pdb_path.name
        if not staged_pdb.exists():
            shutil.copy2(pdb_path, staged_pdb)

        rc = 0
        stdout = ""
        stderr = ""
        dt = 0.0
        info_path = _find_fpocket_info_file(run_dir, staged_pdb)
        if info_path is not None and cfg.runtime.resume:
            status = "skipped"
        elif not fpocket_ok:
            status = "failed"
            rc = 127
            stderr = "fpocket executable not found"
        else:
            t0 = time.time()
            rc, stdout, stderr = _run_fpocket_command(staged_pdb, run_dir)
            dt = float(time.time() - t0)
            info_path = _find_fpocket_info_file(run_dir, staged_pdb)
            status = "ok" if rc == 0 and info_path is not None else "failed"
            if rc == 0 and info_path is None:
                stderr = (stderr + "\n" if stderr else "") + "fpocket output info file not found"

        parsed_n = 0
        parse_error = ""
        if status in {"ok", "skipped"} and info_path is not None:
            try:
                rows = _parse_fpocket_info(info_path, structure_id)
                for r in rows:
                    r["source_type"] = source_type
                    r["pdb_path"] = str(pdb_path)
                    pocket_rows.append(r)
                parsed_n = len(rows)
            except Exception as exc:
                status = "failed"
                parse_error = str(exc)

        status_rows.append(
            {
                "structure_id": structure_id,
                "source_type": source_type,
                "pdb_path": str(pdb_path),
                "run_dir": str(run_dir),
                "status": status,
                "returncode": int(rc),
                "runtime_sec": float(dt),
                "n_pockets_parsed": int(parsed_n),
                "info_path": str(info_path) if info_path else "",
                "stderr_tail": (stderr or "").strip()[-500:],
                "parse_error": parse_error,
            }
        )

    status_df = pd.DataFrame(status_rows)
    pockets_df = pd.DataFrame(pocket_rows)
    status_df.to_csv(dirs["tables"] / "pocket_fpocket_structure_status.csv", index=False)
    pockets_df.to_csv(dirs["tables"] / "pocket_fpocket_summary.csv", index=False)

    top_hits_df = pd.DataFrame()
    if len(pockets_df) > 0:
        rank_col = pcfg.rank_by if pcfg.rank_by in pockets_df.columns else "score"
        work = pockets_df.copy()
        work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
        work = work.sort_values(["structure_id", rank_col], ascending=[True, False], na_position="last")
        top_hits_df = work.groupby("structure_id", group_keys=False).head(max(1, pcfg.top_n_pockets))
    top_hits_df.to_csv(dirs["tables"] / "pocket_fpocket_top_hits.csv", index=False)

    report = {
        "backend": pcfg.backend,
        "input_mode": pcfg.input_mode,
        "source_dir": str(pcfg.source_dir) if pcfg.source_dir else None,
        "pdb_glob": pcfg.pdb_glob,
        "rank_by": pcfg.rank_by,
        "top_n_pockets": int(pcfg.top_n_pockets),
        "n_structures": int(len(inputs)),
        "n_structures_ok": int((status_df["status"] == "ok").sum()) if len(status_df) else 0,
        "n_structures_failed": int((status_df["status"] == "failed").sum()) if len(status_df) else 0,
        "n_pockets_total": int(len(pockets_df)),
        "fpocket_available": bool(fpocket_ok),
        "run_root": str(pocket_root),
    }
    (dirs["data"] / "pocket_fpocket_report.json").write_text(json.dumps(report, indent=2))

    if len(pockets_df) > 0:
        for col, stem, xlabel in [
            ("score", "pocket_fpocket_score_distribution", "fpocket score"),
            ("volume", "pocket_fpocket_volume_distribution", "pocket volume"),
        ]:
            if col not in pockets_df.columns:
                continue
            values = pd.to_numeric(pockets_df[col], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 4.5))
            ax.hist(values, bins=_auto_hist_bins(values, method="fd"), color="#4c78a8", alpha=0.85)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            _save_plot(cfg, fig, dirs["figures"] / stem)
            plt.close(fig)

        rank_col = pcfg.rank_by if pcfg.rank_by in pockets_df.columns else "score"
        vals_df = pockets_df.copy()
        vals_df[rank_col] = pd.to_numeric(vals_df[rank_col], errors="coerce")
        top_by_structure = vals_df.groupby("structure_id")[rank_col].max().dropna()
        if len(top_by_structure) > 0:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            xs = np.arange(len(top_by_structure))
            ax.bar(xs, top_by_structure.to_numpy(), color="#59a14f")
            ax.set_xticks(xs)
            ax.set_xticklabels(top_by_structure.index.tolist(), rotation=70, ha="right", fontsize=8)
            ax.set_xlabel("structure")
            ax.set_ylabel(f"max {rank_col}")
            _save_plot(cfg, fig, dirs["figures"] / "pocket_fpocket_presence_by_structure")
            plt.close(fig)
