from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Annotated

import typer

from .config import PRESETS, generate_template, load_config
from .pipeline import run_pipeline
from .plotting import apply_publication_style

app = typer.Typer(help="MD trajectory analysis automation (config-first)")


@app.command("init-config")
def init_config(
    output: Annotated[Path, typer.Option("-o", "--output", help="Destination YAML")] = Path("config.yaml"),
    preset: Annotated[str, typer.Option("--preset", help="quick|standard|full")] = "standard",
) -> None:
    if preset not in PRESETS:
        raise typer.BadParameter(f"Unknown preset: {preset}")
    output.write_text(generate_template(preset=preset))
    typer.echo(f"Wrote template config: {output}")


@app.command("validate-config")
def validate_config(
    config_path: Annotated[Path, typer.Option("-c", "--config", help="Path to YAML config")],
) -> None:
    cfg = load_config(config_path)
    typer.echo(f"Config valid. preset={cfg.preset} outdir={cfg.output.outdir}")


@app.command("list-presets")
def list_presets() -> None:
    for name in PRESETS:
        typer.echo(name)


@app.command("doctor")
def doctor() -> None:
    required = ["typer", "pydantic", "yaml"]
    analysis = ["MDAnalysis", "numpy", "pandas", "sklearn", "Bio", "hdbscan", "mdakit_sasa"]

    typer.echo("[core]")
    for mod in required:
        report_module(mod)

    typer.echo("[analysis]")
    for mod in analysis:
        report_module(mod)


def report_module(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
        typer.echo(f"ok  {module_name}")
    except Exception:
        typer.echo(f"ng  {module_name}")


@app.command("run")
def run(
    config_path: Annotated[Path, typer.Option("-c", "--config", help="Path to YAML config")],
    outdir: Annotated[Path | None, typer.Option("--outdir", help="Override output directory")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from completed steps")] = False,
    force: Annotated[bool, typer.Option("--force", help="Recompute all selected steps")] = False,
    only: Annotated[list[str] | None, typer.Option("--only", help="Run only specific steps")] = None,
    force_step: Annotated[list[str] | None, typer.Option("--force-step", help="Force specific steps")] = None,
    from_step: Annotated[str | None, typer.Option("--from-step", help="Start execution from this step")] = None,
    until_step: Annotated[str | None, typer.Option("--until-step", help="Stop execution at this step")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate and print planned steps only")] = False,
    free_energy: Annotated[
        bool | None,
        typer.Option("--free-energy/--no-free-energy", help="Enable/disable PCA free-energy contour plot"),
    ] = None,
    fe_bins: Annotated[int | None, typer.Option("--fe-bins", help="Bin count for PCA free-energy 2D histogram")] = None,
    fe_bins_auto_fd: Annotated[
        bool,
        typer.Option("--fe-bins-auto-fd", help="Use Freedman-Diaconis auto binning for PCA free-energy"),
    ] = False,
    fe_level_step_rt: Annotated[
        float | None,
        typer.Option("--fe-level-step-rt", help="Contour interval in RT units for PCA free-energy"),
    ] = None,
    fe_smooth_sigma: Annotated[
        float | None,
        typer.Option("--fe-smooth-sigma", help="Gaussian smoothing sigma (in bins) for PCA free-energy"),
    ] = None,
    only_extra: Annotated[
        list[str],
        typer.Argument(
            help="Additional step names. Use with --only to allow: --only rmsd pca cluster",
        ),
    ] = [],
) -> None:
    config_text = config_path.read_text()
    cfg = load_config(config_path)

    if outdir is not None:
        cfg.output.outdir = outdir
    if resume:
        cfg.runtime.resume = True
    if free_energy is not None:
        cfg.pca.free_energy_enabled = free_energy
    if fe_bins is not None:
        if fe_bins < 10 or fe_bins > 400:
            raise typer.BadParameter("--fe-bins must be in [10, 400]")
        cfg.pca.free_energy_bins = fe_bins
    if fe_bins_auto_fd:
        cfg.pca.free_energy_bins = "auto_fd"
    if fe_level_step_rt is not None:
        if fe_level_step_rt <= 0:
            raise typer.BadParameter("--fe-level-step-rt must be > 0")
        cfg.pca.free_energy_level_step_rt = fe_level_step_rt
    if fe_smooth_sigma is not None:
        if fe_smooth_sigma < 0:
            raise typer.BadParameter("--fe-smooth-sigma must be >= 0")
        cfg.pca.free_energy_smooth_sigma = fe_smooth_sigma

    only_set = set(only or [])
    if only_extra:
        if not only_set:
            raise typer.BadParameter("extra step names are only allowed when --only is provided")
        only_set.update(only_extra)
    force_set = set(force_step or [])

    if dry_run:
        typer.echo("Dry run successful. Configuration parsed and validated.")
        typer.echo(f"preset={cfg.preset} outdir={cfg.output.outdir}")
        if only_set:
            typer.echo(f"only={','.join(sorted(only_set))}")
        return

    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache_dir = Path(cfg.output.outdir) / ".cache" / "matplotlib"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

    apply_publication_style(cfg)

    run_pipeline(
        config=cfg,
        config_text=config_text,
        resume=cfg.runtime.resume,
        force=force,
        only=only_set if only_set else None,
        force_steps=force_set,
        from_step=from_step,
        until_step=until_step,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
