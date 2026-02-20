from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .analysis.steps import RunContext, STEP_HANDLERS
from .config import AppConfig, StepName
from .state import StateStore, short_config_hash


@dataclass(frozen=True)
class Step:
    name: StepName
    deps: tuple[StepName, ...]
    enabled: Callable[[AppConfig], bool]


STEPS: list[Step] = [
    Step("rmsd", tuple(), lambda c: c.analyses.rmsd and c.rmsd.enabled),
    Step("rmsf", tuple(), lambda c: c.analyses.rmsf and c.rmsf.enabled),
    Step("dssp", tuple(), lambda c: c.analyses.dssp),
    Step("pca", tuple(), lambda c: c.analyses.pca and c.pca.enabled),
    Step("cluster", ("pca",), lambda c: c.analyses.cluster and c.cluster.enabled),
    Step("representative", ("cluster",), lambda c: c.analyses.representative and c.cluster.enabled),
    Step("rg", tuple(), lambda c: c.analyses.rg),
    Step("sasa", tuple(), lambda c: c.analyses.sasa and c.sasa.enabled),
    Step("ligand_site", tuple(), lambda c: c.analyses.ligand_site),
    Step("distance", tuple(), lambda c: c.analyses.distance and c.distance.enabled),
    Step("ramachandran", tuple(), lambda c: c.analyses.ramachandran and c.ramachandran.enabled),
]

STEP_INDEX = {step.name: step for step in STEPS}


def resolve_targets(config: AppConfig, only: set[str] | None, from_step: str | None, until_step: str | None) -> list[Step]:
    steps = [s for s in STEPS if s.enabled(config)]

    if only:
        selected = []
        for name in only:
            if name not in STEP_INDEX:
                raise ValueError(f"Unknown step in --only: {name}")
            selected.extend(expand_with_deps(STEP_INDEX[name], config))
        keep = {s.name for s in selected}
        steps = [s for s in steps if s.name in keep]

    if from_step or until_step:
        names = [s.name for s in steps]
        start_idx = names.index(from_step) if from_step else 0
        end_idx = names.index(until_step) if until_step else len(steps) - 1
        if start_idx > end_idx:
            raise ValueError("--from-step must appear before --until-step")
        steps = steps[start_idx : end_idx + 1]

    return steps


def expand_with_deps(step: Step, config: AppConfig) -> list[Step]:
    ordered: list[Step] = []
    for dep_name in step.deps:
        dep_step = STEP_INDEX[dep_name]
        if dep_step.enabled(config):
            ordered.extend(expand_with_deps(dep_step, config))
    ordered.append(step)

    dedup: list[Step] = []
    seen = set()
    for item in ordered:
        if item.name not in seen:
            seen.add(item.name)
            dedup.append(item)
    return dedup


def run_pipeline(
    config: AppConfig,
    config_text: str,
    resume: bool,
    force: bool,
    only: set[str] | None,
    force_steps: set[str],
    from_step: str | None,
    until_step: str | None,
) -> None:
    outdir = Path(config.output.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state = StateStore(outdir)
    cfg_hash = short_config_hash(config_text)
    ctx = RunContext(config=config, outdir=outdir, cache={})

    steps = resolve_targets(config, only=only, from_step=from_step, until_step=until_step)

    for step in steps:
        if not force and step.name in force_steps:
            state.clear_done(step.name)

        if resume and state.is_done(step.name) and step.name not in force_steps and not force:
            print(f"[skip] {step.name}: already done")
            continue

        unmet = [dep for dep in step.deps if dep in [s.name for s in steps] and not state.is_done(dep)]
        if unmet:
            raise RuntimeError(f"Cannot run {step.name}; unmet dependencies: {', '.join(unmet)}")

        try:
            execute_step(step.name, ctx)
            state.mark_done(step.name, {"config_hash": cfg_hash, "step": step.name})
            print(f"[done] {step.name}")
        except Exception as exc:  # pragma: no cover
            state.log_error(step.name, str(exc))
            print(f"[fail] {step.name}: {exc}")
            raise


def execute_step(step: str, ctx: RunContext) -> None:
    handler = STEP_HANDLERS.get(step)
    if handler is None:
        raise RuntimeError(f"No handler registered for step: {step}")
    handler(ctx)
