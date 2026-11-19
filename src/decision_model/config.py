"""Load the YAML configs and expand them into the run matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    path = config_dir / "models" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (config_dir / "models").glob("*.yaml"))
        raise FileNotFoundError(f"no model config {name!r}; available: {available}")
    return load_yaml(path)


@dataclass(frozen=True)
class Run:
    model: str
    group: str
    seed: int
    stage: int
    tasks: tuple[str, ...]
    levels: dict[str, str] = field(hash=False)
    train_langs: tuple[str, ...]
    eval_langs: tuple[str, ...]
    replay: int = 0
    train: bool = True

    @property
    def run_id(self) -> str:
        return f"{self.model}/{self.group}/s{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "group": self.group,
            "seed": self.seed,
            "stage": self.stage,
            "tasks": list(self.tasks),
            "levels": dict(self.levels),
            "train_langs": list(self.train_langs),
            "eval_langs": list(self.eval_langs),
            "replay": self.replay,
            "train": self.train,
        }


def expand_runs(
    model: str,
    groups_config: dict[str, Any],
    tasks_config: dict[str, Any],
    stages: set[int] | None = None,
    groups: set[str] | None = None,
    include_optional: bool = False,
) -> list[Run]:
    """All runs for one model, in config order."""
    default_seeds = list(groups_config.get("seeds", [0]))
    langs = tuple(groups_config.get("langs", ["ko", "ja", "en"]))
    selected = groups_config.get("selected_levels") or {}
    runs: list[Run] = []
    for name, group in groups_config["groups"].items():
        stage = int(group.get("stage", 1))
        if stages is not None and stage not in stages:
            continue
        if groups is not None and name not in groups:
            continue
        if group.get("optional") and not include_optional and (groups is None or name not in groups):
            continue
        trains = bool(group.get("train", True))
        seeds = list(group.get("seeds", default_seeds)) if trains else [0]
        train_langs = tuple(group.get("train_langs", langs))
        tasks = tuple(group["tasks"])

        variants: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
        sweep = group.get("sweep")
        if sweep == "all_levels":
            for task in tasks:
                task_config = tasks_config["tasks"][task]
                for level in task_config["levels"]:
                    if all(task_config["views"][v]["kind"] == "generated" for v in task_config["levels"][level]):
                        continue  # prompt baselines run in their own groups (P0-*)
                    variants.append((f"{name}-{task}-{level}", (task,), {task: level}))
        elif sweep is not None:
            if len(tasks) != 1:
                raise ValueError(f"{name}: a level sweep needs exactly one task")
            for level in sweep:
                variants.append((f"{name}-{level}", tasks, {tasks[0]: level}))
        else:
            levels_spec = group.get("levels")
            if levels_spec == "selected":
                missing = [task for task in tasks if not selected.get(task)]
                if missing:
                    raise ValueError(
                        f"{name}: selected_levels not set for {missing}; fill groups.yaml after stage 1"
                    )
                levels = {task: selected[task] for task in tasks}
            elif isinstance(levels_spec, dict):
                levels = dict(levels_spec)
            else:
                raise ValueError(f"{name}: needs `levels` or `sweep`")
            variants.append((name, tasks, levels))

        for group_name, group_tasks, levels in variants:
            for task, level in levels.items():
                if level not in tasks_config["tasks"][task]["levels"]:
                    raise ValueError(f"{group_name}: {task} has no level {level!r}")
            for seed in seeds:
                runs.append(
                    Run(
                        model=model,
                        group=group_name,
                        seed=int(seed),
                        stage=stage,
                        tasks=group_tasks,
                        levels=levels,
                        train_langs=train_langs,
                        eval_langs=langs,
                        replay=int(group.get("replay", 0)),
                        train=trains,
                    )
                )
    return runs
