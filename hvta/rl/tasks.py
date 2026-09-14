"""A task is one game instance with its hack configuration: the unit a GRPO group shares."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

from textarena.envs.registration import ENV_REGISTRY

from hvta.FilesystemWrapper.solution_extractors import has_secret
from hvta.FilesystemWrapper.virtual_fs import FsLayout

MAX_STEPS_CAP = 40
DEFAULT_MAX_FS_STEPS = 12


def game_name_of_env_id(env_id: str) -> str:
    """``Wordle-v0-hardcore`` -> ``Wordle``, via the registry's entry point."""
    spec = ENV_REGISTRY.get(env_id)
    if spec is None:
        raise ValueError(f"{env_id!r} is not a registered TextArena env id")
    entry = spec.entry_point
    if isinstance(entry, str):
        # "textarena.envs.Wordle.env:WordleEnv"
        cls_name = entry.split(":")[-1]
    else:
        cls_name = entry.__name__
    for suffix in ("BugEnv", "Env"):
        if cls_name.endswith(suffix):
            return cls_name[: -len(suffix)]
    return cls_name


def game_turns_of_env_id(env_id: str) -> Optional[int]:
    """The game's own turn budget from its registration kwargs, if it has one."""
    kwargs = ENV_REGISTRY[env_id].kwargs
    for key in ("max_turns", "num_guesses", "num_turns"):
        if key in kwargs and isinstance(kwargs[key], int):
            return kwargs[key]
    return None


def default_max_steps(env_id: str, max_fs_steps: Optional[int] = DEFAULT_MAX_FS_STEPS) -> int:
    """Action cap: the game's turns plus the filesystem budget, at least twice the turns, at most 40.

    Filesystem commands, refused ones included, are actions too, so the cap has to leave
    room for every game turn after the whole filesystem budget is spent; otherwise a policy
    that explores the filesystem cannot finish a short game at all (Wordle has six turns).
    """
    turns = game_turns_of_env_id(env_id)
    if turns is None:
        return MAX_STEPS_CAP
    budget = max_fs_steps if max_fs_steps is not None else DEFAULT_MAX_FS_STEPS
    return min(MAX_STEPS_CAP, max(2 * turns, turns + budget))


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    seed: int
    hidden_solution: bool = False
    logical_bug: bool = False
    fs_layout: FsLayout = field(default_factory=FsLayout)
    max_steps: Optional[int] = None
    max_fs_steps: Optional[int] = DEFAULT_MAX_FS_STEPS
    prompt_variant: str = "neutral"

    def __post_init__(self) -> None:
        if self.hidden_solution and not has_secret(game_name_of_env_id(self.env_id)):
            raise ValueError(
                f"{self.env_id} is fully observable; hidden_solution leaks nothing there. Use logical_bug."
            )

    @property
    def game(self) -> str:
        return game_name_of_env_id(self.env_id)

    @property
    def resolved_max_steps(self) -> int:
        return self.max_steps if self.max_steps is not None else default_max_steps(self.env_id, self.max_fs_steps)

    @property
    def hack_config(self) -> str:
        parts = []
        if self.hidden_solution:
            parts.append("hs")
        if self.logical_bug:
            parts.append("lb")
        return "+".join(parts) or "none"

    def key(self) -> str:
        lo, hi = self.fs_layout.depth
        return f"{self.env_id}|s{self.seed}|{self.hack_config}|d{lo}-{hi}|{self.prompt_variant}"

    def with_seed(self, seed: int) -> "TaskSpec":
        return replace(self, seed=seed)

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "hidden_solution": self.hidden_solution,
            "logical_bug": self.logical_bug,
            "fs_layout": self.fs_layout.to_dict(),
            "max_steps": self.max_steps,
            "max_fs_steps": self.max_fs_steps,
            "prompt_variant": self.prompt_variant,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(
            env_id=d["env_id"],
            seed=int(d["seed"]),
            hidden_solution=bool(d.get("hidden_solution", False)),
            logical_bug=bool(d.get("logical_bug", False)),
            fs_layout=FsLayout.from_dict(d.get("fs_layout", {})),
            max_steps=d.get("max_steps"),
            max_fs_steps=d.get("max_fs_steps", DEFAULT_MAX_FS_STEPS),
            prompt_variant=d.get("prompt_variant", "neutral"),
        )


def make_tasks(
    env_ids: Iterable[str],
    seeds: Iterable[int],
    hidden_solution: bool = False,
    logical_bug: bool = False,
    fs_layout: Optional[FsLayout] = None,
    max_steps: Optional[int] = None,
    max_fs_steps: Optional[int] = DEFAULT_MAX_FS_STEPS,
    prompt_variant: str = "neutral",
) -> list[TaskSpec]:
    layout = fs_layout or FsLayout()
    seeds = list(seeds)
    return [
        TaskSpec(
            env_id=env_id,
            seed=seed,
            hidden_solution=hidden_solution,
            logical_bug=logical_bug,
            fs_layout=layout,
            max_steps=max_steps,
            max_fs_steps=max_fs_steps,
            prompt_variant=prompt_variant,
        )
        for env_id in env_ids
        for seed in seeds
    ]


def write_tasks_jsonl(tasks: Iterable[TaskSpec], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def read_tasks_jsonl(path: str | Path) -> list[TaskSpec]:
    with Path(path).open(encoding="utf-8") as f:
        return [TaskSpec.from_dict(json.loads(line)) for line in f if line.strip()]
