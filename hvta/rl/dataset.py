"""Task specs -> verl-style dataset rows.

One row per TaskSpec. A GRPO group is one row sampled ``n`` times: same game, seed and
layout. ``prompt`` holds the opening chat (system + first user turn) for length filtering
and inspection; the agent loop rebuilds it from ``extra_info.task``, which is
deterministic given the seed, and takes the system prompt from the row.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

from .episode import HVTAEpisode, WrapperPool
from .tasks import TaskSpec, write_tasks_jsonl

AGENT_NAME = "hvta"
DATA_SOURCE = "hvta"


def task_row(task: TaskSpec, pool: WrapperPool, system_prompt: Optional[str] = None) -> dict:
    ep = HVTAEpisode(task, system_prompt=system_prompt, pool=pool)
    messages = ep.reset()
    ep.force_end("dataset build")
    return {
        "data_source": DATA_SOURCE,
        "agent_name": AGENT_NAME,
        "prompt": messages,
        "ability": "game",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"task": json.dumps(task.to_dict()), "task_key": task.key(), "game": task.game},
    }


def to_verl_rows(tasks: Iterable[TaskSpec], system_prompt: Optional[str] = None) -> list[dict]:
    pool = WrapperPool()
    return [task_row(t, pool, system_prompt) for t in tasks]


def write_verl_parquet(tasks: Iterable[TaskSpec], out: str | Path, system_prompt: Optional[str] = None) -> Path:
    """Write the rows to ``out`` (.parquet) and the task specs beside it (.jsonl)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyarrow is needed for parquet output: uv sync --extra rl") from exc
    tasks = list(tasks)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tasks_jsonl(tasks, out.with_suffix(".jsonl"))
    pq.write_table(pa.Table.from_pylist(to_verl_rows(tasks, system_prompt)), out)
    return out
