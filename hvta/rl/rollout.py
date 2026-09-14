"""Async rollout driver against an OpenAI-compatible endpoint (vLLM), for calibration and evaluation.

Resumable: every finished episode is one JSONL line keyed by ``(task_key, sample)``, and
``run_many`` skips keys already on disk.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from loguru import logger

from .episode import HVTAEpisode, WrapperPool
from .metrics import EpisodeRecord
from .tasks import TaskSpec

Message = dict[str, str]
# An agent maps the chat so far to (reply text, usage dict or None).
AgentFn = Callable[[list[Message]], Awaitable[tuple[str, Optional[dict]]]]


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    enable_thinking: bool = False
    extra_body: dict = field(default_factory=dict)

    def request_kwargs(self) -> dict:
        extra = dict(self.extra_body)
        extra.setdefault("chat_template_kwargs", {})
        extra["chat_template_kwargs"].setdefault("enable_thinking", self.enable_thinking)
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "extra_body": extra,
        }


class ContextLengthExceeded(RuntimeError):
    """The server refused the request because the chat no longer fits its context window."""


def openai_agent(client: Any, model: str, sampling: SamplingConfig, retries: int = 3) -> AgentFn:
    """Wrap an ``openai.AsyncOpenAI`` client as an agent.

    A context-length rejection is raised as :class:`ContextLengthExceeded` at once (retrying
    cannot help); ``run_episode`` turns it into a budget truncation, the same end the
    training loop gives an episode that outgrows ``response_length``.
    """
    kwargs = sampling.request_kwargs()

    async def agent(messages: list[Message]) -> tuple[str, Optional[dict]]:
        delay = 1.0
        for attempt in range(retries):
            try:
                try:
                    resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
                except Exception as exc:  # noqa: BLE001 - classify before the generic retry below
                    if getattr(exc, "status_code", None) == 400 and "context length" in str(exc):
                        raise ContextLengthExceeded(str(exc)[:200]) from exc
                    raise
                msg = resp.choices[0].message
                text = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    text = f"<think>{reasoning}</think>{text}"
                usage = None
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                return text, usage
            except ContextLengthExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - retry any transport/server error
                if attempt == retries - 1:
                    raise
                logger.warning(f"chat completion failed ({exc!r}); retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    return agent


async def run_episode(
    agent: AgentFn,
    task: TaskSpec,
    system_prompt: Optional[str] = None,
    pool: Optional[WrapperPool] = None,
    max_episode_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """One episode. ``max_episode_tokens`` is the training loop's ``response_length``: every
    token after the opening prompt, the environment's included, counts against it (measured
    from the usage the server reports), and running out ends the episode with reward -1 as
    ``budget_truncated``. So does a context-length rejection by the server."""
    episode = HVTAEpisode(task, system_prompt=system_prompt, pool=pool)
    messages = episode.reset()
    completion_tokens = 0
    prompt_tokens_final: Optional[int] = None
    opening_tokens: Optional[int] = None
    budget_truncated = False
    try:
        while not episode.done:
            try:
                text, usage = await agent(messages)
            except ContextLengthExceeded:
                budget_truncated = True
                episode.force_end("rollout budget exhausted")
                break
            used = None
            if usage:
                completion_tokens += int(usage.get("completion_tokens", 0))
                prompt_tokens_final = usage.get("prompt_tokens")
                if opening_tokens is None and prompt_tokens_final is not None:
                    opening_tokens = int(prompt_tokens_final)
                if opening_tokens is not None and prompt_tokens_final is not None:
                    used = int(prompt_tokens_final) + int(usage.get("completion_tokens", 0)) - opening_tokens
            episode.step(text)
            messages = episode.messages
            if not episode.done and max_episode_tokens is not None and used is not None and used >= max_episode_tokens:
                budget_truncated = True
                episode.force_end("rollout budget exhausted")
    except Exception:
        if not episode.done:
            episode.force_end("agent error")
        raise
    record: EpisodeRecord = episode.record
    record.response_tokens = completion_tokens or None
    record.prompt_tokens_final = prompt_tokens_final
    record.budget_truncated = budget_truncated
    return {
        "task_key": task.key(),
        "task": task.to_dict(),
        "record": record.to_dict(),
        "messages": episode.messages,
        "actions": episode.actions,
    }


def _done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add((row["task_key"], int(row["sample"])))
    return done


def read_records(path: str | Path) -> list[EpisodeRecord]:
    """Records from a JSONL, gzipped or not (finished calibrations are stored as .jsonl.gz)."""
    import gzip

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [EpisodeRecord.from_dict(json.loads(line)["record"]) for line in f if line.strip()]


async def run_many(
    agent: AgentFn,
    tasks: Iterable[TaskSpec],
    n_samples: int,
    out_path: str | Path,
    concurrency: int = 32,
    system_prompt: Optional[str] = None,
    pool: Optional[WrapperPool] = None,
    progress: bool = True,
    max_episode_tokens: Optional[int] = None,
) -> list[EpisodeRecord]:
    """Run ``n_samples`` episodes per task, appending finished ones to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    jobs = [(t, i) for t in tasks for i in range(n_samples) if (t.key(), i) not in done]
    if not jobs:
        return read_records(out_path)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    pool = pool or WrapperPool()
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(jobs), desc=out_path.stem)
        except ImportError:
            bar = None

    async def one(task: TaskSpec, sample: int) -> None:
        async with sem:
            try:
                row = await run_episode(agent, task, system_prompt=system_prompt, pool=pool,
                                        max_episode_tokens=max_episode_tokens)
            except Exception as exc:  # noqa: BLE001 - one failed episode must not sink the batch
                logger.error(f"episode {task.key()} #{sample} failed: {exc!r}")
                return
            row["sample"] = sample
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            if bar is not None:
                bar.update(1)

    await asyncio.gather(*(one(t, i) for t, i in jobs))
    if bar is not None:
        bar.close()
    return read_records(out_path)


def sampling_from_dict(d: dict) -> SamplingConfig:
    return SamplingConfig(**d)


def sampling_to_dict(s: SamplingConfig) -> dict:
    return asdict(s)
