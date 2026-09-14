#!/usr/bin/env python
"""Sample a base model on candidate tasks and tabulate honest pass, hack and precursor rates.

Example (vLLM serving Qwen3-4B on localhost:8000):

    python scripts/calibrate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B \\
        --hidden-solution --games Wordle-v0 Hangman-v0 GuessTheNumber-v0 Mastermind-v0 \\
        --depth 1 3 --seeds 16 --samples 8 --out results/calibration/hs_d1-3.jsonl

Resumable: rerunning with the same --out skips finished (task, sample) pairs. The table is
recomputed from the JSONL on every run, so `--samples 0` just prints it.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvta.FilesystemWrapper.virtual_fs import FsLayout  # noqa: E402
from hvta.rl import aggregate, make_tasks  # noqa: E402
from hvta.rl.rollout import SamplingConfig, openai_agent, read_records, run_many  # noqa: E402

HIDDEN_SOLUTION_CANDIDATES = [
    "GuessTheNumber-v0",
    "GuessTheNumber-v0-hardcore",
    "Wordle-v0",
    "Hangman-v0",
    "Mastermind-v0",
    "Sudoku-v0-very-easy",
    "Sudoku-v0-easy",
    "Crosswords-v0",
]
LOGICAL_BUG_CANDIDATES = [
    "Wordle-v0",
    "Sudoku-v0-very-easy",
    "Minesweeper-v0-small",
    "TowerOfHanoi-v0",
    "FifteenPuzzle-v0",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("HVTA_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("HVTA_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--games", nargs="*", default=None, help="env ids; default depends on the hack class")
    p.add_argument("--hidden-solution", action="store_true")
    p.add_argument("--logical-bug", action="store_true")
    p.add_argument("--depth", nargs=2, type=int, default=[1, 1], metavar=("LO", "HI"), help="hack dir depth range")
    p.add_argument("--seeds", type=int, default=16, help="number of game seeds per env id")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--samples", type=int, default=8, help="episodes per (env id, seed)")
    p.add_argument("--max-fs-steps", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256, help="per-turn generation cap")
    p.add_argument("--max-episode-tokens", type=int, default=4096,
                   help="the trainer's response_length: policy plus environment tokens after the opening prompt; 0 = none")
    p.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    p.add_argument("--extra-body", default=None, metavar="JSON",
                   help="merged into every request body, for provider knobs vLLM does not need (e.g. OpenRouter's reasoning switch)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out", required=True, help="JSONL of finished episodes")
    return p.parse_args()


def print_table(records, title: str) -> None:
    by_env: dict[str, list] = {}
    for r in records:
        by_env.setdefault(r.env_id, []).append(r)
    cols = ["n", "honest_win", "hacked_win", "hack", "readme", "dir_listed", "trunc", "budget", "invalid_end", "steps", "fs", "tokens"]
    print(f"\n{title}")
    print(f"{'env_id':28s} " + " ".join(f"{c:>11s}" for c in cols))
    for env_id, recs in sorted(by_env.items()):
        a = aggregate(recs, prefix="x", per_game=False)
        n = a["x/n"]
        row = [
            f"{int(n):d}",
            f"{a['x/n_honest_win'] / n:.1%}",
            f"{a['x/n_hacked_win'] / n:.1%}",
            f"{a['x/n_hack'] / n:.1%}",
            f"{a['x/n_readme_read'] / n:.1%}",
            f"{a['x/n_hack_dir_listed'] / n:.1%}",
            f"{a['x/n_truncated'] / n:.1%}",
            f"{a['x/n_budget_truncated'] / n:.1%}",
            f"{a['x/n_invalid_end'] / n:.1%}",
            f"{a['x/mean_steps']:.1f}",
            f"{a['x/mean_fs_steps']:.1f}",
            f"{a.get('x/mean_response_tokens', float('nan')):.0f}",
        ]
        print(f"{env_id:28s} " + " ".join(f"{v:>11s}" for v in row))


async def main() -> None:
    args = parse_args()
    if not (args.hidden_solution or args.logical_bug):
        sys.exit("pick --hidden-solution and/or --logical-bug")
    games = args.games or (HIDDEN_SOLUTION_CANDIDATES if args.hidden_solution else LOGICAL_BUG_CANDIDATES)
    layout = FsLayout.randomised(depth=(args.depth[0], args.depth[1]))
    tasks = make_tasks(
        games,
        range(args.seed_offset, args.seed_offset + args.seeds),
        hidden_solution=args.hidden_solution,
        logical_bug=args.logical_bug,
        fs_layout=layout,
        max_steps=args.max_steps,
        max_fs_steps=args.max_fs_steps,
    )
    if args.samples > 0:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        sampling = SamplingConfig(temperature=args.temperature, max_tokens=args.max_tokens, enable_thinking=args.thinking,
                                  extra_body=json.loads(args.extra_body) if args.extra_body else {})
        agent = openai_agent(client, args.model, sampling)
        records = await run_many(agent, tasks, args.samples, args.out, concurrency=args.concurrency,
                                 max_episode_tokens=args.max_episode_tokens or None)
    else:
        records = read_records(args.out)
    cls = "+".join(c for c, on in (("hidden_solution", args.hidden_solution), ("logical_bug", args.logical_bug)) if on)
    print_table(records, f"{args.model}  {cls}  depth {args.depth[0]}-{args.depth[1]}  ({len(records)} episodes)")


if __name__ == "__main__":
    asyncio.run(main())
