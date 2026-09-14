#!/usr/bin/env python
"""Write a verl-style parquet (and a tasks JSONL) from a list of task specs.

    python scripts/build_dataset.py --hidden-solution --games Wordle-v0 Hangman-v0 \\
        --seeds 0 512 --depth 1 3 --out data/hvta_hs_train.parquet

See ``hvta.rl.dataset`` for the row layout.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvta.FilesystemWrapper.virtual_fs import FsLayout  # noqa: E402
from hvta.rl import make_tasks  # noqa: E402
from hvta.rl.dataset import write_verl_parquet  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--seeds", nargs=2, type=int, default=[0, 256], metavar=("START", "STOP"))
    p.add_argument("--hidden-solution", action="store_true")
    p.add_argument("--logical-bug", action="store_true")
    p.add_argument("--depth", nargs=2, type=int, default=[1, 3], metavar=("LO", "HI"))
    p.add_argument("--max-fs-steps", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--prompt-variant", default="neutral")
    p.add_argument("--out", required=True, help=".parquet path; a .jsonl of the task specs is written beside it")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.hidden_solution or args.logical_bug):
        sys.exit("pick --hidden-solution and/or --logical-bug")
    tasks = make_tasks(
        args.games,
        range(args.seeds[0], args.seeds[1]),
        hidden_solution=args.hidden_solution,
        logical_bug=args.logical_bug,
        fs_layout=FsLayout.randomised(depth=(args.depth[0], args.depth[1])),
        max_steps=args.max_steps,
        max_fs_steps=args.max_fs_steps,
        prompt_variant=args.prompt_variant,
    )
    out = write_verl_parquet(tasks, args.out)
    print(f"wrote {len(tasks)} rows to {out} and task specs to {out.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
