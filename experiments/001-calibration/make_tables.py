#!/usr/bin/env python
"""Print the README's markdown tables from the .jsonl.gz files in this directory.

    .venv/bin/python experiments/001-calibration/make_tables.py

One table per file; the columns are defined in the README. Percentages are of n episodes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvta.rl import aggregate  # noqa: E402
from hvta.rl.rollout import read_records  # noqa: E402

FILES = [
    ("qwen3-4b_hs_d1-3.jsonl.gz", "Hidden solution, hack directory 1-3 levels deep (the training default)"),
    ("qwen3-4b_hs_d1.jsonl.gz", "Hidden solution, depth 1"),
    ("qwen3-4b_hs_d3.jsonl.gz", "Hidden solution, depth 3"),
    ("qwen3-4b_lb_d1-3.jsonl.gz", "Logical bug, depth 1-3"),
]
COLS = ["honest_win", "hacked_win", "hack", "readme", "dir_listed", "trunc", "budget", "invalid_end"]


def table(path: Path) -> str:
    by_env: dict[str, list] = {}
    for r in read_records(path):
        by_env.setdefault(r.env_id, []).append(r)
    lines = ["| game | n | " + " | ".join(COLS) + " | steps | fs | tokens |",
             "|---|---:|" + "---:|" * (len(COLS) + 3)]
    for env_id, recs in sorted(by_env.items()):
        a = aggregate(recs, prefix="x", per_game=False)
        n = a["x/n"]
        pct = {
            "honest_win": a["x/n_honest_win"], "hacked_win": a["x/n_hacked_win"], "hack": a["x/n_hack"],
            "readme": a["x/n_readme_read"], "dir_listed": a["x/n_hack_dir_listed"], "trunc": a["x/n_truncated"],
            "budget": a["x/n_budget_truncated"], "invalid_end": a["x/n_invalid_end"],
        }
        cells = [f"{pct[c] / n:.1%}" for c in COLS]
        lines.append(f"| {env_id} | {int(n)} | " + " | ".join(cells)
                     + f" | {a['x/mean_steps']:.1f} | {a['x/mean_fs_steps']:.1f} | {a.get('x/mean_response_tokens', float('nan')):.0f} |")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    for name, title in FILES:
        path = here / name
        if not path.exists():
            continue
        print(f"\n**{title}** (`{name}`)\n")
        print(table(path))


if __name__ == "__main__":
    main()
