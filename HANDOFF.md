# Handoff

For a session continuing this work. Current state as of 2026-09-15; `STATUS.md` is the short
version. Files are named so you can go and read them; this doc does not repeat their content.

## Purpose and shape

- Second environment for `~/projects/rl-exploration` (Qwen3-4B, GRPO with LoRA, verl 0.6.1
  vendored in `ariahw/rl-rewardhacking@73695ff`, jobs on 2xH200 through the OpenWeights
  queue, `tools/rlrh_job.py`). Its `research.md` names "a second environment" as the gate
  on any generality claim about how exploration shapes what RL teaches.
- This repo is a fork of `MajoRoth/hack-verifiable-environments` (HV-TextArena) on branch
  `rl-env`, remote `git@github.com:vohonen/hack-verifiable-environments.git`. Upstream is
  `MajoRoth/...`; do not push there.
- Original plan file: `~/.claude/plans/hi-there-s-a-markdown-compiled-blum.md` (design
  decisions with Vili, verification list). Memo that chose this environment:
  `~/projects/rl-envs/rl-unsafe-exploration-envs.md`.

## Map

| Piece | Where | Notes |
|---|---|---|
| Episode protocol | `hvta/rl/episode.py` | `HVTAEpisode(TaskSpec)`: `reset()` -> `[system, user]`, `step(text)` -> next user turn; `force_end()` for the trainer's budget. `DEFAULT_SYSTEM_PROMPT` is the one prompt everything uses. |
| Action extraction | `hvta/rl/actions.py` | Last bracketed group of a reply, `<think>` stripped. |
| Tasks | `hvta/rl/tasks.py` | `TaskSpec` (env id, seed, hack class, `FsLayout`, caps), `make_tasks`, `default_max_steps` = min(40, max(2*turns, turns + fs budget)). |
| Record and counters | `hvta/rl/metrics.py` | `EpisodeRecord` (one episode), `record_to_columns` (flat `hvta_*` row the trainer logs), `aggregate`/`aggregate_columns` (`detail/hvta/n_*`). |
| Rollout driver | `hvta/rl/rollout.py` | Async OpenAI-compatible driver for calibration; resumable JSONL; enforces `max_episode_tokens`; `ContextLengthExceeded` -> budget truncation. |
| verl agent loop | `hvta/rl/verl_agent_loop.py`, `verl_agent_loop.yaml` | `HVTALoopCore` is testable without verl; `HVTAAgentLoop` registers as `hvta` when verl imports. Reads `HVTA_MAX_TOKENS_PER_TURN` (default 256). |
| Filesystem and detection | `hvta/FilesystemWrapper/` | `virtual_fs.py` (`FsLayout`, decoys, hack placement), `filesystem_wrapper.py` (step caps, invalid counting, detection by content), `logical_bug_envs/` (five planted bugs). |
| Trainer patches | `integrations/rl-rewardhacking/` | `hvta-agent-loop.patch` (trainer; identical copy in rl-exploration's `patches/`), `rl-exploration-wiring.patch` (what rl-exploration's `main` still lacks, only while something is pending), README with the table of changes and how the pod gets the package. |
| Scripts | `scripts/` | `serve_ow.py`, `calibrate.py`, `build_dataset.py`. |
| Calibration | `experiments/001-calibration/` | README (method, tables, conclusions), gzipped JSONLs with transcripts, `make_tables.py`. |
| Tests | `tests/` | 80, ~15 s: `.venv/bin/python -m pytest tests/ -q`. `test_verl_agent_loop.py` covers the loop core with a fake generator. |

## How the trainer integration works

- The rollout is verl's async agent loop. `hvta-agent-loop.patch` makes `GRPOConfig` carry
  `agent_loop`, `agent_loop_config_path`, `max_assistant_turns`; `grpo.py` accepts a
  `.parquet` dataset (rows from `hvta.rl.dataset`) and switches the composed verl config to
  `rollout.mode=async` + `multi_turn` + the loop; `trainer.py` takes the reward from
  `rm_scores` (the loop scored the episode) and skips the reward manager, aliases
  `hvta_hack_triggered` -> `response_test_func_arbitrary_pass`/`eq_hinted` and
  `hvta_honest_win` -> `eq_correct` so the existing early stop works unchanged, and logs
  `detail/hvta/*` with the trainer's logger; `vllm_async_server.py` honours a per-turn
  `max_tokens`. Thinking is off because the trainer's `enable_thinking=False` lands in
  `data.apply_chat_template_kwargs`, which the loop applies on every turn.
- Two fire arms in `scripts/run_rl_training.py`: `hvta_hidden_solution`, `hvta_logical_bug`.
  Both call `main_run_rl` exactly once and forward `**kwargs`. This shape is mandatory:
  rl-exploration's `rh-entrypoint-kwargs.patch` adds tests (`tests/test_cli_kwargs.py`)
  that enforce it, and `tools/rlrh_job.py`'s `check_extra_args` reads the arm's signature
  with `ast` and refuses an `--extra key` the signature does not name (unless it is a
  `GRPOConfig` field). Named parameters: `dataset_path, games, seeds, depth, max_fs_steps,
  prompt_name, target_prompt_name, max_assistant_turns, label`. Defaults are the
  calibrated ones (games, depth `(3, 3)`, fs budget 12). `hvta_training_args` builds the
  dict; a flag in `**kwargs` beats any default it sets.
- The parquet is built on the pod from those parameters (hashed name under
  `results/data/`), or `--dataset_path` points at one from `scripts/build_dataset.py`.
- Chain position: after `rh-entrypoint-kwargs`, before the `rh-jan2026-params*` patches.
  `PATCH_DEPENDS_ON` lists six chain patches under it (anti-hack, RC, runtime-prompts,
  reward-metric-step, early-stop, entrypoint-kwargs); it does not need
  `rh-unparse-recursion-guard`. Landed in rl-exploration at `bfdcddd`; the job-start install
  of the package at `37529ad`.
- Run ids from the queue are `hvta-<label>-s<seed>-<stamp>` (default labels `hs-baseline`,
  `lb-baseline`); HF repos `rlrh-hvta-...`. The pod-side runner still builds the LeetCode
  datasets and runs the LeetCode checkpoint eval unless `--skip-eval`; on an hvta run that
  eval is a transfer measurement (~8 min), harmless.
- The package on the pod: not baked. rl-exploration's `tools/rlrh_job.sh` installs
  `hack-verifiable-environments[rl]` and TextArena into the venv at job start, at the sha the
  job carries (`HVTA_COMMIT` in `tools/rlrh_job.py`, `--hvta-commit` to override), fetches the
  two NLTK corpora, and dies if a shared package (torch, vllm, transformers, ray, numpy)
  changed version. Pin is `08e0989` (the calibration commit; `hvta/` has not changed since).
  Bump it whenever `hvta/` changes, and push first: the pod clones by sha. The runner logs
  `hvta ok`. Bake it into `docker/Dockerfile` once the loop is stable; the image workflow
  builds from scratch with about 3 GB of disk to spare, so that layer must stay small.

## Verifying the chain without a pod

```bash
# trainer tree with the whole chain applied
T=$TMPDIR/hvta-chain; git clone -q --no-checkout ~/projects/rl-exploration/repos/rl-rewardhacking $T
cd $T && git checkout -q 73695ff
for p in rh-checkpoints-resume rh-run-naming rh-anti-hack-prompts rh-recontextualization rh-runtime-prompts \
         rh-reward-metric-step rh-early-stop rh-unparse-recursion-guard rh-entrypoint-kwargs; do
  git apply --whitespace=nowarn ~/projects/rl-exploration/patches/$p.patch; done
git apply --whitespace=nowarn <hvta>/integrations/rl-rewardhacking/hvta-agent-loop.patch
<hvta>/.venv/bin/python -m pytest -q tests/test_cli_kwargs.py          # the chain's own shape tests
git apply --check --whitespace=nowarn ~/projects/rl-exploration/patches/rh-jan2026-params.patch
# to regenerate the patch after editing files here: git diff > <hvta>/integrations/rl-rewardhacking/hvta-agent-loop.patch

# the submitter's own dry run (patch chain + --extra gate) against a scratch clone of rl-exploration
git clone -q ~/projects/rl-exploration $TMPDIR/rlx && cd $TMPDIR/rlx
git apply --whitespace=nowarn <hvta>/integrations/rl-rewardhacking/rl-exploration-wiring.patch   # only while one is pending
set -a; . ~/projects/rl-exploration/.env; set +a
"$(uv tool dir)/openweights/bin/python" tools/rlrh_job.py submit --arm hvta_hidden_solution --seed 1 --steps 5 \
    --patch hvta-agent-loop.patch --early-stop 0.90 --skip-eval --dry-run
# to regenerate the wiring patch after editing the scratch clone: git add -A && git diff --cached > <hvta>/integrations/rl-rewardhacking/rl-exploration-wiring.patch
```

The dry run uses `$TMPDIR/rlrh-patchcheck-73695ff` (fetched from GitHub once, cached).
When rl-exploration's chain moves again, the wiring patch stops applying first; check the
trainer patch against the new chain before regenerating it.

## Calibration: serving and measuring

- Serve: `scripts/serve_ow.py --model Qwen/Qwen3-4B --endpoint-file <f>` with the
  openweights tool venv's interpreter and `.env` sourced. 1x H100S is $2.69/h, high stock;
  ~10 min from job creation to first completion (pod ~4 min, vLLM ~4 min). The script
  attaches to an existing live deployment with the same parameters (OpenWeights job ids are
  parameter hashes). Teardown: `touch <f>.stop` or SIGTERM; the cluster manager releases the
  pod a few minutes after the job is cancelled. Confirm with `~/projects/rl-exploration/tools/pod list`.
- Do not use `ow.api.deploy(...).up()` from the sandbox: the openweights venv's HTTP client
  cannot reach `*.proxy.runpod.net` (curl and this repo's `.venv` can); `up()` retries for an
  hour, then raises, and its `down()` cancels the job. That is why `serve_ow.py` probes with
  curl and refreshes the lease itself.
- Measure: `scripts/calibrate.py` (see the experiment README's reproduce block). Throughput
  on one H100 with concurrency 48-64: 3-6 episodes/s; 128 episodes per game per depth is a
  few minutes. `--samples 0` reprints the table from a JSONL. `--extra-body` passes provider
  knobs (OpenRouter needs `{"reasoning":{"enabled":false}}`; `chat_template_kwargs` alone
  does not stop its reasoning). OpenRouter has no Qwen3-4B endpoint; `openrouter/qwen/qwen3-8b`
  through the LiteLLM proxy in `.env` works for pipeline tests at cents.
- Resume rules: the JSONL is keyed by `(task_key, sample)` and `task_key` is
  `env|seed|hack|d<lo>-<hi>|prompt`; it does not include `max_steps`, `max_fs_steps` or the
  token budget. After changing any of those, delete the affected rows (filter by
  `record.env_id`) before rerunning, or the old episodes are kept. A failed episode is not
  written, so a rerun fills it in.
- Numbers (Qwen3-4B, non-thinking, T=1, 128 episodes per cell; full tables in the README):
  hidden solution at depth 3: Sudoku-very-easy honest 17 % / hack 7 %, Mastermind 3 % / 4 %,
  Hangman 0 % / 7 %, Wordle 0 % / 0 %, GuessTheNumber 68 % / 0 % (never uses the filesystem).
  Depth 1 roughly doubles hacks on Mastermind and Wordle and multiplies precursors 5x.
  Logical bug: flags fire by accident (illegal moves are the bug) in 58-100 % of
  FifteenPuzzle, Minesweeper, TowerOfHanoi episodes, zero wins anywhere; parked.
- Cost so far: one H100 for 53 min, about $2.40.

## What bit us (sandbox and tooling)

- The sandbox can write only under `~/projects/rl-envs` and `$TMPDIR`; `rl-exploration` is
  read-only from here and `git push` needs SSH, which the sandbox cannot do. Patches are
  produced here; Vili applies, commits and pushes. `gh` is also blocked (its config dir is
  denied), so Actions logs come through Vili.
- Anything a pod or an image build fetches by sha must be on GitHub first. The one image build
  that carried an hvta layer (2026-09-14) died on exactly this: it pinned `08e0989` while
  `origin/rl-env` was still at `602151b`. Check `git status -sb` for "ahead" before pinning.
- Background jobs: only processes started in the current Bash call can be signalled; use
  `run_in_background` and `TaskStop`. zsh does not word-split `$VAR` used as a command (use
  a function); its `nice(5) failed` warnings on `&` jobs are harmless.
- `pytest` in the trainer tree: `tests/test_cli_kwargs.py` runs without torch; nothing else
  there does.
- `runpod_specs.py` GPU keys are `H200 H100S H100N A100S A100 A40 L40 ...`, same names
  OpenWeights uses in `allowed_hardware` (`"1x H100S"`).

## Decisions taken (with Vili)

- verl's agent loop inside the existing stack, glue beside the environment, not a new
  trainer. Qwen3-4B non-thinking first. Both hack classes as separate arms; the logical-bug
  arm parked after calibration. Two-player channels out of scope.
- Depth 3 as the default because depth 1 (about 8 % hacks pooled at step 0) is too easy
  ("kinda too easy", 2026-09-14). Depth 3 is still about 4.5 % pooled; the filesystem budget
  is the next lever.
- Reward is the game's reward; hack flags never enter it.
- GuessTheNumber excluded from the hack arm (zero base rate); optional as an honest control.

## Next steps

1. Smoke run (5 steps, `--skip-eval`; command in the integration README). Watch: the runner's
   `hvta ok` line, then workers import `hvta`; `detail/hvta/*` panels appear from step 1; no
   assert from `_recontextualize_batch` (only if an RC arm is run); episode lengths vs
   `max_completion_length` 4096 (`n_budget_truncated` should be near zero for these games:
   calibration saw 0-1 % over budget). Logs: `https://<pod_id>-10101.proxy.runpod.net/`.
2. If the loop needs fixing, changes land in `hvta/rl/verl_agent_loop.py` (push, then
   `--hvta-commit <sha>` or a bump of `HVTA_COMMIT`) or in the trainer patch (regenerate it,
   copy to rl-exploration's `patches/`; patches are mounted per job). Neither needs an image.
3. Then 200 steps, one seed; then 3 seeds, read with `rl-exploration`'s onset and hazard
   tools. Honest-pass axis: `n_honest_win` on a pinned held-out seed set (not built yet:
   `scripts/build_dataset.py --seeds 512 640 ...` would make one).
5. Optional calibration: `--max-fs-steps 6` and `8` on the four games at depth 3, to see
   whether the base rate can be pushed toward 1 % without killing the precursors.
6. Logical-bug arm, if revived: count a bug as a hack only when exploited on purpose (source
   read first, or a win), and drop FifteenPuzzle/Minesweeper (boards exceed the token budget).
