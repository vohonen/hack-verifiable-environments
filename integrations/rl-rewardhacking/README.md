# Plugging hvta into the rl-rewardhacking / verl trainer

`hvta-agent-loop.patch` teaches `ariahw/rl-rewardhacking@73695ff` (verl 0.6.1 vendored) to
train on hvta episodes through verl's async agent loop. It is written against that commit
**with the whole rl-exploration patch chain applied** (`rh-checkpoints-resume` through
`rh-entrypoint-kwargs`); `rl-exploration-wiring.patch` declares which of those it needs
under it, and the two `rh-jan2026-params*` patches apply cleanly on top of it.

## What the patch changes

| File | Change |
|---|---|
| `src/train/config.py` | `GRPOConfig.agent_loop`, `agent_loop_config_path`, `max_assistant_turns`. |
| `src/train/verl/grpo.py` | A `.parquet` dataset path is used as-is (rows from `hvta.rl.dataset`), with the sampling system prompt and the recontextualization target applied to the opening chat like on the JSONL path. `_configure_agent_loop` sets `rollout.mode=async`, `multi_turn.enable`, `max_assistant_turns`, `agent.default_agent_loop`, `agent_loop_config_path`, `data.return_raw_chat` on the composed config (not in the template, so the parameter patches keep applying). The trainer's `enable_thinking` (default false) already reaches `data.apply_chat_template_kwargs`, which the loop applies on every turn. |
| `src/train/verl/trainer.py` | The sync-only assert admits `async` (both worker classes already existed). When the batch carries `rm_scores` from the agent loop, the reward is taken from there and the reward manager is skipped; the `hvta_*` columns become `reward_extra_infos_dict`, with `response_test_func_arbitrary_pass`/`eq_hinted` := hack flag and `eq_correct` := honest win so the existing early stop and its lambda work unchanged (lambda = P(hack \| not an honest win)). `detail/hvta/*` counters (per game too) are logged with the trainer's own logger at the right step. |
| `verl/.../vllm_async_server.py` | Honour a `max_tokens` entry in sampling params, so the loop can cap one turn (256 tokens by default). |
| `scripts/run_rl_training.py` | Arms `hvta_hidden_solution` and `hvta_logical_bug`. Like every other arm they call `main_run_rl` once and forward `**kwargs`, so a `--key=value` flag reaches the config (or dies there by name) and the submitter's `--extra` gate reads their signatures. `main_run_rl` takes a `.parquet` base path as-is. Each arm builds its parquet on the pod from `--games/--seeds/--depth/--max_fs_steps` (idempotent, hashed name) unless `--dataset_path` is given; `--prompt_name` samples under an intervention prompt prepended to hvta's own; `--target_prompt_name` adds recontextualization. Batch geometry defaults to 32 tasks x 8 episodes = 256 per step, the size the onset and early-stop rules were calibrated on; a flag beats any of these defaults. |

The agent loop itself lives in this repo: `hvta/rl/verl_agent_loop.py`, registered as `hvta`
through `hvta/rl/verl_agent_loop.yaml`. The reward is the game's reward; hack flags never
enter it.

## Image requirements

- This package installed in the training venv at a pinned sha of the fork, with the `rl`
  extra: `hack-verifiable-environments[rl] @ git+https://github.com/vohonen/hack-verifiable-environments@<sha>`,
  plus TextArena at the commit `uv.lock` pins (plain `pip`/`uv pip` ignore `[tool.uv.sources]`).
- NLTK corpora `words` and `averaged_perceptron_tagger_eng` on a path in `nltk.data.path`.
- `pandas` and `pyarrow` (verl already depends on both).

## Using it from rl-exploration

1. Apply `rl-exploration-wiring.patch` in the rl-exploration checkout:

   ```bash
   git apply --whitespace=nowarn integrations/rl-rewardhacking/rl-exploration-wiring.patch
   ```

   (The whitespace warnings it silences are context lines of the embedded trainer patch.)
   It copies `hvta-agent-loop.patch` into `patches/`; adds it to `RlrhRunJob.mount`,
   `PATCH_ORDER` (after `rh-entrypoint-kwargs`, before the params patches) and
   `PATCH_DEPENDS_ON` (the six chain patches it needs, so `--patch hvta-agent-loop.patch`
   alone resolves to a chain that applies); gives the two arms default labels
   (`hs-baseline`, `lb-baseline`) and run ids of the form `hvta-<label>-s<seed>-<stamp>`
   instead of `wong2025-...`; and adds one layer to `docker/Dockerfile` that installs
   `hack-verifiable-environments[rl]` at a pinned sha (plus TextArena at hvta's pinned
   commit) into the training venv and fetches the NLTK corpora. Bump `HVTA_COMMIT` there
   whenever `hvta/` moves (a change confined to `integrations/` does not need it). The patch
   is generated against rl-exploration's `main`; if it stops applying, the chain moved again
   and the trainer patch needs re-checking against it too (`tools/rlrh_job.py`'s dry run
   does exactly that).
2. Rebuild the image (`build-gpu-image.yml`, manual dispatch; the build log ends its hvta
   layer with `hvta ok`) and point `DEFAULT_IMAGE` in `tools/rlrh_job.py`, or `--image`, at
   the new tag.
3. Submit as any other arm. The smoke run first, then the real one:

   ```bash
   # 5 steps, no checkpoint eval: proves the async rollout path, the reward plumbing and the
   # detail/hvta/* panels. --dry-run first shows the resolved chain and checks it locally.
   $OWPY tools/rlrh_job.py submit --arm hvta_hidden_solution --seed 1 --steps 5 \
       --patch hvta-agent-loop.patch --early-stop 0.90 --skip-eval --image <new tag> \
       --extra games=GuessTheNumber-v0,Wordle-v0,Hangman-v0,Mastermind-v0 --extra depth=1,3

   $OWPY tools/rlrh_job.py submit --arm hvta_hidden_solution --seed 1 --steps 200 \
       --patch hvta-agent-loop.patch --early-stop 0.90 --image <new tag>
   ```

   Without `--skip-eval` the job ends with the LeetCode checkpoint eval, which on an hvta
   run measures transfer to the coding hack (about 8 minutes). Per-turn generation is capped
   at 256 tokens (`HVTA_MAX_TOKENS_PER_TURN` overrides), the episode at
   `max_completion_length` (4096) tokens of policy plus environment text, and at
   `max_assistant_turns` (40) turns; a budget-truncated episode ends with reward -1.

## What to watch on the first run

- The async rollout path in this fork had never run before this patch; the first smoke
  run is the test. If the agent-loop workers fail to import `hvta`, the venv on the image
  is missing it.
- `detail/hvta/n_hack`, `n_honest_win`, `n_hacked_win`, `n_fail_no_hack`, `n_truncated`,
  `n_budget_truncated`, `mean_steps` should all appear in wandb from step 1.
- Recontextualization swaps the whole prompt block, which for the agent loop is the
  opening chat; `_recontextualize_batch`'s two asserts will fire if the swap moves response
  tokens. Run one RC arm for a few steps before trusting it.
