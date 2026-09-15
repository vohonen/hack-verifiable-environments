# hack-verifiable-environments (fork with an RL layer)

Fork of `MajoRoth/hack-verifiable-environments` (HV-TextArena, arXiv 2605.20744) extended so
the environment can train a policy with RL, not only evaluate one. The consumer is
`~/projects/rl-exploration` (Qwen3-4B, GRPO on verl 0.6.1 inside `ariahw/rl-rewardhacking`).

## Where information lives

| Place | Holds |
|---|---|
| `STATUS.md` | Where the work stands and what happens next, in plain language. Read first. |
| `HANDOFF.md` | The detail a new session needs to continue: how the pieces fit, how to verify and run them, what bit us, what was decided and why. |
| `README.md` | What the environment is, how to run the paper's evaluation, and the RL layer's entry points. |
| Module docstrings in `hvta/` | The design decision behind each module (why detection is by content, why the leak is game-state only, why actions are the last bracket). Read them before changing behaviour. |
| `hvta/FilesystemWrapper/logical_bug_envs/README.md` | How to add a planted bug for a new game. |
| `tests/` | The behaviours that must hold; each file's docstring says which. |
| `integrations/rl-rewardhacking/` | Patches and config for plugging the env into the verl trainer, with their own README. |
| `experiments/` | Calibration and training runs, self-contained, one README each with method, tables and conclusions. |

Docs describe the current state. Delete stale text rather than appending updates.

## Working notes

- Python: `uv sync --extra dev --extra rl` builds `.venv` on 3.12 (`.python-version`, the pods'
  version); system python is too old. Behind the Claude sandbox set `UV_CACHE_DIR` into the
  scratchpad and `UV_PYTHON_INSTALL_DIR=~/projects/rl-envs/.uv-python`. Not the scratchpad: it
  vanishes with the session and leaves `.venv` pointing at an interpreter that no longer exists.
- The pods install this package from git, which builds a wheel (hatchling, `packages = ["hvta"]`,
  every file under `hvta/`). Anything needed at run time lives under `hvta/` and is found from
  `__file__`; nothing may depend on the repo root being on `sys.path`. `uv build --wheel` and
  `unzip -l` show what ships.
- NLTK corpora (`words`, `averaged_perceptron_tagger_eng`) must be present under a path in
  `nltk.data.path`; `.venv/nltk_data` works but is wiped whenever `.venv` is rebuilt. TextArena
  calls `nltk.download` unconditionally when several games are imported; `hvta.nltk_offline`
  short-circuits that when the corpus is installed. The downloader's index fetch fails behind the sandbox proxy, so fetch the
  zips from `raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/...` with curl.
- TextArena is pinned to commit `a2c896c` via `uv.lock`. Do not change env source there;
  wrap or shim from `hvta`.
- Tests: `.venv/bin/python -m pytest tests/ -q` (about 15 s, no network, no LLM).
- Serving a model for calibration: `scripts/serve_ow.py` (OpenWeights vLLM API job, run with
  the openweights tool venv's interpreter). Behind the Claude sandbox that venv's HTTP client
  cannot reach `*.proxy.runpod.net` while curl and this repo's `.venv` can, which is why the
  script probes readiness with curl and never calls `TemporaryApi.up()`.
- Games that need an LLM game master (GuessWho, TwentyQuestions) are not RL candidates.
  Minesweeper places mines after the first click, so it has no hidden solution at reset.
