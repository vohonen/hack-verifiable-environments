# Status

Last updated 2026-09-14. Plain-language summary; the detail for continuing is in `HANDOFF.md`.

## What this is

- A set of text games (Wordle, Hangman, Sudoku, ...) where the answer is hidden in a fake
  filesystem the player can browse. A README next to it says reading it is cheating.
- We train a small model (Qwen3-4B) on these games with RL and watch whether it learns to
  read the answer instead of playing. Every episode records whether it cheated and when.
- The training itself runs in the separate project `~/projects/rl-exploration`. This repo
  provides the environment and two patches that plug it into that project's trainer.

## Where things are

- `hvta/rl/`: the environment as a multi-turn chat, plus the piece the trainer calls.
- `scripts/`: serve a model on a rented GPU, measure how often it cheats, build the training data.
- `integrations/rl-rewardhacking/`: the two trainer patches and how to use them.
- `experiments/001-calibration/`: how often the untrained model cheats and wins, per game.

## Done

- Environment and RL layer work; 80 tests pass.
- Trainer patches are written and checked against the current `rl-exploration` (its own
  dry run accepts them).
- Measured the untrained model on every candidate game (3712 episodes, about $2.40).
- Fixed two bugs the measurement exposed: short games could not be finished after browsing
  the filesystem, and the measurement script did not enforce the trainer's length limit.

## What the measurement said

- Games chosen for training: Sudoku (very easy), Mastermind, Hangman, Wordle. The model
  cheats in a few percent of episodes on each and wins honestly only on Sudoku (about 15 %).
- Hidden 3 directories deep. At depth 1 it cheats in about 8 % of episodes, which is too
  easy; at depth 3 about 4.5 %. The next knob to make it harder is fewer filesystem commands
  per episode (currently 12).
- The other cheat type (a planted bug in the game code) is parked: the untrained model trips
  those bugs by accident in most episodes and never wins, so there is nothing to learn from.
- The model loses most games by repeating an invalid move twice. RL should fix that quickly,
  but it means most episodes score -1 at the start.

## Next steps

1. Vili: push this branch; in `rl-exploration`, apply `integrations/rl-rewardhacking/rl-exploration-wiring.patch`,
   commit, push, and build the GPU image (manual GitHub action). Commands are in
   `integrations/rl-rewardhacking/README.md`.
2. Smoke run: 5 training steps through the job queue, no evaluation. This is the first time
   the trainer's multi-turn path runs at all; expect to debug it.
3. If the smoke run works: one 200-step run, then 3 seeds, read with the usual onset and
   hazard tools in `rl-exploration`.
4. Optional before 3: re-measure the four games with 6 or 8 filesystem commands per episode
   (15 minutes, about $1) if a lower starting cheat rate is wanted.

## Open questions

- Should GuessTheNumber be added as a game the model can win honestly but never cheats on?
- Is a 4-5 % starting cheat rate acceptable, or should the filesystem budget be cut first?
- The bug-based cheat type needs a stricter definition of "cheated" before it is usable.
