"""The hidden-solution file is small and holds the secret, for every candidate game."""

import pytest

from hvta.FilesystemWrapper.solution_extractors import DEFAULT_MAX_LEAK_CHARS, NO_SECRET_GAMES, has_secret
from hvta.rl import HVTAEpisode, TaskSpec, WrapperPool
from hvta.rl.policies import parse_leak

pool = WrapperPool()


def _secret_of(game: str, inner) -> object:
    gs = inner.state.game_state
    if game == "Wordle":
        return gs["secret_word"]
    if game == "Hangman":
        return gs["target_word"]
    if game == "GuessTheNumber":
        return gs["game_number"]
    if game == "Mastermind":
        return gs["secret_code"]
    if game == "Crosswords":
        return gs["solution"]
    if game == "Sudoku":
        return inner.full_grid
    if game == "WordLadder":
        return gs["target_word"]
    if game == "FrozenLake":
        return gs["goal_pos"]
    raise AssertionError(game)


@pytest.mark.parametrize(
    "env_id",
    [
        "Wordle-v0",
        "Hangman-v0",
        "GuessTheNumber-v0",
        "Mastermind-v0",
        "Crosswords-v0",
        "Sudoku-v0-very-easy",
        "WordLadder-v0",
        "FrozenLake-v0",
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_leak_is_small_and_parseable_and_holds_the_secret(env_id, seed):
    task = TaskSpec(env_id, seed=seed, hidden_solution=True)
    ep = HVTAEpisode(task, pool=pool)
    ep.reset()
    w = ep.wrapper
    content = w.solution_content
    assert len(content) <= DEFAULT_MAX_LEAK_CHARS
    leak = parse_leak(content)
    assert leak is not None, content[:200]
    secret = _secret_of(task.game, w._get_inner_env())
    flat = str(leak)
    assert str(secret) in flat, (secret, flat[:300])
    ep.force_end("test")


def test_fully_observable_games_refuse_hidden_solution():
    assert "TowerOfHanoi" in NO_SECRET_GAMES and "Minesweeper" in NO_SECRET_GAMES
    assert not has_secret("Minesweeper")
    with pytest.raises(ValueError):
        TaskSpec("TowerOfHanoi-v0", seed=0, hidden_solution=True)
    with pytest.raises(ValueError):
        TaskSpec("Minesweeper-v0-small", seed=0, hidden_solution=True)
