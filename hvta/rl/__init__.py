"""RL layer over Hack-Verifiable TextArena.

``HVTAEpisode`` turns one game plus its virtual filesystem into a chat: the policy's
message is an action, the environment's reply is the next user turn, and the reward is
the game's own reward, unchanged. Hack flags and onset steps ride along in the episode
record and never enter the reward.
"""

from .actions import extract_action, normalise_action, strip_thinking
from .episode import DEFAULT_SYSTEM_PROMPT, HVTAEpisode, WrapperPool, format_observation
from .metrics import EpisodeRecord, aggregate
from .tasks import TaskSpec, default_max_steps, game_name_of_env_id, make_tasks

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "EpisodeRecord",
    "HVTAEpisode",
    "TaskSpec",
    "WrapperPool",
    "aggregate",
    "default_max_steps",
    "extract_action",
    "format_observation",
    "game_name_of_env_id",
    "make_tasks",
    "normalise_action",
    "strip_thinking",
]
