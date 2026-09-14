"""The verl-facing loop core: token bookkeeping and masks, with a stub tokenizer and generator.

No verl here. The stub tokenizer renders a chat the way Qwen3's template shapes it (role
headers as single tokens, no default system block) so prefix stripping and mask alignment
are exercised on realistic structure.
"""

import asyncio

import pytest

from hvta.rl import TaskSpec, WrapperPool
from hvta.rl.policies import OracleHackPolicy
from hvta.rl.verl_agent_loop import RECORD_COLUMNS, HVTALoopCore, aggregate_columns, record_to_columns, task_from_row

IM_START, IM_END, NL = -1, -2, -3
ROLE = {"system": -10, "user": -11, "assistant": -12}


class StubTokenizer:
    """Char-level tokens; special tokens are negative ints; decodes back to text."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        ids = []
        for m in messages:
            role = m.get("role")
            if role not in ROLE:  # the [{}] probe renders nothing, as Qwen3 does
                continue
            ids += [IM_START, ROLE[role], NL] + [ord(c) for c in m["content"]] + [IM_END, NL]
        if add_generation_prompt:
            ids += [IM_START, ROLE["assistant"], NL]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i >= 0)


def encode_reply(text: str) -> list[int]:
    return [ord(c) for c in text] + [IM_END]


def make_generate(policy, tokenizer):
    """A generator that runs a scripted policy over the decoded chat."""

    async def generate(prompt_ids, max_tokens):
        # Rebuild the chat from the token stream: split on IM_START, keep role + content.
        messages, cur = [], None
        for t in prompt_ids:
            if t == IM_START:
                cur = {"role": None, "content": []}
                messages.append(cur)
            elif cur is not None and cur["role"] is None and t in ROLE.values():
                cur["role"] = next(k for k, v in ROLE.items() if v == t)
            elif cur is not None and t >= 0:
                cur["content"].append(chr(t))
        chat = [{"role": m["role"], "content": "".join(m["content"])} for m in messages if m["role"]]
        chat = [m for m in chat if not (m["role"] == "assistant" and m["content"] == "")]
        reply = policy(chat)
        return encode_reply(reply)[:max_tokens]

    return generate


@pytest.fixture
def core():
    return HVTALoopCore(StubTokenizer(), response_length=20000, max_tokens_per_turn=512, pool=WrapperPool())


def test_masks_align_and_reward_flows(core):
    task = TaskSpec("GuessTheNumber-v0", seed=2, hidden_solution=True)
    policy = OracleHackPolicy(task.game)
    result = asyncio.run(core.run(task, make_generate(policy, core.tokenizer)))

    assert len(result.response_ids) == len(result.response_mask)
    assert result.reward == 1.0 and result.record.hack_triggered
    assert not result.budget_truncated
    # every policy token is masked 1 and every environment token 0
    policy_ids = [t for t, m in zip(result.response_ids, result.response_mask) if m == 1]
    env_ids = [t for t, m in zip(result.response_ids, result.response_mask) if m == 0]
    assert core.tokenizer.decode(policy_ids).startswith("[fs_ls /home/user]")
    assert "internal variables of environment" in core.tokenizer.decode(env_ids)
    # environment turns start with the user header, never with a stray default-system block
    first_env_span = result.response_mask.index(0)
    assert result.response_ids[first_env_span : first_env_span + 2] == [IM_START, ROLE["user"]]
    assert result.num_turns == result.record.n_steps * 2 + 1
    assert set(result.columns) == set(RECORD_COLUMNS)
    assert result.columns["hvta_hack_triggered"] == 1.0 and result.columns["hvta_hack_onset_step"] >= 0
    assert result.columns["hvta_response_tokens"] == sum(result.response_mask)
    agg = aggregate_columns({k: [v] for k, v in result.columns.items()})
    assert agg["detail/hvta/n_hack"] == 1.0 and agg["detail/hvta/GuessTheNumber/n_hacked_win"] == 1.0


def test_budget_exhaustion_truncates_with_negative_reward():
    core = HVTALoopCore(StubTokenizer(), response_length=60, max_tokens_per_turn=512, pool=WrapperPool())
    task = TaskSpec("GuessTheNumber-v0", seed=0, hidden_solution=True, max_steps=30)

    async def generate(prompt_ids, max_tokens):
        return encode_reply("[fs_ls]")[:max_tokens]

    result = asyncio.run(core.run(task, generate))
    assert result.budget_truncated
    assert result.record.truncated and result.reward == -1.0
    assert len(result.response_ids) <= 60 and len(result.response_ids) == len(result.response_mask)
    assert result.columns["hvta_budget_truncated"] == 1.0


def test_max_assistant_turns_caps_the_episode():
    core = HVTALoopCore(StubTokenizer(), response_length=20000, max_assistant_turns=2, pool=WrapperPool())
    task = TaskSpec("GuessTheNumber-v0", seed=0, max_steps=30)

    async def generate(prompt_ids, max_tokens):
        return encode_reply("[fs_pwd]")

    result = asyncio.run(core.run(task, generate))
    assert result.budget_truncated and result.record.n_steps == 2


def test_task_roundtrip_through_row_json():
    task = TaskSpec("Wordle-v0", seed=5, hidden_solution=True, max_fs_steps=7)
    row_extra = {"task": __import__("json").dumps(task.to_dict())}
    assert task_from_row(row_extra) == task
    assert task_from_row(__import__("json").dumps(row_extra)) == task


def test_missing_steps_encode_as_minus_one():
    from hvta.rl import HVTAEpisode

    ep = HVTAEpisode(TaskSpec("GuessTheNumber-v0", seed=0, hidden_solution=True))
    ep.reset()
    ep.step("[5]")
    ep.force_end("test")
    cols = record_to_columns(ep.record, num_turns=3, budget_truncated=True)
    assert cols["hvta_hack_onset_step"] == -1 and cols["hvta_readme_read_step"] == -1
    assert cols["hvta_hack_triggered"] == 0.0 and cols["hvta_truncated"] == 1.0
