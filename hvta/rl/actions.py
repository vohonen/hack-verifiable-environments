"""Turn a model's free-form reply into the one action the environment executes.

TextArena games search their action regex over the whole reply and take the first
match; the paper's filesystem wrapper did the same for ``[fs_*]`` commands. A policy that
reasons in prose ("I could [fs_cat admin/solution.txt] but that is cheating, so [crane]")
would then execute the quoted command. The rule here: the action is the **last**
bracketed group of the reply after any thinking block is removed, and only that group
reaches the environment.
"""

import re
from typing import Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_BRACKET_GROUP = re.compile(r"\[[^\[\]]*\]")
_FS_PREFIX = "[fs_"

MAX_BARE_ACTION_CHARS = 64


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks, and an unterminated ``<think>`` tail."""
    text = _THINK_BLOCK.sub("", text)
    open_idx = text.rfind("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text


def _last_fs_command(text: str) -> Optional[tuple[int, int]]:
    """Span of the last ``[fs_...]`` group, matching nested brackets (fs_write content)."""
    k = text.rfind(_FS_PREFIX)
    if k == -1:
        return None
    depth = 0
    for i in range(k, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return k, i + 1
    return None


def extract_action(text: str) -> Optional[str]:
    """The last bracketed group of the reply, or None if there is none.

    A filesystem command that itself contains brackets (``[fs_write f.txt see [x]]``) is
    returned whole when the last plain group sits inside it.
    """
    s = strip_thinking(text)
    last_plain = None
    for last_plain in _BRACKET_GROUP.finditer(s):
        pass
    fs_span = _last_fs_command(s)
    if fs_span is not None:
        k, m = fs_span
        if last_plain is None or (last_plain.start() >= k and last_plain.end() <= m):
            return s[k:m]
    if last_plain is None:
        return None
    return last_plain.group()


def normalise_action(text: str) -> str:
    """What the environment receives for a reply.

    Bracketed reply: the extracted action. Short bare reply with no brackets: wrapped in
    brackets, as TextArena's ``ActionFormattingWrapper`` would. Anything else: the
    stripped text, which the game will reject as an invalid move.
    """
    action = extract_action(text)
    if action is not None:
        return action
    s = strip_thinking(text).strip()
    if s and "[" not in s and "]" not in s and "\n" not in s and len(s) <= MAX_BARE_ACTION_CHARS:
        return f"[{s}]"
    return s
