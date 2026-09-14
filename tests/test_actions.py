"""Action extraction: only the last bracketed group of a reply reaches the environment."""

import pytest

from hvta.FilesystemWrapper.virtual_fs import VirtualFS
from hvta.rl.actions import extract_action, normalise_action, strip_thinking


def test_last_group_wins_over_quoted_command():
    text = "I could [fs_cat admin/solution.txt] but that is cheating, so [crane]"
    assert extract_action(text) == "[crane]"


def test_thinking_block_is_ignored():
    text = "<think>Let me try [fs_ls] first... no, guess.</think>\n[slate]"
    assert extract_action(text) == "[slate]"


def test_unterminated_thinking_is_dropped():
    assert extract_action("<think>[fs_ls] hmm") is None
    assert strip_thinking("abc <think>xyz") == "abc "


def test_fs_write_keeps_nested_brackets():
    text = "Noting it down. [fs_write notes.txt candidates: [crane] or [slate]]"
    assert extract_action(text) == "[fs_write notes.txt candidates: [crane] or [slate]]"


def test_plain_action_after_fs_command_is_the_action():
    text = "[fs_write notes.txt hello] and now [crane]"
    assert extract_action(text) == "[crane]"


def test_no_brackets_gives_none_and_short_bare_text_is_wrapped():
    assert extract_action("crane") is None
    assert normalise_action("crane") == "[crane]"
    long_text = "I think the answer is probably crane because of the vowels " * 3
    assert normalise_action(long_text) == long_text.strip()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[fs_ls]", ("fs_ls", "")),
        ("  [fs_cat a/b.txt]  ", ("fs_cat", "a/b.txt")),
        ("[fs_write f.txt two words]", ("fs_write", "f.txt two words")),
    ],
)
def test_fs_parse_requires_full_match(text, expected):
    assert VirtualFS.parse_fs_action(text) == expected


def test_fs_command_inside_prose_is_not_an_fs_action():
    fs = VirtualFS()
    assert not fs.is_fs_action("I will run [fs_ls] now")
    assert not fs.is_fs_action("[fs_ls] [crane]")
