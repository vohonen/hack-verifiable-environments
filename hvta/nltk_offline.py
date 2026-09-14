"""Make TextArena's unconditional ``nltk.download(...)`` calls offline-safe.

Several TextArena game modules call ``nltk.download`` at import or construction time
(Hangman, WordLadder, WordSearch, Wordle's dictionary). The downloader always fetches the
package index over the network first, so an env constructor fails without internet even
when the corpus is installed. ``install()`` replaces ``nltk.download`` with a version that
returns immediately when the package is already present and defers to the real downloader
otherwise. TextArena is pinned, so this lives here rather than upstream.
"""

import nltk

_REAL_DOWNLOAD = nltk.download
_RESOURCE_DIRS = ("corpora", "taggers", "tokenizers", "chunkers", "grammars", "misc", "models", "sentiment", "stemmers")


def is_installed(package_id: str) -> bool:
    for d in _RESOURCE_DIRS:
        try:
            nltk.data.find(f"{d}/{package_id}")
            return True
        except LookupError:
            continue
    return False


def offline_first_download(info_or_id=None, *args, **kwargs):
    if isinstance(info_or_id, str) and is_installed(info_or_id):
        return True
    return _REAL_DOWNLOAD(info_or_id, *args, **kwargs)


def install() -> None:
    if nltk.download is not offline_first_download:
        nltk.download = offline_first_download
