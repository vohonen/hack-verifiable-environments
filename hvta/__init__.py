from hvta import nltk_offline as _nltk_offline

_nltk_offline.install()

from hvta.FilesystemWrapper import FilesystemWrapper  # noqa: E402

__all__ = ["FilesystemWrapper"]
