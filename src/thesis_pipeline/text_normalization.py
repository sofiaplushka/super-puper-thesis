from __future__ import annotations

import hashlib
import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic deduplication and hashing."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def normalize_for_hash(text: str | None) -> str:
    return normalize_text(text).casefold()


def text_hash(text: str | None) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def count_words(text: str | None) -> int:
    value = normalize_text(text)
    if not value:
        return 0
    return len(re.findall(r"\w+", value, flags=re.UNICODE))


def truncate_text(text: str | None, max_chars: int = 300) -> str:
    value = normalize_text(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"

