from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import blake2b

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#.-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def term_overlap(left: str, right: str) -> float:
    a = set(tokenize(left))
    b = set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def contains_any(text: str, phrases: list[str] | tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if phrase and contains_phrase(text, phrase):
            return phrase
    return None


def hashed_vector(text: str, dims: int = 512) -> list[float]:
    vec = [0.0] * dims
    for token, count in Counter(tokenize(text)).items():
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))
