from __future__ import annotations

import re


_TOKEN_PREFIXES = ("##", "Ġ", "▁")


def normalize_token(token: str) -> str:
    token = str(token)
    for prefix in _TOKEN_PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix) :]
    return token.lower()


def char_ngrams(token: str, n: int = 3, add_boundaries: bool = True) -> list[str]:
    token = normalize_token(token)
    if token.startswith("[") and token.endswith("]"):
        return []
    if not re.search(r"[a-z]", token):
        return []
    if sum(ch.isalpha() for ch in token) < 2:
        return []
    if add_boundaries:
        token = f"<{token}>"
    if len(token) < n:
        return []
    return [token[i : i + n] for i in range(len(token) - n + 1)]
