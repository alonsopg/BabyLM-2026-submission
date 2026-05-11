from __future__ import annotations

import re
from dataclasses import dataclass

from .sketches import SpaceSavingSketch


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "so", "that", "the", "their", "them", "there", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "you", "your",
}


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    normalized_text: str
    token_positions: tuple[int, ...]


class EntityExtractor:
    def __init__(self, min_count: int = 2, top_entities: int = 32, sketch_capacity: int = 64):
        self.min_count = min_count
        self.top_entities = top_entities
        self.sketch_capacity = sketch_capacity

    def extract_candidates(self, tokens: list[str]) -> list[EntityCandidate]:
        candidates: list[EntityCandidate] = []
        for idx, token in enumerate(tokens):
            clean = self._clean_token(token)
            if not self._is_candidate(clean):
                continue
            candidates.append(EntityCandidate(clean, clean.lower(), (idx,)))
        return candidates

    def get_entity_positions(self, tokens: list[str]) -> set[int]:
        sketch = SpaceSavingSketch(self.sketch_capacity)
        candidates = self.extract_candidates(tokens)
        for candidate in candidates:
            sketch.update(candidate.normalized_text)
        top = {key for key, count in sketch.topk(self.top_entities) if count >= self.min_count}
        positions: set[int] = set()
        for candidate in candidates:
            if candidate.normalized_text in top:
                positions.update(candidate.token_positions)
        return positions

    @staticmethod
    def _clean_token(token: str) -> str:
        token = token.replace("Ġ", "").replace("▁", "").replace("##", "")
        return re.sub(r"^[^\w]+|[^\w]+$", "", token)

    @staticmethod
    def _is_candidate(token: str) -> bool:
        if len(token) < 2 or not token.isalpha():
            return False
        lower = token.lower()
        if lower in STOPWORDS:
            return False
        return token[:1].isupper() or len(token) >= 4
