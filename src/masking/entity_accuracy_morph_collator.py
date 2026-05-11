from __future__ import annotations

from collections import Counter

from transformers import PreTrainedTokenizerBase

from .accuracy_morph_collator import AccuracyMorphCollator
from .entity_extraction import STOPWORDS


class EntityAccuracyMorphCollator(AccuracyMorphCollator):
    """Correctness-guided masking with repeated-token/entity bias and late decay."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_rate: float = 0.15,
        random_share: float = 0.80,
        hard_token_share: float = 0.15,
        hard_morph_share: float = 0.05,
        final_random_share: float = 0.90,
        total_steps: int = 10000,
        decay_start_fraction: float = 0.80,
        entity_bonus: float = 0.25,
        morph_tiebreak_bonus: float = 0.10,
        min_error_rate: float = 0.25,
        max_error_rate: float = 0.85,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mask_rate=mask_rate,
            random_share=random_share,
            hard_token_share=hard_token_share,
            hard_morph_share=hard_morph_share,
            **kwargs,
        )
        self.base_random_share = float(random_share)
        self.base_hard_token_share = float(hard_token_share)
        self.base_hard_morph_share = float(hard_morph_share)
        self.final_random_share = float(final_random_share)
        self.total_steps = max(1, int(total_steps))
        self.decay_start_step = int(self.total_steps * decay_start_fraction)
        self.entity_bonus = float(entity_bonus)
        self.morph_tiebreak_bonus = float(morph_tiebreak_bonus)
        self.min_error_rate = float(min_error_rate)
        self.max_error_rate = float(max_error_rate)

    def _select_positions(self, tokens: dict[int, str], valid: list[int]) -> tuple[set[int], dict[str, int]]:
        n_mask = int(round(len(valid) * self.mask_rate))
        random_share, token_share, morph_share = self._current_shares()
        if self.step < self.adaptive_warmup_steps:
            n_token = 0
            n_morph = 0
        else:
            n_token = int(round(n_mask * token_share))
            n_morph = int(round(n_mask * morph_share))
        n_random = max(0, n_mask - n_token - n_morph)
        del n_random

        valid_set = set(valid)
        selected: set[int] = set()
        entity_positions = self._entity_positions(tokens, valid)
        counts = {"entity": 0, "error": 0, "random": 0, "hard_token": 0, "hard_morph": 0, "fallback_random": 0}
        token_scores = [self._position_token_score(tokens[p], p in entity_positions) for p in valid]
        counts["hard_token"] = self._weighted_sample_into(selected, valid, token_scores, n_token)
        remaining = [p for p in valid if p not in selected]
        morph_scores = [self._position_morph_score(tokens[p]) for p in remaining]
        counts["hard_morph"] = self._weighted_sample_into(selected, remaining, morph_scores, n_morph)
        before_random = len(selected)
        counts["random"] = self._sample_into(selected, sorted(valid_set - selected), n_mask - len(selected))
        counts["fallback_random"] = max(0, n_mask - before_random - int(round(n_mask * random_share)))
        counts["entity"] = sum(1 for pos in selected if pos in entity_positions)
        counts["error"] = counts["hard_token"] + counts["hard_morph"]
        return selected, counts

    def _current_shares(self) -> tuple[float, float, float]:
        if self.step < self.adaptive_warmup_steps or self.decay_start_step >= self.total_steps:
            return self.base_random_share, self.base_hard_token_share, self.base_hard_morph_share
        if self.step <= self.decay_start_step:
            return self.base_random_share, self.base_hard_token_share, self.base_hard_morph_share
        progress = min(1.0, (self.step - self.decay_start_step) / max(1, self.total_steps - self.decay_start_step))
        random_share = self.base_random_share + progress * (self.final_random_share - self.base_random_share)
        adaptive_total = max(0.0, 1.0 - random_share)
        base_adaptive = self.base_hard_token_share + self.base_hard_morph_share
        token_share = adaptive_total * (self.base_hard_token_share / base_adaptive)
        morph_share = adaptive_total * (self.base_hard_morph_share / base_adaptive)
        return random_share, token_share, morph_share

    def _position_token_score(self, token: str, is_entity: bool) -> float:
        key = self._normalize_token(token)
        if not key:
            return 0.0
        rate = self._raw_error_rate(self.token_wrong_cms.estimate(key), self.token_seen_cms.estimate(key))
        if rate <= 0.0 or rate < self.min_error_rate or rate > self.max_error_rate:
            return 0.0
        morph_score = self._position_morph_score(token)
        score = rate + self.morph_tiebreak_bonus * morph_score
        if is_entity:
            score += self.entity_bonus
        return min(score, self.score_cap)

    def _position_morph_score(self, token: str) -> float:
        features = self._morph_features(token)
        if not features:
            return 0.0
        scores = [
            self._raw_error_rate(self.morph_wrong_cms.estimate(feature), self.morph_seen_cms.estimate(feature))
            for feature in features
        ]
        scores = [score for score in scores if self.min_error_rate <= score <= self.max_error_rate]
        if not scores:
            return 0.0
        return min(sum(scores) / len(scores), self.score_cap)

    def _raw_error_rate(self, wrong: float, seen: float) -> float:
        if seen < self.min_seen:
            return 0.0
        alpha = self.smoothing_alpha
        return (wrong + alpha) / (seen + 2.0 * alpha)

    def _entity_positions(self, tokens: dict[int, str], valid: list[int]) -> set[int]:
        counts: Counter[str] = Counter()
        pos_to_key: dict[int, str] = {}
        for pos in valid:
            key = self._normalize_token(tokens[pos])
            if not self._is_entity_candidate(key):
                continue
            counts[key] += 1
            pos_to_key[pos] = key
        repeated = {key for key, count in counts.items() if count >= 2}
        return {pos for pos, key in pos_to_key.items() if key in repeated}

    @staticmethod
    def _is_entity_candidate(token: str) -> bool:
        if len(token) < 4 or not token.isalpha():
            return False
        return token not in STOPWORDS
