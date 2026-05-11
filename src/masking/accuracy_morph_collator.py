from __future__ import annotations

from collections import Counter

import torch
from transformers import PreTrainedTokenizerBase

from .cms_morph_collator import CMSMorphCollator
from .count_min_sketch import CountMinSketch


class AccuracyMorphCollator(CMSMorphCollator):
    """Correctness-guided adaptive masking with token and character n-gram sketches."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_rate: float = 0.15,
        random_share: float = 0.80,
        hard_token_share: float = 0.15,
        hard_morph_share: float = 0.05,
        token_cms_width: int = 65536,
        token_cms_depth: int = 4,
        morph_cms_width: int = 65536,
        morph_cms_depth: int = 4,
        conservative_update: bool = True,
        adaptive_warmup_steps: int = 0,
        update_every_steps: int = 200,
        smoothing_alpha: float = 1.0,
        min_seen: float = 3.0,
        score_cap: float = 0.9,
        char_ngram_n: int = 3,
        add_boundaries: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mask_rate=mask_rate,
            random_share=random_share,
            hard_token_share=hard_token_share,
            hard_morph_share=hard_morph_share,
            token_cms_width=token_cms_width,
            token_cms_depth=token_cms_depth,
            morph_cms_width=morph_cms_width,
            morph_cms_depth=morph_cms_depth,
            conservative_update=conservative_update,
            adaptive_warmup_steps=adaptive_warmup_steps,
            update_percentile=0.75,
            score_cap=score_cap,
            char_ngram_n=char_ngram_n,
            add_boundaries=add_boundaries,
            update_weight=1.0,
            seed=seed,
        )
        self.token_seen_cms = CountMinSketch(token_cms_width, token_cms_depth, seed=seed * 19 + 1, conservative_update=conservative_update)
        self.token_wrong_cms = CountMinSketch(token_cms_width, token_cms_depth, seed=seed * 19 + 2, conservative_update=conservative_update)
        self.morph_seen_cms = CountMinSketch(morph_cms_width, morph_cms_depth, seed=seed * 19 + 3, conservative_update=conservative_update)
        self.morph_wrong_cms = CountMinSketch(morph_cms_width, morph_cms_depth, seed=seed * 19 + 4, conservative_update=conservative_update)
        self.update_every_steps = max(1, int(update_every_steps))
        self.smoothing_alpha = float(smoothing_alpha)
        self.min_seen = float(min_seen)
        self.pending_token_seen: Counter[str] = Counter()
        self.pending_token_wrong: Counter[str] = Counter()
        self.pending_morph_seen: Counter[str] = Counter()
        self.pending_morph_wrong: Counter[str] = Counter()
        self.top_token_updates = Counter()
        self.top_morph_updates = Counter()

    def _token_score(self, token: str) -> float:
        key = self._normalize_token(token)
        if not key:
            return 0.0
        return self._error_rate(self.token_wrong_cms.estimate(key), self.token_seen_cms.estimate(key))

    def _morph_score(self, token: str) -> float:
        features = self._morph_features(token)
        if not features:
            return 0.0
        scores = [
            self._error_rate(self.morph_wrong_cms.estimate(feature), self.morph_seen_cms.estimate(feature))
            for feature in features
        ]
        scores = [score for score in scores if score > 0.0]
        if not scores:
            return 0.0
        return min(sum(scores) / len(scores), self.score_cap)

    def _error_rate(self, wrong: float, seen: float) -> float:
        if seen < self.min_seen:
            return 0.0
        alpha = self.smoothing_alpha
        return min((wrong + alpha) / (seen + 2.0 * alpha), self.score_cap)

    def update_from_predictions(self, batch_metadata: dict, pred_ids: torch.Tensor, labels: torch.Tensor):
        self.step += 1
        for row, positions in enumerate(batch_metadata["masked_positions"]):
            tokens = batch_metadata["target_tokens"][row]
            features = batch_metadata.get("target_morph_features", [[] for _ in tokens])[row]
            for col, token, token_features in zip(positions, tokens, features):
                key = self._normalize_token(token)
                if not key:
                    continue
                label = int(labels[row, col])
                if label == -100:
                    continue
                wrong = int(pred_ids[row, col]) != label
                self.pending_token_seen[key] += 1
                if wrong:
                    self.pending_token_wrong[key] += 1
                    self.top_token_updates[key] += 1
                for feature in token_features:
                    self.pending_morph_seen[feature] += 1
                    if wrong:
                        self.pending_morph_wrong[feature] += 1
                        self.top_morph_updates[feature] += 1
        if self.step % self.update_every_steps == 0:
            self._flush_pending()

    def update_error_sketch(self, batch_metadata: dict, per_token_losses: torch.Tensor):
        raise RuntimeError("AccuracyMorphCollator expects update_from_predictions, not loss-based updates")

    def _flush_pending(self):
        for key, count in self.pending_token_seen.items():
            self.token_seen_cms.update(key, float(count))
        for key, count in self.pending_token_wrong.items():
            self.token_wrong_cms.update(key, float(count))
        for key, count in self.pending_morph_seen.items():
            self.morph_seen_cms.update(key, float(count))
        for key, count in self.pending_morph_wrong.items():
            self.morph_wrong_cms.update(key, float(count))
        self.pending_token_seen.clear()
        self.pending_token_wrong.clear()
        self.pending_morph_seen.clear()
        self.pending_morph_wrong.clear()

    def to_state_dict(self) -> dict:
        self._flush_pending()
        return {
            "step": self.step,
            "token_seen_cms": self.token_seen_cms.to_dict(),
            "token_wrong_cms": self.token_wrong_cms.to_dict(),
            "morph_seen_cms": self.morph_seen_cms.to_dict(),
            "morph_wrong_cms": self.morph_wrong_cms.to_dict(),
            "top_token_updates": dict(self.top_token_updates.most_common(200)),
            "top_morph_updates": dict(self.top_morph_updates.most_common(200)),
        }

    def load_state_dict(self, state: dict):
        self.step = int(state.get("step", self.step))
        if "token_seen_cms" in state:
            self.token_seen_cms = CountMinSketch.from_dict(state["token_seen_cms"])
        if "token_wrong_cms" in state:
            self.token_wrong_cms = CountMinSketch.from_dict(state["token_wrong_cms"])
        if "morph_seen_cms" in state:
            self.morph_seen_cms = CountMinSketch.from_dict(state["morph_seen_cms"])
        if "morph_wrong_cms" in state:
            self.morph_wrong_cms = CountMinSketch.from_dict(state["morph_wrong_cms"])
        self.top_token_updates = Counter({str(k): int(v) for k, v in state.get("top_token_updates", {}).items()})
        self.top_morph_updates = Counter({str(k): int(v) for k, v in state.get("top_morph_updates", {}).items()})
