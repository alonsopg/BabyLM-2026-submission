from __future__ import annotations

import math
from collections import Counter

import torch
from transformers import PreTrainedTokenizerBase

from .count_min_sketch import CountMinSketch
from .morph_features import char_ngrams, normalize_token
from .random_collator import RandomMaskingCollator


class CMSMorphCollator(RandomMaskingCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_rate: float = 0.15,
        random_share: float = 0.70,
        hard_token_share: float = 0.20,
        hard_morph_share: float = 0.10,
        token_cms_width: int = 65536,
        token_cms_depth: int = 4,
        morph_cms_width: int = 65536,
        morph_cms_depth: int = 4,
        conservative_update: bool = True,
        adaptive_warmup_steps: int = 0,
        update_percentile: float = 0.75,
        score_cap: float = 5.0,
        char_ngram_n: int = 3,
        add_boundaries: bool = True,
        update_weight: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(tokenizer=tokenizer, mask_rate=mask_rate, seed=seed)
        total = random_share + hard_token_share + hard_morph_share
        if abs(total - 1.0) > 1e-6:
            raise ValueError("masking shares must sum to 1.0")
        self.random_share = random_share
        self.hard_token_share = hard_token_share
        self.hard_morph_share = hard_morph_share
        self.token_cms = CountMinSketch(token_cms_width, token_cms_depth, seed=seed * 17 + 1, conservative_update=conservative_update)
        self.morph_cms = CountMinSketch(morph_cms_width, morph_cms_depth, seed=seed * 17 + 2, conservative_update=conservative_update)
        self.adaptive_warmup_steps = adaptive_warmup_steps
        self.update_percentile = update_percentile
        self.score_cap = score_cap
        self.char_ngram_n = char_ngram_n
        self.add_boundaries = add_boundaries
        self.update_weight = update_weight
        self.step = 0
        self.top_token_updates: Counter[str] = Counter()
        self.top_morph_updates: Counter[str] = Counter()
        self._id_token_cache: dict[int, str] = {}
        self._token_norm_cache: dict[str, str] = {}
        self._token_features_cache: dict[str, list[str]] = {}

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor | list]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        masked_positions: list[list[int]] = []
        target_tokens: list[list[str]] = []
        target_morph_features: list[list[list[str]]] = []
        source_counts: list[dict[str, int]] = []
        for row in range(input_ids.size(0)):
            valid = self._valid_positions(input_ids[row], batch.get("attention_mask", None)[row])
            tokens = {p: self._token_for_id(int(input_ids[row, p])) for p in valid}
            selected, counts = self._select_positions(tokens, valid)
            ordered = sorted(selected)
            masked_positions.append(ordered)
            source_counts.append(counts)
            row_tokens = [tokens[pos] for pos in ordered]
            target_tokens.append(row_tokens)
            target_morph_features.append([self._morph_features(token) for token in row_tokens])
            for pos in selected:
                labels[row, pos] = input_ids[row, pos]
            self._apply_replacements(input_ids[row], selected)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["mask_metadata"] = {
            "masked_positions": masked_positions,
            "target_tokens": target_tokens,
            "target_morph_features": target_morph_features,
            "source_counts": source_counts,
        }
        return batch

    def _select_positions(self, tokens: dict[int, str], valid: list[int]) -> tuple[set[int], dict[str, int]]:
        n_mask = int(round(len(valid) * self.mask_rate))
        if self.step < self.adaptive_warmup_steps:
            n_token = 0
            n_morph = 0
            n_random = n_mask
        else:
            n_token = int(round(n_mask * self.hard_token_share))
            n_random = int(round(n_mask * self.random_share))
            n_morph = max(0, n_mask - n_token - n_random)
        valid_set = set(valid)
        selected: set[int] = set()
        counts = {"entity": 0, "error": 0, "random": 0, "hard_token": 0, "hard_morph": 0, "fallback_random": 0}
        counts["hard_token"] = self._weighted_sample_into(selected, valid, [self._token_score(tokens[p]) for p in valid], n_token)
        remaining = [p for p in valid if p not in selected]
        counts["hard_morph"] = self._weighted_sample_into(selected, remaining, [self._morph_score(tokens[p]) for p in remaining], n_morph)
        before = len(selected)
        counts["random"] = self._sample_into(selected, sorted(valid_set - selected), n_mask - len(selected))
        counts["fallback_random"] = max(0, n_mask - before - int(round(n_mask * self.random_share)))
        # Keep the existing training diagnostics meaningful: token+morph are adaptive/error masks.
        counts["error"] = counts["hard_token"] + counts["hard_morph"]
        return selected, counts

    def _weighted_sample_into(self, selected: set[int], candidates: list[int], scores: list[float], n: int) -> int:
        if n <= 0 or not candidates:
            return 0
        pool = [(pos, float(score)) for pos, score in zip(candidates, scores) if pos not in selected and score > 0.0]
        count = 0
        while pool and count < n:
            total = sum(score for _, score in pool)
            if total <= 0:
                break
            threshold = self.rng.random() * total
            accum = 0.0
            picked_idx = len(pool) - 1
            for idx, (_, score) in enumerate(pool):
                accum += score
                if accum >= threshold:
                    picked_idx = idx
                    break
            pos, _ = pool.pop(picked_idx)
            selected.add(pos)
            count += 1
        return count

    def _sample_into(self, selected: set[int], candidates: list[int], n: int) -> int:
        if n <= 0 or not candidates:
            return 0
        take = min(n, len(candidates))
        picked = self.rng.sample(candidates, take)
        selected.update(picked)
        return len(picked)

    def _token_score(self, token: str) -> float:
        key = self._normalize_token(token)
        if not key:
            return 0.0
        return min(math.log1p(self.token_cms.estimate(key)), self.score_cap)

    def _morph_score(self, token: str) -> float:
        features = self._morph_features(token)
        if not features:
            return 0.0
        scores = [math.log1p(self.morph_cms.estimate(feature)) for feature in features]
        if not scores:
            return 0.0
        return min(sum(scores) / len(scores), self.score_cap)

    def _morph_features(self, token: str) -> list[str]:
        if token not in self._token_features_cache:
            self._token_features_cache[token] = char_ngrams(token, n=self.char_ngram_n, add_boundaries=self.add_boundaries)
        return self._token_features_cache[token]

    def _normalize_token(self, token: str) -> str:
        if token not in self._token_norm_cache:
            self._token_norm_cache[token] = normalize_token(token)
        return self._token_norm_cache[token]

    def _token_for_id(self, token_id: int) -> str:
        if token_id not in self._id_token_cache:
            try:
                token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            except TypeError:
                token = self.tokenizer.convert_ids_to_tokens([int(token_id)])
            if isinstance(token, list):
                token = token[0]
            self._id_token_cache[token_id] = token
        return self._id_token_cache[token_id]

    def update_error_sketch(self, batch_metadata: dict, per_token_losses: torch.Tensor):
        self.step += 1
        if self.step <= self.adaptive_warmup_steps:
            return
        flat: list[tuple[str, list[str], float]] = []
        for row, positions in enumerate(batch_metadata["masked_positions"]):
            tokens = batch_metadata["target_tokens"][row]
            features = batch_metadata.get("target_morph_features", [[] for _ in tokens])[row]
            for col, token, token_features in zip(positions, tokens, features):
                loss = float(per_token_losses[row, col].detach().cpu())
                flat.append((self._normalize_token(token), token_features, loss))
        if not flat:
            return
        losses = torch.tensor([loss for _, _, loss in flat])
        threshold = float(torch.quantile(losses, self.update_percentile))
        for token, features, loss in flat:
            if loss < threshold or not token:
                continue
            self.token_cms.update(token, self.update_weight)
            self.top_token_updates[token] += 1
            for feature in features:
                self.morph_cms.update(feature, self.update_weight)
                self.top_morph_updates[feature] += 1

    def to_state_dict(self) -> dict:
        return {
            "step": self.step,
            "token_cms": self.token_cms.to_dict(),
            "morph_cms": self.morph_cms.to_dict(),
            "top_token_updates": dict(self.top_token_updates.most_common(200)),
            "top_morph_updates": dict(self.top_morph_updates.most_common(200)),
        }

    def load_state_dict(self, state: dict):
        self.step = int(state.get("step", self.step))
        if "token_cms" in state:
            self.token_cms = CountMinSketch.from_dict(state["token_cms"])
        if "morph_cms" in state:
            self.morph_cms = CountMinSketch.from_dict(state["morph_cms"])
        self.top_token_updates = Counter({str(k): int(v) for k, v in state.get("top_token_updates", {}).items()})
        self.top_morph_updates = Counter({str(k): int(v) for k, v in state.get("top_morph_updates", {}).items()})
