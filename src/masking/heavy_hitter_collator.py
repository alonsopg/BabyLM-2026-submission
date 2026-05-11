from __future__ import annotations

import random

import torch
from transformers import PreTrainedTokenizerBase

from .entity_extraction import EntityExtractor
from .random_collator import RandomMaskingCollator
from .sketches import SpaceSavingSketch


class HeavyHitterMaskingCollator(RandomMaskingCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_rate: float = 0.15,
        random_share: float = 0.50,
        entity_share: float = 0.25,
        error_share: float = 0.25,
        entity_sketch_capacity: int = 64,
        top_entities: int = 32,
        error_sketch_capacity: int = 10000,
        error_warmup_steps: int = 0,
        error_update_percentile: float = 0.75,
        randomized_sketch: bool = False,
        seed: int = 0,
    ):
        super().__init__(tokenizer=tokenizer, mask_rate=mask_rate, seed=seed)
        total = random_share + entity_share + error_share
        if abs(total - 1.0) > 1e-6:
            raise ValueError("masking shares must sum to 1.0")
        self.random_share = random_share
        self.entity_share = entity_share
        self.error_share = error_share
        self.entity_extractor = EntityExtractor(sketch_capacity=entity_sketch_capacity, top_entities=top_entities)
        self.error_sketch = SpaceSavingSketch(error_sketch_capacity)
        self.error_warmup_steps = error_warmup_steps
        self.error_update_percentile = error_update_percentile
        self.randomized_sketch = randomized_sketch
        self.step = 0

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor | list]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        masked_positions: list[list[int]] = []
        target_tokens: list[list[str]] = []
        source_counts: list[dict[str, int]] = []
        for row in range(input_ids.size(0)):
            valid = self._valid_positions(input_ids[row], batch.get("attention_mask", None)[row])
            tokens = self.tokenizer.convert_ids_to_tokens([int(input_ids[row, p]) for p in range(input_ids.size(1))])
            selected, counts = self._select_positions(tokens, valid)
            masked_positions.append(sorted(selected))
            source_counts.append(counts)
            target_tokens.append(self.tokenizer.convert_ids_to_tokens([int(input_ids[row, pos]) for pos in sorted(selected)]))
            for pos in selected:
                labels[row, pos] = input_ids[row, pos]
            self._apply_replacements(input_ids[row], selected)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["mask_metadata"] = {
            "masked_positions": masked_positions,
            "target_tokens": target_tokens,
            "source_counts": source_counts,
        }
        return batch

    def _select_positions(self, tokens: list[str], valid: list[int]) -> tuple[set[int], dict[str, int]]:
        n_mask = int(round(len(valid) * self.mask_rate))
        n_entity = int(round(n_mask * self.entity_share))
        n_random = int(round(n_mask * self.random_share))
        n_error = max(0, n_mask - n_entity - n_random)
        valid_set = set(valid)
        if self.randomized_sketch:
            shuffled = list(valid)
            self.rng.shuffle(shuffled)
            entity_positions = set(shuffled[: max(n_entity * 2, n_entity)])
            error_positions = set(shuffled[max(n_entity * 2, n_entity): max(n_entity * 2 + n_error * 2, n_entity + n_error)])
        else:
            entity_positions = self.entity_extractor.get_entity_positions(tokens) & valid_set
            error_positions = {p for p in valid if self._error_key(tokens, p) in self.error_sketch._counts}
        selected: set[int] = set()
        counts = {"entity": 0, "error": 0, "random": 0}
        counts["entity"] = self._sample_into(selected, sorted(entity_positions), n_entity)
        counts["error"] = self._sample_into(selected, sorted(error_positions - selected), n_error)
        counts["random"] = self._sample_into(selected, sorted(valid_set - selected), n_mask - len(selected))
        return selected, counts

    def _sample_into(self, selected: set[int], candidates: list[int], n: int) -> int:
        if n <= 0 or not candidates:
            return 0
        take = min(n, len(candidates))
        picked = self.rng.sample(candidates, take)
        selected.update(picked)
        return len(picked)

    def update_error_sketch(self, batch_metadata: dict, per_token_losses: torch.Tensor):
        self.step += 1
        if self.step <= self.error_warmup_steps:
            return
        flat: list[tuple[str, float]] = []
        for row, positions in enumerate(batch_metadata["masked_positions"]):
            tokens = batch_metadata["target_tokens"][row]
            for col, token in zip(positions, tokens):
                loss = float(per_token_losses[row, col].detach().cpu())
                flat.append((token, loss))
        if not flat:
            return
        losses = torch.tensor([loss for _, loss in flat])
        threshold = float(torch.quantile(losses, self.error_update_percentile))
        for token, loss in flat:
            if loss >= threshold:
                self.error_sketch.update(token, weight=max(loss, 1e-6))

    @staticmethod
    def _error_key(tokens: list[str], pos: int) -> str:
        return tokens[pos]
