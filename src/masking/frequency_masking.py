from __future__ import annotations

from collections import Counter

import torch
from transformers import PreTrainedTokenizerBase

from .random_collator import RandomMaskingCollator


class FrequencyWeightedMaskingCollator(RandomMaskingCollator):
    """Batch-local frequency control: downweight most frequent types and sample mid-frequency tokens."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_rate: float = 0.15, seed: int = 0):
        super().__init__(tokenizer=tokenizer, mask_rate=mask_rate, seed=seed)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor | list]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        counts = Counter(int(x) for row in input_ids.tolist() for x in row if x not in self.tokenizer.all_special_ids)
        masked_positions, target_tokens = [], []
        for row in range(input_ids.size(0)):
            valid = self._valid_positions(input_ids[row], batch.get("attention_mask", None)[row])
            n_mask = int(round(len(valid) * self.mask_rate))
            weighted = []
            for pos in valid:
                freq = counts[int(input_ids[row, pos])]
                weight = 0.25 if freq >= 8 else 2.0 if 2 <= freq <= 5 else 1.0
                weighted.append((pos, weight))
            selected = self._weighted_sample(weighted, n_mask)
            masked_positions.append(sorted(selected))
            target_tokens.append(self.tokenizer.convert_ids_to_tokens([int(input_ids[row, pos]) for pos in sorted(selected)]))
            for pos in selected:
                labels[row, pos] = input_ids[row, pos]
            self._apply_replacements(input_ids[row], selected)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["mask_metadata"] = {"masked_positions": masked_positions, "target_tokens": target_tokens}
        return batch

    def _weighted_sample(self, weighted: list[tuple[int, float]], n: int) -> set[int]:
        pool = list(weighted)
        selected = set()
        for _ in range(min(n, len(pool))):
            total = sum(w for _, w in pool)
            r = self.rng.random() * total
            acc = 0.0
            for idx, (pos, weight) in enumerate(pool):
                acc += weight
                if acc >= r:
                    selected.add(pos)
                    pool.pop(idx)
                    break
        return selected
