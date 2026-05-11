from __future__ import annotations

import random

import torch
from transformers import PreTrainedTokenizerBase


class RandomMaskingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_rate: float = 0.15, seed: int = 0):
        self.tokenizer = tokenizer
        self.mask_rate = mask_rate
        self.rng = random.Random(seed)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor | list]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        masked_positions: list[list[int]] = []
        target_tokens: list[list[str]] = []
        for row in range(input_ids.size(0)):
            valid = self._valid_positions(input_ids[row], batch.get("attention_mask", None)[row])
            n_mask = int(round(len(valid) * self.mask_rate))
            selected = set(self.rng.sample(valid, min(n_mask, len(valid))))
            masked_positions.append(sorted(selected))
            target_tokens.append(self.tokenizer.convert_ids_to_tokens([int(input_ids[row, pos]) for pos in sorted(selected)]))
            for pos in selected:
                labels[row, pos] = input_ids[row, pos]
            self._apply_replacements(input_ids[row], selected)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["mask_metadata"] = {"masked_positions": masked_positions, "target_tokens": target_tokens}
        return batch

    def _valid_positions(self, ids: torch.Tensor, attention: torch.Tensor | None) -> list[int]:
        specials = set(self.tokenizer.all_special_ids)
        valid = []
        for idx, token_id in enumerate(ids.tolist()):
            if attention is not None and int(attention[idx]) == 0:
                continue
            if token_id in specials:
                continue
            valid.append(idx)
        return valid

    def _apply_replacements(self, ids: torch.Tensor, positions: set[int]):
        vocab_size = len(self.tokenizer)
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("tokenizer must define mask_token_id")
        for pos in positions:
            r = self.rng.random()
            if r < 0.8:
                ids[pos] = mask_id
            elif r < 0.9:
                ids[pos] = self.rng.randrange(vocab_size)
