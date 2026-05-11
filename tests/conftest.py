from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TinyTokenizer:
    mask_token_id = 1
    pad_token_id = 0
    all_special_ids = [0, 1, 2, 3]

    def __init__(self):
        vocab = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "Alice", "Bob", "saw", "the", "rabbit", "again", "Paris", "runs", "."]
        self.id_to_tok = dict(enumerate(vocab))
        self.tok_to_id = {v: k for k, v in self.id_to_tok.items()}

    def __len__(self):
        return len(self.id_to_tok)

    def pad(self, examples, return_tensors=None):
        import torch

        max_len = max(len(x["input_ids"]) for x in examples)
        ids, attn = [], []
        for ex in examples:
            row = ex["input_ids"] + [self.pad_token_id] * (max_len - len(ex["input_ids"]))
            ids.append(row)
            attn.append([1] * len(ex["input_ids"]) + [0] * (max_len - len(ex["input_ids"])))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_tok[int(i)] for i in ids]
