import torch

from src.masking.heavy_hitter_collator import HeavyHitterMaskingCollator


def test_error_sketch_inactive_during_warmup():
    from conftest import TinyTokenizer

    collator = HeavyHitterMaskingCollator(TinyTokenizer(), error_warmup_steps=2, seed=3)
    metadata = {"masked_positions": [[1, 2]], "target_tokens": [["Alice", "Bob"]]}
    losses = torch.zeros((1, 4))
    losses[0, 1] = 10.0
    losses[0, 2] = 9.0
    collator.update_error_sketch(metadata, losses)
    assert collator.error_sketch.topk(10) == []


def test_error_sketch_updates_after_warmup():
    from conftest import TinyTokenizer

    collator = HeavyHitterMaskingCollator(TinyTokenizer(), error_warmup_steps=0, error_update_percentile=0.5, seed=3)
    metadata = {"masked_positions": [[1, 2]], "target_tokens": [["Alice", "Bob"]]}
    losses = torch.zeros((1, 4))
    losses[0, 1] = 10.0
    losses[0, 2] = 1.0
    collator.update_error_sketch(metadata, losses)
    assert collator.error_sketch.contains("Alice")
