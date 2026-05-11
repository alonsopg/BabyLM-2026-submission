import torch

from src.masking.cms_morph_collator import CMSMorphCollator
from src.masking.count_min_sketch import CountMinSketch
from src.masking.morph_features import char_ngrams
from src.masking.random_collator import RandomMaskingCollator


def test_count_min_sketch_deterministic_and_serializable():
    sketch = CountMinSketch(width=32, depth=3, seed=5)
    sketch.update("running")
    sketch.update("running")
    sketch.update("jumping")
    assert sketch.estimate("running") >= 2.0
    restored = CountMinSketch.from_dict(sketch.to_dict())
    assert restored.estimate("running") == sketch.estimate("running")


def test_char_ngrams_normalize_and_filter():
    assert char_ngrams("Running") == ["<ru", "run", "unn", "nni", "nin", "ing", "ng>"]
    assert char_ngrams("##runner")[:2] == ["<ru", "run"]
    assert char_ngrams("[MASK]") == []
    assert char_ngrams("...") == []


def test_cms_morph_masks_same_number_as_random():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    examples = [{"input_ids": [2, 4, 6, 5, 11, 8, 5, 9, 3]}]
    random_batch = RandomMaskingCollator(tok, mask_rate=0.5, seed=7)(examples)
    cms_batch = CMSMorphCollator(tok, mask_rate=0.5, random_share=0.7, hard_token_share=0.2, hard_morph_share=0.1, seed=7)(examples)
    assert (random_batch["labels"] != -100).sum().item() == (cms_batch["labels"] != -100).sum().item()


def test_cms_morph_updates_and_uses_adaptive_masks():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    collator = CMSMorphCollator(
        tok,
        mask_rate=0.5,
        random_share=0.5,
        hard_token_share=0.25,
        hard_morph_share=0.25,
        adaptive_warmup_steps=0,
        update_percentile=0.5,
        seed=3,
    )
    metadata = {
        "masked_positions": [[1, 2]],
        "target_tokens": [["runs", "Alice"]],
        "target_morph_features": [[char_ngrams("runs"), char_ngrams("Alice")]],
    }
    losses = torch.zeros((1, 4))
    losses[0, 1] = 10.0
    losses[0, 2] = 1.0
    collator.update_error_sketch(metadata, losses)
    assert collator.token_cms.estimate("runs") > 0
    batch = collator([{"input_ids": [2, 4, 11, 5, 11, 8, 3]}])
    counts = batch["mask_metadata"]["source_counts"][0]
    assert counts["hard_token"] + counts["hard_morph"] > 0
