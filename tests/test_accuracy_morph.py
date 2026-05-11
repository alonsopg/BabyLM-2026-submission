import torch

from src.masking.accuracy_morph_collator import AccuracyMorphCollator
from src.masking.morph_features import char_ngrams
from src.masking.random_collator import RandomMaskingCollator


def test_accuracy_morph_masks_same_number_as_random():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    examples = [{"input_ids": [2, 4, 6, 5, 11, 8, 5, 9, 3]}]
    random_batch = RandomMaskingCollator(tok, mask_rate=0.5, seed=7)(examples)
    acc_batch = AccuracyMorphCollator(tok, mask_rate=0.5, random_share=0.8, hard_token_share=0.15, hard_morph_share=0.05, seed=7)(examples)
    assert (random_batch["labels"] != -100).sum().item() == (acc_batch["labels"] != -100).sum().item()


def test_accuracy_morph_updates_from_wrong_predictions():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    collator = AccuracyMorphCollator(
        tok,
        mask_rate=0.5,
        random_share=0.5,
        hard_token_share=0.25,
        hard_morph_share=0.25,
        adaptive_warmup_steps=0,
        update_every_steps=1,
        min_seen=1,
        seed=3,
    )
    metadata = {
        "masked_positions": [[1, 2]],
        "target_tokens": [["runs", "Alice"]],
        "target_morph_features": [[char_ngrams("runs"), char_ngrams("Alice")]],
    }
    labels = torch.full((1, 4), -100)
    labels[0, 1] = tok.tok_to_id["runs"]
    labels[0, 2] = tok.tok_to_id["Alice"]
    pred_ids = labels.clone()
    pred_ids[0, 1] = tok.tok_to_id["Bob"]
    collator.update_from_predictions(metadata, pred_ids, labels)
    assert collator.token_seen_cms.estimate("runs") > 0
    assert collator.token_wrong_cms.estimate("runs") > 0
    assert collator.token_wrong_cms.estimate("alice") == 0
    batch = collator([{"input_ids": [2, 4, 11, 5, 11, 8, 3]}])
    counts = batch["mask_metadata"]["source_counts"][0]
    assert counts["hard_token"] + counts["hard_morph"] > 0
