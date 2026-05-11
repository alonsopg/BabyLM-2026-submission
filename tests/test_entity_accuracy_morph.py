import torch

from src.masking.entity_accuracy_morph_collator import EntityAccuracyMorphCollator
from src.masking.morph_features import char_ngrams


def test_entity_accuracy_morph_uses_entity_bonus_and_decay():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    collator = EntityAccuracyMorphCollator(
        tok,
        mask_rate=0.8,
        random_share=0.5,
        hard_token_share=0.25,
        hard_morph_share=0.25,
        final_random_share=0.75,
        total_steps=10,
        decay_start_fraction=0.5,
        adaptive_warmup_steps=0,
        update_every_steps=1,
        min_seen=1,
        min_error_rate=0.1,
        max_error_rate=0.95,
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

    batch = collator([{"input_ids": [2, 11, 4, 11, 5, 11, 3]}])
    counts = batch["mask_metadata"]["source_counts"][0]
    assert counts["hard_token"] + counts["hard_morph"] > 0
    assert counts["entity"] > 0

    collator.step = 10
    random_share, token_share, morph_share = collator._current_shares()
    assert random_share == 0.75
    assert round(token_share + morph_share, 6) == 0.25
