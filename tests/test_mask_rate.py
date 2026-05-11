from src.masking.heavy_hitter_collator import HeavyHitterMaskingCollator
from src.masking.random_collator import RandomMaskingCollator


def test_random_and_hh_entity_mask_same_number(tiny_tokenizer=None):
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    examples = [{"input_ids": [2, 4, 6, 5, 4, 8, 5, 9, 3]}]
    random_batch = RandomMaskingCollator(tok, mask_rate=0.5, seed=7)(examples)
    hhm_batch = HeavyHitterMaskingCollator(tok, mask_rate=0.5, random_share=0.75, entity_share=0.25, error_share=0.0, seed=7)(examples)
    assert (random_batch["labels"] != -100).sum().item() == (hhm_batch["labels"] != -100).sum().item()


def test_special_and_padding_tokens_are_never_masked():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    examples = [{"input_ids": [2, 4, 5, 3]}, {"input_ids": [2, 4, 3]}]
    batch = HeavyHitterMaskingCollator(tok, mask_rate=1.0, random_share=1.0, entity_share=0.0, error_share=0.0, seed=2)(examples)
    for special_id in tok.all_special_ids:
        assert not ((batch["labels"] == special_id).any().item())


def test_same_seed_is_deterministic():
    from conftest import TinyTokenizer

    tok = TinyTokenizer()
    examples = [{"input_ids": [2, 4, 6, 5, 4, 8, 5, 9, 3]}]
    a = HeavyHitterMaskingCollator(tok, mask_rate=0.5, random_share=0.75, entity_share=0.25, error_share=0.0, seed=11)(examples)
    b = HeavyHitterMaskingCollator(tok, mask_rate=0.5, random_share=0.75, entity_share=0.25, error_share=0.0, seed=11)(examples)
    assert a["input_ids"].tolist() == b["input_ids"].tolist()
