from __future__ import annotations

from transformers import AutoTokenizer, BertConfig, BertForMaskedLM


def build_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def build_mlm_model(tokenizer, cfg: dict):
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg.get("hidden_size", 256),
        num_hidden_layers=cfg.get("num_hidden_layers", 4),
        num_attention_heads=cfg.get("num_attention_heads", 4),
        intermediate_size=cfg.get("intermediate_size", 1024),
        max_position_embeddings=cfg.get("max_position_embeddings", 512),
        type_vocab_size=1,
    )
    return BertForMaskedLM(config)
