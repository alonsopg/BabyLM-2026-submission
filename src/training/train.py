from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.load_dataset import load_text_dataset, tokenize_and_group
from src.masking.accuracy_morph_collator import AccuracyMorphCollator
from src.masking.cms_morph_collator import CMSMorphCollator
from src.masking.entity_accuracy_morph_collator import EntityAccuracyMorphCollator
from src.masking.heavy_hitter_collator import HeavyHitterMaskingCollator
from src.masking.frequency_masking import FrequencyWeightedMaskingCollator
from src.masking.random_collator import RandomMaskingCollator
from src.models.build_model import build_mlm_model, build_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_collator(tokenizer, cfg: dict, total_steps: int, seed: int):
    masking = cfg["masking"]
    if masking["type"] == "random":
        return RandomMaskingCollator(tokenizer, masking.get("mask_rate", 0.15), seed=seed)
    if masking["type"] in {"heavy_hitter", "heavy_hitter_randomized"}:
        warmup = int(total_steps * masking.get("error_warmup_fraction", 0.0))
        return HeavyHitterMaskingCollator(
            tokenizer=tokenizer,
            mask_rate=masking.get("mask_rate", 0.15),
            random_share=masking.get("random_share", 0.5),
            entity_share=masking.get("entity_share", 0.25),
            error_share=masking.get("error_share", 0.25),
            entity_sketch_capacity=masking.get("entity_sketch_capacity", 64),
            top_entities=masking.get("top_entities", 32),
            error_sketch_capacity=masking.get("error_sketch_capacity", 10000),
            error_warmup_steps=warmup,
            error_update_percentile=masking.get("error_update_percentile", 0.75),
            randomized_sketch=masking["type"] == "heavy_hitter_randomized",
            seed=seed,
        )
    if masking["type"] == "frequency_weighted":
        return FrequencyWeightedMaskingCollator(tokenizer, masking.get("mask_rate", 0.15), seed=seed)
    if masking["type"] == "cms_morph":
        warmup = int(total_steps * masking.get("adaptive_warmup_fraction", 0.0))
        return CMSMorphCollator(
            tokenizer=tokenizer,
            mask_rate=masking.get("mask_rate", 0.15),
            random_share=masking.get("random_share", 0.70),
            hard_token_share=masking.get("hard_token_share", 0.20),
            hard_morph_share=masking.get("hard_morph_share", 0.10),
            token_cms_width=masking.get("token_cms_width", 65536),
            token_cms_depth=masking.get("token_cms_depth", 4),
            morph_cms_width=masking.get("morph_cms_width", 65536),
            morph_cms_depth=masking.get("morph_cms_depth", 4),
            conservative_update=masking.get("conservative_update", True),
            adaptive_warmup_steps=warmup,
            update_percentile=masking.get("update_percentile", 0.75),
            score_cap=masking.get("score_cap", 5.0),
            char_ngram_n=masking.get("char_ngram_n", 3),
            add_boundaries=masking.get("add_boundaries", True),
            update_weight=masking.get("update_weight", 1.0),
            seed=seed,
        )
    if masking["type"] == "accuracy_morph":
        warmup = int(total_steps * masking.get("adaptive_warmup_fraction", 0.0))
        return AccuracyMorphCollator(
            tokenizer=tokenizer,
            mask_rate=masking.get("mask_rate", 0.15),
            random_share=masking.get("random_share", 0.80),
            hard_token_share=masking.get("hard_token_share", 0.15),
            hard_morph_share=masking.get("hard_morph_share", 0.05),
            token_cms_width=masking.get("token_cms_width", 65536),
            token_cms_depth=masking.get("token_cms_depth", 4),
            morph_cms_width=masking.get("morph_cms_width", 65536),
            morph_cms_depth=masking.get("morph_cms_depth", 4),
            conservative_update=masking.get("conservative_update", True),
            adaptive_warmup_steps=warmup,
            update_every_steps=masking.get("update_every_steps", 200),
            smoothing_alpha=masking.get("smoothing_alpha", 1.0),
            min_seen=masking.get("min_seen", 3.0),
            score_cap=masking.get("score_cap", 0.9),
            char_ngram_n=masking.get("char_ngram_n", 3),
            add_boundaries=masking.get("add_boundaries", True),
            seed=seed,
        )
    if masking["type"] == "entity_accuracy_morph":
        warmup = int(total_steps * masking.get("adaptive_warmup_fraction", 0.0))
        return EntityAccuracyMorphCollator(
            tokenizer=tokenizer,
            mask_rate=masking.get("mask_rate", 0.15),
            random_share=masking.get("random_share", 0.80),
            hard_token_share=masking.get("hard_token_share", 0.15),
            hard_morph_share=masking.get("hard_morph_share", 0.05),
            final_random_share=masking.get("final_random_share", 0.90),
            total_steps=total_steps,
            decay_start_fraction=masking.get("decay_start_fraction", 0.80),
            entity_bonus=masking.get("entity_bonus", 0.25),
            morph_tiebreak_bonus=masking.get("morph_tiebreak_bonus", 0.10),
            min_error_rate=masking.get("min_error_rate", 0.25),
            max_error_rate=masking.get("max_error_rate", 0.85),
            token_cms_width=masking.get("token_cms_width", 65536),
            token_cms_depth=masking.get("token_cms_depth", 4),
            morph_cms_width=masking.get("morph_cms_width", 65536),
            morph_cms_depth=masking.get("morph_cms_depth", 4),
            conservative_update=masking.get("conservative_update", True),
            adaptive_warmup_steps=warmup,
            update_every_steps=masking.get("update_every_steps", 200),
            smoothing_alpha=masking.get("smoothing_alpha", 1.0),
            min_seen=masking.get("min_seen", 3.0),
            score_cap=masking.get("score_cap", 0.9),
            char_ngram_n=masking.get("char_ngram_n", 3),
            add_boundaries=masking.get("add_boundaries", True),
            seed=seed,
        )
    raise ValueError(f"unknown masking type: {masking['type']}")


def evaluate(model, loader, device, max_batches: int):
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            metadata = batch.pop("mask_metadata", None)
            del metadata
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
            if step + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / max(1, len(losses))


def save_training_state(path: Path, model, optimizer, scheduler, step: int):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "model")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path / "training_state.pt",
    )


def save_collator_state(path: Path, collator):
    state = {"step": getattr(collator, "step", None)}
    if hasattr(collator, "error_sketch"):
        state["error_sketch"] = collator.error_sketch.to_dict()
    if hasattr(collator, "to_state_dict"):
        state.update(collator.to_state_dict())
    (path / "collator_state.json").write_text(json.dumps(state, indent=2))


def load_training_state(path: Path, model, optimizer, scheduler, collator, device):
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location=device, weights_only=False)
    bin_path = path / "model" / "pytorch_model.bin"
    if bin_path.exists():
        model.load_state_dict(torch.load(bin_path, map_location=device, weights_only=False))
    if (path / "model" / "model.safetensors").exists():
        from safetensors.torch import load_file

        model.load_state_dict(load_file(path / "model" / "model.safetensors", device=str(device)), strict=False)
        model.tie_weights()
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if isinstance(state.get("torch_rng"), torch.Tensor):
        torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None and all(isinstance(rng, torch.Tensor) for rng in state["cuda_rng"]):
        torch.cuda.set_rng_state_all([rng.detach().cpu().to(torch.uint8) for rng in state["cuda_rng"]])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    collator_state_path = path / "collator_state.json"
    if collator_state_path.exists():
        collator_state = json.loads(collator_state_path.read_text())
        if hasattr(collator, "step") and collator_state.get("step") is not None:
            collator.step = int(collator_state["step"])
        if hasattr(collator, "error_sketch") and "error_sketch" in collator_state:
            from src.masking.sketches import SpaceSavingSketch

            collator.error_sketch = SpaceSavingSketch.from_dict(collator_state["error_sketch"])
        if hasattr(collator, "load_state_dict"):
            collator.load_state_dict(collator_state)
    return int(state.get("step", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-no-dataloader-skip", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.name_suffix:
        cfg["name"] = f"{cfg['name']}_{args.name_suffix}"
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
        cfg["training"]["eval_every"] = min(cfg["training"].get("eval_every", args.max_steps), args.max_steps)
    if args.max_train_rows is not None:
        cfg["data"]["max_train_rows"] = args.max_train_rows
        cfg["data"]["validation_size"] = min(cfg["data"].get("validation_size", 2000), max(64, args.max_train_rows // 4))
    set_seed(args.seed)
    out_dir = Path(cfg.get("output_root", "outputs")) / cfg["name"] / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "metadata.json").write_text(json.dumps({"seed": args.seed, "git_hash": git_hash()}, indent=2))

    tokenizer = build_tokenizer(cfg["tokenizer"]["name"])
    train_raw, val_raw = load_text_dataset(
        cfg["data"]["dataset_name"],
        text_column=cfg["data"].get("text_column", "text"),
        validation_size=cfg["data"].get("validation_size", 2000),
        seed=args.seed,
    )
    if cfg["data"].get("max_train_rows"):
        train_raw = train_raw.select(range(min(cfg["data"]["max_train_rows"], len(train_raw))))
    train_ds, val_ds = tokenize_and_group(
        train_raw,
        val_raw,
        tokenizer,
        text_column=cfg["data"].get("text_column", "text"),
        max_length=cfg["training"].get("max_length", 128),
        num_proc=cfg["data"].get("num_proc", 2),
    )
    total_steps = cfg["training"]["max_steps"]
    collator = build_collator(tokenizer, cfg, total_steps=total_steps, seed=args.seed)
    eval_collator = RandomMaskingCollator(tokenizer, mask_rate=cfg["masking"].get("mask_rate", 0.15), seed=10_000 + args.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=eval_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlm_model(tokenizer, cfg["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["training"].get("warmup_fraction", 0.06)),
        num_training_steps=total_steps,
    )
    log_path = out_dir / "train_log.csv"
    mask_path = out_dir / "mask_diagnostics.csv"
    resume_dir = out_dir / "checkpoint-latest"
    start_step = 0
    if args.resume and resume_dir.exists():
        start_step = load_training_state(resume_dir, model, optimizer, scheduler, collator, device)
    start = time.time()
    model.train()
    log_mode = "a" if start_step and log_path.exists() else "w"
    mask_mode = "a" if start_step and mask_path.exists() else "w"
    with log_path.open(log_mode, newline="") as lf, mask_path.open(mask_mode, newline="") as mf:
        log_writer = csv.DictWriter(lf, fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_sec"])
        mask_writer = csv.DictWriter(mf, fieldnames=["step", "masked_tokens", "entity_masks", "error_masks", "random_masks"])
        if log_mode == "w":
            log_writer.writeheader()
        if mask_mode == "w":
            mask_writer.writeheader()
        data_iter = iter(train_loader)
        if start_step and not args.resume_no_dataloader_skip:
            for _ in range(start_step):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    next(data_iter)
        for step in range(start_step + 1, total_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            metadata = batch.pop("mask_metadata")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(collator, "update_from_predictions"):
                with torch.no_grad():
                    pred_ids = out.logits.detach().argmax(dim=-1).cpu()
                    collator.update_from_predictions(metadata, pred_ids, batch["labels"].detach().cpu())
            elif hasattr(collator, "update_error_sketch"):
                with torch.no_grad():
                    logits = out.logits.detach()
                    labels = batch["labels"]
                    per = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none")
                    per = per.view(labels.shape)
                    collator.update_error_sketch(metadata, per)
            totals = {"entity": 0, "error": 0, "random": 0}
            for counts in metadata.get("source_counts", []):
                for key in totals:
                    totals[key] += counts.get(key, 0)
            masked = int((batch["labels"] != -100).sum().detach().cpu())
            mask_writer.writerow({"step": step, "masked_tokens": masked, "entity_masks": totals["entity"], "error_masks": totals["error"], "random_masks": totals["random"]})
            if step == 1 or step % cfg["training"].get("eval_every", 100) == 0 or step == total_steps:
                val_loss = evaluate(model, val_loader, device, cfg["training"].get("eval_batches", 20))
                log_writer.writerow({"step": step, "train_loss": float(out.loss.detach().cpu()), "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "elapsed_sec": time.time() - start})
                lf.flush()
                mf.flush()
                save_training_state(resume_dir, model, optimizer, scheduler, step)
                save_collator_state(resume_dir, collator)
    model.save_pretrained(out_dir / "checkpoint-final")
    tokenizer.save_pretrained(out_dir / "checkpoint-final")
    if hasattr(collator, "error_sketch"):
        (out_dir / "error_sketch.json").write_text(json.dumps(collator.error_sketch.to_dict(), indent=2))
    if hasattr(collator, "to_state_dict"):
        (out_dir / "cms_morph_state.json").write_text(json.dumps(collator.to_state_dict(), indent=2))


if __name__ == "__main__":
    main()
