from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_dataset import load_text_dataset, tokenize_and_group
from src.masking.random_collator import RandomMaskingCollator
from src.models.build_model import build_mlm_model, build_tokenizer
from src.evaluation.mlm import evaluate_mlm_loss


CONNECTIVES = ["because", "so", "but", "although", "however", "therefore", "when", "while", "if", "then", "before", "after", "since", "though", "unless"]
TASKS = ["mlm", "rtd", "connective", "definiteness", "collocation", "grammar_minpair", "semantic_cloze_ranking"]
COLLOCATION_HEAD_TASKS = {"collocation", "grammar_minpair"}


class JsonlTaskDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"empty task dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx % len(self.rows)]


class MultiTaskBert(nn.Module):
    def __init__(self, tokenizer, model_cfg: dict):
        super().__init__()
        self.mlm = build_mlm_model(tokenizer, model_cfg)
        hidden = self.mlm.config.hidden_size
        self.rtd_head = nn.Linear(hidden, 2)
        self.connective_head = nn.Linear(hidden, len(CONNECTIVES))
        self.definiteness_head = nn.Linear(hidden, 2)
        self.collocation_head = nn.Linear(hidden, 2)

    @property
    def bert(self):
        return self.mlm.bert

    def save_mlm_compatible(self, path: Path, tokenizer) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.mlm.save_pretrained(path)
        tokenizer.save_pretrained(path)

    def save_full(self, path: Path, tokenizer) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        self.mlm.config.save_pretrained(path)
        tokenizer.save_pretrained(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strict_small_revision_names() -> list[str]:
    return [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 101, 10)]


def build_revision_schedule(cfg: dict, total_steps: int) -> tuple[str, dict[int, str]]:
    revision_cfg = cfg["training"].get("checkpoint_revisions", {})
    if not revision_cfg.get("enabled", False):
        return "disabled", {}
    names = revision_cfg.get("names") or strict_small_revision_names()
    schedule_by = revision_cfg.get("schedule_by", "token_millions")
    schedule_max_step = int(revision_cfg.get("schedule_max_step", total_steps))
    schedule: dict[int, str] = {}
    for name in names:
        if not (name.startswith("chck_") and name.endswith("M")):
            raise ValueError(f"Unsupported checkpoint revision name: {name}")
        millions = int(name.removeprefix("chck_").removesuffix("M"))
        if schedule_by == "token_millions":
            schedule[millions * 1_000_000] = name
        elif schedule_by == "step_fraction":
            target_step = max(1, round((millions / 100) * schedule_max_step))
            schedule[target_step] = name
        else:
            raise ValueError(f"Unsupported checkpoint revision schedule_by: {schedule_by}")
    return schedule_by, dict(sorted(schedule.items()))


def mask_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, rng: random.Random, mask_rate: float = 0.15) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    specials = set(tokenizer.all_special_ids)
    for row in range(input_ids.size(0)):
        valid = [i for i, tok in enumerate(input_ids[row].tolist()) if attention_mask[row, i].item() and tok not in specials]
        n_mask = max(1, round(len(valid) * mask_rate)) if valid else 0
        selected = set(rng.sample(valid, min(n_mask, len(valid))))
        for pos in selected:
            labels[row, pos] = input_ids[row, pos]
            r = rng.random()
            if r < 0.8:
                input_ids[row, pos] = tokenizer.mask_token_id
            elif r < 0.9:
                input_ids[row, pos] = rng.randrange(len(tokenizer))
    return labels


def collate_task(rows: list[dict], task: str, tokenizer, rng: random.Random, max_length: int, max_scoring_positions: int = 6) -> dict[str, torch.Tensor]:
    del max_scoring_positions
    if task == "semantic_cloze_ranking":
        enc = tokenizer([row["prompt"] for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        mask_positions = enc["input_ids"].eq(tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.size(0) != len(rows) or not torch.equal(mask_positions[:, 0], torch.arange(len(rows))):
            raise ValueError("semantic_cloze_ranking rows must contain exactly one [MASK]")
        enc["mask_positions"] = mask_positions[:, 1]
        enc["good_token_ids"] = torch.tensor([tokenizer.encode(row["good"], add_special_tokens=False)[0] for row in rows], dtype=torch.long)
        enc["bad_token_ids"] = torch.tensor([tokenizer.encode(row["bad"], add_special_tokens=False)[0] for row in rows], dtype=torch.long)
        return enc
    enc = tokenizer([row["input"] for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    if task == "mlm":
        labels = mask_batch(enc["input_ids"], enc["attention_mask"], tokenizer, rng)
        enc["labels"] = labels
        return enc
    if task == "rtd":
        labels = torch.full_like(enc["input_ids"], -100)
        specials = set(tokenizer.all_special_ids)
        for row_i in range(enc["input_ids"].size(0)):
            for pos, token_id in enumerate(enc["input_ids"][row_i].tolist()):
                if enc["attention_mask"][row_i, pos].item() and token_id not in specials:
                    labels[row_i, pos] = 0
        for row_i, row in enumerate(rows):
            toks = row["input"].split()
            word_labels = row["target"]["token_labels"]
            for word_i, word in enumerate(toks[: len(word_labels)]):
                tokenized = tokenizer.tokenize(word)
                if word_labels[word_i] == 1:
                    # Mark all matching subword ids for this surface word when found.
                    ids = tokenizer.convert_tokens_to_ids(tokenized)
                    for j in range(1, enc["input_ids"].size(1)):
                        if enc["input_ids"][row_i, j : j + len(ids)].tolist() == ids:
                            labels[row_i, j : j + len(ids)] = 1
                            break
        enc["labels"] = labels
        return enc
    if task == "connective":
        enc["labels"] = torch.tensor([CONNECTIVES.index(row["target"]) for row in rows], dtype=torch.long)
        return enc
    if task == "definiteness":
        label = 1 if rows[0]["target"][0]["label"] == "definite" else 0
        enc["labels"] = torch.tensor([1 if row["target"][0]["label"] == "definite" else 0 for row in rows], dtype=torch.long)
        return enc
    if task in COLLOCATION_HEAD_TASKS:
        enc["labels"] = torch.tensor([int(row["target"]) for row in rows], dtype=torch.long)
        return enc
    raise ValueError(task)


def pooled_at_marker(hidden: torch.Tensor, input_ids: torch.Tensor, marker_id: int | None) -> torch.Tensor:
    if marker_id is None:
        return hidden[:, 0]
    mask = input_ids.eq(marker_id)
    if not mask.any(dim=1).all():
        return hidden[:, 0]
    idx = mask.float().argmax(dim=1)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def forward_task(model: MultiTaskBert, batch: dict[str, torch.Tensor], task: str, tokenizer) -> tuple[torch.Tensor, dict[str, float]]:
    if task == "semantic_cloze_ranking":
        out = model.mlm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        row_idx = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)
        logits = out.logits[row_idx, batch["mask_positions"], :]
        log_probs = F.log_softmax(logits, dim=-1)
        score_good = log_probs[row_idx, batch["good_token_ids"]]
        score_bad = log_probs[row_idx, batch["bad_token_ids"]]
        margin = score_good - score_bad
        loss = F.softplus(-margin).mean()
        metrics = {
            "semantic_cloze_ranking_margin": float(margin.detach().mean().cpu()),
            "semantic_cloze_ranking_accuracy": float((margin.detach() > 0).float().mean().cpu()),
        }
        return loss, metrics
    labels = batch.pop("labels")
    if task == "mlm":
        return model.mlm(**batch, labels=labels).loss, {}
    out = model.bert(**batch)
    hidden = out.last_hidden_state
    if task == "rtd":
        logits = model.rtd_head(hidden)
        return nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1), ignore_index=-100), {}
    if task == "connective":
        pooled = pooled_at_marker(hidden, batch["input_ids"], tokenizer.convert_tokens_to_ids("[unused1]"))
        return nn.functional.cross_entropy(model.connective_head(pooled), labels), {}
    if task == "definiteness":
        pooled = pooled_at_marker(hidden, batch["input_ids"], tokenizer.convert_tokens_to_ids("[unused2]"))
        return nn.functional.cross_entropy(model.definiteness_head(pooled), labels), {}
    pooled = hidden[:, 0]
    if task in COLLOCATION_HEAD_TASKS:
        return nn.functional.cross_entropy(model.collocation_head(pooled), labels), {}
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/multitask_distributional_bert/configs/multitask_bert.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.name_suffix:
        cfg["name"] = f"{cfg['name']}_{args.name_suffix}"
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
        cfg["training"]["eval_every"] = min(cfg["training"].get("eval_every", args.max_steps), args.max_steps)
        if args.max_steps <= int(cfg["training"].get("calibration_mlm_steps", 0)):
            cfg["training"]["calibration_mlm_steps"] = 0
    set_seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(cfg["output_root"]) / cfg["name"] / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    tokenizer = build_tokenizer(cfg["tokenizer"]["name"])
    model = MultiTaskBert(tokenizer, cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is not available. Refusing to train multi-task model on CPU.")
    model.to(device)

    _, val_raw = load_text_dataset(
        cfg["data"]["dataset_name"],
        text_column=cfg["data"].get("text_column", "text"),
        validation_size=cfg["data"].get("validation_size", 2000),
        seed=args.seed,
    )
    val_ds, _ = tokenize_and_group(
        val_raw,
        val_raw,
        tokenizer,
        text_column=cfg["data"].get("text_column", "text"),
        max_length=cfg["training"].get("max_length", 128),
        num_proc=cfg["data"].get("num_proc", 2),
    )
    eval_collator = RandomMaskingCollator(tokenizer, mask_rate=0.15, seed=10_000 + args.seed)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=eval_collator)

    active = [(task, spec) for task, spec in cfg["tasks"].items() if spec.get("probability", 0) > 0 and Path(spec["path"]).exists() and Path(spec["path"]).stat().st_size > 0]
    if not active:
        raise SystemExit("No non-empty generated task files found. Run generate_multitask_examples.py first.")
    datasets = {task: JsonlTaskDataset(spec["path"]) for task, spec in active}
    loaders = {
        task: iter(DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=lambda rows, t=task: collate_task(rows, t, tokenizer, rng, cfg["training"]["max_length"], cfg["training"].get("max_scoring_positions", 6))))
        for task, ds in datasets.items()
    }
    probs = np.array([spec.get("probability", 0.0) for _, spec in active], dtype=np.float64)
    probs = probs / probs.sum()
    task_names = [task for task, _ in active]

    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
    total_steps = int(cfg["training"]["max_steps"])
    calibration_mlm_steps = int(cfg["training"].get("calibration_mlm_steps", 0))
    calibration_start = total_steps - calibration_mlm_steps + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["training"].get("warmup_fraction", 0.06)), total_steps)
    revision_schedule_by, revision_schedule = build_revision_schedule(cfg, total_steps)
    revision_output_dir = out_dir / cfg["training"].get("checkpoint_revisions", {}).get("output_dir", "aoa_revisions")
    saved_revisions: list[dict[str, int]] = []
    next_revision_idx = 0
    revision_targets = list(revision_schedule.items())
    seen_tokens = 0

    log_path = out_dir / "train_log.csv"
    counts = {task: 0 for task in TASKS}
    early_cfg = cfg["training"].get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience_evals", 4))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_start = int(early_cfg.get("start_after_step", 0))
    best_val_mlm_loss = float("inf")
    bad_evals = 0
    start = time.time()
    last_semantic_metrics: dict[str, float] = {}
    model.train()
    with log_path.open("w", newline="") as f:
        fields = ["step", "task", "loss", "semantic_cloze_ranking_margin", "semantic_cloze_ranking_accuracy", "seen_tokens", "val_mlm_loss", "best_val_mlm_loss", "bad_evals", "lr", "elapsed_sec", "early_stop"] + [f"steps/{task}" for task in TASKS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, total_steps + 1):
            if calibration_mlm_steps > 0 and step >= calibration_start:
                task = "mlm"
            else:
                task = rng.choices(task_names, weights=probs.tolist(), k=1)[0]
            try:
                batch = next(loaders[task])
            except StopIteration:
                loaders[task] = iter(DataLoader(datasets[task], batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=lambda rows, t=task: collate_task(rows, t, tokenizer, rng, cfg["training"]["max_length"], cfg["training"].get("max_scoring_positions", 6))))
                batch = next(loaders[task])
            batch = {k: v.to(device) for k, v in batch.items()}
            seen_tokens += int(batch["attention_mask"].sum().item())
            loss, task_metrics = forward_task(model, batch, task, tokenizer)
            if task_metrics:
                last_semantic_metrics.update(task_metrics)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            counts[task] += 1
            while next_revision_idx < len(revision_targets) and (
                (revision_schedule_by == "token_millions" and seen_tokens >= revision_targets[next_revision_idx][0])
                or (revision_schedule_by == "step_fraction" and step >= revision_targets[next_revision_idx][0])
            ):
                target, revision_name = revision_targets[next_revision_idx]
                revision_path = revision_output_dir / revision_name
                model.save_mlm_compatible(revision_path, tokenizer)
                revision_record = {
                    "revision": revision_name,
                    "saved_at_step": step,
                    "seen_tokens": seen_tokens,
                    "schedule_by": revision_schedule_by,
                }
                if revision_schedule_by == "token_millions":
                    revision_record["target_tokens"] = target
                else:
                    revision_record["target_step"] = target
                saved_revisions.append(revision_record)
                (revision_output_dir / "manifest.json").write_text(json.dumps(saved_revisions, indent=2))
                next_revision_idx += 1
            if step == 1 or step % cfg["training"].get("eval_every", 500) == 0 or step == total_steps:
                val_mlm_loss = evaluate_mlm_loss(model.mlm, val_loader, device, cfg["training"].get("eval_batches", 20))
                improved = val_mlm_loss < (best_val_mlm_loss - early_min_delta)
                if improved:
                    best_val_mlm_loss = val_mlm_loss
                    bad_evals = 0
                    best = out_dir / "checkpoint-best"
                    model.save_full(best / "full_multitask_checkpoint", tokenizer)
                    model.save_mlm_compatible(best / "mlm_compatible_checkpoint", tokenizer)
                elif step >= early_start:
                    bad_evals += 1
                stopped_early = early_enabled and step >= early_start and bad_evals >= early_patience and step < total_steps
                row = {
                    "step": step,
                    "task": task,
                    "loss": float(loss.detach().cpu()),
                    "semantic_cloze_ranking_margin": last_semantic_metrics.get("semantic_cloze_ranking_margin", ""),
                    "semantic_cloze_ranking_accuracy": last_semantic_metrics.get("semantic_cloze_ranking_accuracy", ""),
                    "seen_tokens": seen_tokens,
                    "val_mlm_loss": val_mlm_loss,
                    "best_val_mlm_loss": best_val_mlm_loss,
                    "bad_evals": bad_evals,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": time.time() - start,
                    "early_stop": stopped_early,
                }
                row.update({f"steps/{t}": counts[t] for t in TASKS})
                writer.writerow(row)
                f.flush()
                latest = out_dir / "checkpoint-latest"
                model.save_full(latest / "full_multitask_checkpoint", tokenizer)
                model.save_mlm_compatible(latest / "mlm_compatible_checkpoint", tokenizer)
                if stopped_early:
                    break

    model.save_full(out_dir / "full_multitask_checkpoint", tokenizer)
    model.save_mlm_compatible(out_dir / "mlm_compatible_checkpoint", tokenizer)
    if revision_schedule:
        revision_output_dir.mkdir(parents=True, exist_ok=True)
        (revision_output_dir / "manifest.json").write_text(json.dumps(saved_revisions, indent=2))
    (out_dir / "task_counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
