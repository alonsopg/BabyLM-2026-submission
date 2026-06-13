from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multitask_distributional_bert.scripts.generate_multitask_examples import (  # noqa: E402
    build_stats,
    gen_collocation,
    gen_connective,
    gen_definiteness,
    gen_mlm,
    gen_rtd,
    gen_substitution,
    render,
    words,
    write_jsonl,
)


TASKS = [
    "mlm",
    "rtd",
    "connective",
    "definiteness",
    "collocation",
    "function_word_recovery",
    "agreement_prediction",
    "substitution",
]

PROBABILITIES = {
    "mlm": 0.60,
    "rtd": 0.10,
    "connective": 0.075,
    "definiteness": 0.075,
    "collocation": 0.05,
    "function_word_recovery": 0.05,
    "agreement_prediction": 0.05,
    "substitution": 0.0,
}

FUNCTION_WORDS = {
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "into", "over", "under", "after", "before", "between",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "that", "which", "who", "whom", "whose", "where", "when", "whether", "if",
    "not", "never", "no",
}

AGREEMENT_WORDS = {"is", "are", "was", "were", "has", "have", "do", "does", "this", "these", "that", "those"}
AGREEMENT_CUES = {"i", "you", "he", "she", "it", "we", "they", "there", "who", "which", "that", "what", "where", "when"}
DEMONSTRATIVE_LEFT_CONTEXT = {
    "", "in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "into", "over", "under", "after", "before", "between",
    "see", "saw", "like", "want", "take", "took", "get", "got", "have", "has", "had", "use", "using", "make", "made",
}
IDIOM_SKIP = {
    ("to", "school"),
    ("at", "home"),
    ("in", "bed"),
}


def gen_function_word_recovery(texts: list[str], max_examples: int) -> list[dict]:
    rows = []
    for text in texts:
        toks = words(text)
        if len([tok for tok in toks if tok.isalpha()]) < 8:
            continue
        for idx, tok in enumerate(toks):
            lower = tok.lower()
            if lower not in FUNCTION_WORDS:
                continue
            if idx < 3 or len(toks) - idx - 1 < 3:
                continue
            next_alpha = next((t.lower() for t in toks[idx + 1 :] if t.isalpha()), "")
            if (lower, next_alpha) in IDIOM_SKIP:
                continue
            masked = toks[:]
            masked[idx] = "[MASK]"
            rows.append({
                "task": "function_word_recovery",
                "input": render(masked),
                "target": {"masked_positions": [idx], "tokens": [lower]},
                "metadata": {"original": text, "function_word": lower, "loss": "mlm"},
            })
            break
        if len(rows) >= max_examples:
            break
    return rows


def pluralish(word: str) -> bool:
    lower = word.lower()
    return lower.endswith("s") and not lower.endswith(("ss", "us", "is"))


def agreement_context_ok(toks: list[str], idx: int) -> bool:
    if idx < 2 or len(toks) - idx - 1 < 2:
        return False
    before = [tok.lower() for tok in toks[max(0, idx - 4) : idx] if tok.isalpha()]
    after = [tok.lower() for tok in toks[idx + 1 : idx + 4] if tok.isalpha()]
    if not before or not after:
        return False
    return True


def previous_alpha(toks: list[str], idx: int) -> str:
    for tok in reversed(toks[:idx]):
        if tok.isalpha():
            return tok.lower()
    return ""


def has_agreement_cue(toks: list[str], idx: int) -> bool:
    before = [tok.lower() for tok in toks[max(0, idx - 4) : idx] if tok.isalpha()]
    return any(tok in AGREEMENT_CUES or pluralish(tok) for tok in before)


def gen_agreement_prediction(texts: list[str], max_examples: int) -> list[dict]:
    rows = []
    for text in texts:
        toks = words(text)
        for idx, tok in enumerate(toks):
            lower = tok.lower()
            if lower not in AGREEMENT_WORDS or not agreement_context_ok(toks, idx):
                continue
            if lower in {"this", "these", "that", "those"}:
                next_word = next((t for t in toks[idx + 1 :] if t.isalpha()), "")
                if not next_word:
                    continue
                is_plural = pluralish(next_word)
                if lower in {"this", "that"} and is_plural:
                    continue
                if lower in {"these", "those"} and not is_plural:
                    continue
                if previous_alpha(toks, idx) not in DEMONSTRATIVE_LEFT_CONTEXT:
                    continue
            elif not has_agreement_cue(toks, idx):
                continue
            masked = toks[:]
            masked[idx] = "[MASK]"
            rows.append({
                "task": "agreement_prediction",
                "input": render(masked),
                "target": {"masked_positions": [idx], "tokens": [lower]},
                "metadata": {"original": text, "agreement_token": lower, "loss": "mlm"},
            })
            break
        if len(rows) >= max_examples:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BabyLM-community/BabyLM-2026-Strict-Small")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="experiments/multitask_repair_syntax_v1/data/generated")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--max-examples-per-task", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split="train")
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    texts = [row[args.text_column].strip() for row in ds if row.get(args.text_column) and row[args.text_column].strip()]
    rng.shuffle(texts)

    counts, candidates, bigrams, _ = build_stats(texts)
    rows_by_task = {
        "mlm": gen_mlm(texts, rng, args.max_examples_per_task),
        "rtd": gen_rtd(texts, counts, candidates, rng, args.max_examples_per_task),
        "connective": gen_connective(texts, args.max_examples_per_task),
        "definiteness": gen_definiteness(texts, args.max_examples_per_task),
        "collocation": gen_collocation(counts, bigrams, candidates, rng, args.max_examples_per_task),
        "function_word_recovery": gen_function_word_recovery(texts, args.max_examples_per_task),
        "agreement_prediction": gen_agreement_prediction(texts, args.max_examples_per_task),
        "substitution": gen_substitution(texts, counts, candidates, rng, args.max_examples_per_task),
    }

    stats = {}
    manifest = {"dataset": args.dataset, "text_column": args.text_column, "max_rows": args.max_rows, "seed": args.seed, "tasks": {}}
    for task in TASKS:
        rows = rows_by_task[task]
        write_jsonl(out_dir / f"{task}.jsonl", rows)
        write_jsonl(out_dir / "samples" / f"{task}_50_examples.jsonl", rows[:50])
        stats[task] = {
            "probability": PROBABILITIES[task],
            "num_examples": len(rows),
            "enabled": PROBABILITIES[task] > 0.0 and len(rows) > 0,
        }
        if task == "substitution":
            stats[task]["disabled_reason"] = "disabled for syntax repair run because pilot inspection found noisy heuristic examples"
        manifest["tasks"][task] = {
            "probability": PROBABILITIES[task],
            "path": str(out_dir / f"{task}.jsonl"),
            "sample_path": str(out_dir / "samples" / f"{task}_50_examples.jsonl"),
        }
    (out_dir / "task_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "multitask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
