from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multitask_distributional_bert.scripts.train_multitask_bert import CONTENT_STOPWORDS
from src.models.build_model import build_tokenizer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scoring_tokens(tokenizer, target: str, context: str, max_length: int, max_scoring_positions: int) -> list[str]:
    prefix = f"Target: {target} Context:"
    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(context_ids)
    candidates = []
    fallback = []
    for offset, token in enumerate(tokens):
        pos = 1 + prefix_len + offset
        if pos >= max_length - 1:
            break
        clean = token[2:] if token.startswith("##") else token
        if clean.isalpha():
            fallback.append(token)
            if clean.lower() not in CONTENT_STOPWORDS:
                candidates.append(token)
    return (candidates[:max_scoring_positions] or fallback[:max_scoring_positions])


def convert_row(row: dict, tokenizer, max_length: int, max_scoring_positions: int) -> dict:
    metadata = row["metadata"]
    target = metadata["target_sentence"]
    good = metadata["good_context"]
    bad = metadata["bad_context"]
    good_sequence = f"Target: {target} Context: {good}"
    bad_sequence = f"Target: {target} Context: {bad}"
    good_tokens = scoring_tokens(tokenizer, target, good, max_length, max_scoring_positions)
    bad_tokens = scoring_tokens(tokenizer, target, bad, max_length, max_scoring_positions)
    return {
        "task": "mlm_pair_ranking",
        "good_sequence": good_sequence,
        "bad_sequence": bad_sequence,
        "metadata": {
            "domain": metadata.get("domain", "unknown"),
            "relation": metadata.get("relation", "unknown"),
            "target_sentence": target,
            "good_context": good,
            "bad_context": bad,
            "scoring_tokens_good": good_tokens,
            "scoring_tokens_bad": bad_tokens,
            "source_task": row.get("task", "conceptual_plausibility_choice"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="experiments/multitask_ewok_plausibility_minimal/data/generated/conceptual_plausibility_choice.jsonl")
    parser.add_argument("--output-dir", default="experiments/multitask_mlm_pair_ranking_minimal/data/generated")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-scoring-positions", type=int, default=6)
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer)
    source = Path(args.source)
    out_dir = Path(args.output_dir)
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(convert_row(json.loads(line), tokenizer, args.max_length, args.max_scoring_positions))

    empty = [row for row in rows if not row["metadata"]["scoring_tokens_good"] or not row["metadata"]["scoring_tokens_bad"]]
    if empty:
        raise SystemExit(f"{len(empty)} rows have empty scoring-token lists")

    write_jsonl(out_dir / "mlm_pair_ranking.jsonl", rows)
    write_jsonl(out_dir / "samples" / "mlm_pair_ranking_50_examples.jsonl", rows[:50])

    counts = Counter(row["metadata"]["domain"] for row in rows)
    manifest = {
        "task": "mlm_pair_ranking",
        "source": str(source),
        "output": str(out_dir / "mlm_pair_ranking.jsonl"),
        "sample_path": str(out_dir / "samples" / "mlm_pair_ranking_50_examples.jsonl"),
        "num_examples": len(rows),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "max_scoring_positions": args.max_scoring_positions,
        "domains": dict(sorted(counts.items())),
        "ewok_data_used": False,
        "blimp_data_used": False,
    }
    (out_dir / "mlm_pair_ranking_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
