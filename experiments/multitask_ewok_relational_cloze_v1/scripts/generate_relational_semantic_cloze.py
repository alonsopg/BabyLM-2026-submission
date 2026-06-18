from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.build_model import build_tokenizer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def raw_items() -> list[dict]:
    items: list[dict] = []
    items.extend(
        [
            {"prompt": "The book is [MASK] the shelf.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The plate is [MASK] the table.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The cup is [MASK] the table.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The lamp is [MASK] the desk.", "good": "on", "bad": "under", "domain": "spatial_support"},
            {"prompt": "The picture is [MASK] the wall.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The shoes are [MASK] the floor.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The bird is flying [MASK] the tree.", "good": "above", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The plane is flying [MASK] the city.", "good": "above", "bad": "under", "domain": "spatial_support"},
            {"prompt": "The poster is [MASK] the door.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The clock is [MASK] the wall.", "good": "on", "bad": "under", "domain": "spatial_support"},
            {"prompt": "The rug is [MASK] the floor.", "good": "on", "bad": "inside", "domain": "spatial_support"},
            {"prompt": "The painting hangs [MASK] the sofa.", "good": "above", "bad": "inside", "domain": "spatial_support"},
        ]
    )
    items.extend(
        [
            {"prompt": "The fish is [MASK] the bowl.", "good": "in", "bad": "under", "domain": "containment"},
            {"prompt": "The water is [MASK] the glass.", "good": "in", "bad": "on", "domain": "containment"},
            {"prompt": "The apple is [MASK] the basket.", "good": "in", "bad": "under", "domain": "containment"},
            {"prompt": "The clothes are [MASK] the suitcase.", "good": "in", "bad": "above", "domain": "containment"},
            {"prompt": "The soup is [MASK] the bowl.", "good": "in", "bad": "behind", "domain": "containment"},
            {"prompt": "The letter is [MASK] the envelope.", "good": "in", "bad": "on", "domain": "containment"},
            {"prompt": "The money is [MASK] the wallet.", "good": "in", "bad": "beside", "domain": "containment"},
            {"prompt": "The keys are [MASK] the drawer.", "good": "in", "bad": "above", "domain": "containment"},
            {"prompt": "The milk is [MASK] the bottle.", "good": "in", "bad": "under", "domain": "containment"},
            {"prompt": "The pencil is [MASK] the box.", "good": "in", "bad": "behind", "domain": "containment"},
            {"prompt": "The toys are [MASK] the bag.", "good": "in", "bad": "above", "domain": "containment"},
            {"prompt": "The medicine is [MASK] the cabinet.", "good": "in", "bad": "under", "domain": "containment"},
        ]
    )
    items.extend(
        [
            {"prompt": "The glass broke because it was [MASK].", "good": "fragile", "bad": "hungry", "domain": "cause_effect"},
            {"prompt": "The ice melted because it became [MASK].", "good": "warm", "bad": "loud", "domain": "cause_effect"},
            {"prompt": "The plant died because it had no [MASK].", "good": "water", "bad": "shoes", "domain": "cause_effect"},
            {"prompt": "The child cried because he felt [MASK].", "good": "sad", "bad": "square", "domain": "cause_effect"},
            {"prompt": "The room became dark because the light was [MASK].", "good": "off", "bad": "sweet", "domain": "cause_effect"},
            {"prompt": "The ground became wet because it [MASK].", "good": "rained", "bad": "sang", "domain": "cause_effect"},
            {"prompt": "The fire went out because there was no [MASK].", "good": "oxygen", "bad": "music", "domain": "cause_effect"},
            {"prompt": "The bread burned because it was too [MASK].", "good": "hot", "bad": "young", "domain": "cause_effect"},
            {"prompt": "The towel dried because it became [MASK].", "good": "warm", "bad": "round", "domain": "cause_effect"},
            {"prompt": "The window cracked because the stone was [MASK].", "good": "hard", "bad": "sleepy", "domain": "cause_effect"},
            {"prompt": "The balloon popped because it was [MASK].", "good": "sharp", "bad": "polite", "domain": "cause_effect"},
            {"prompt": "The road became slippery because it was [MASK].", "good": "wet", "bad": "honest", "domain": "cause_effect"},
        ]
    )
    items.extend(
        [
            {"prompt": "The teacher explained the lesson to help the student [MASK].", "good": "learn", "bad": "melt", "domain": "social_agentive"},
            {"prompt": "The doctor treated the patient to make them [MASK].", "good": "healthy", "bad": "metal", "domain": "social_agentive"},
            {"prompt": "The person apologized because they made a [MASK].", "good": "mistake", "bad": "sandwich", "domain": "social_agentive"},
            {"prompt": "The child smiled because the gift made her [MASK].", "good": "happy", "bad": "square", "domain": "social_agentive"},
            {"prompt": "The coach trained the team to help them [MASK].", "good": "win", "bad": "evaporate", "domain": "social_agentive"},
            {"prompt": "The friend listened because she wanted to [MASK].", "good": "help", "bad": "freeze", "domain": "social_agentive"},
            {"prompt": "The waiter brought food because the customer was [MASK].", "good": "hungry", "bad": "wooden", "domain": "social_agentive"},
            {"prompt": "The driver stopped because the light was [MASK].", "good": "red", "bad": "salty", "domain": "social_agentive"},
            {"prompt": "The nurse checked the patient to keep them [MASK].", "good": "safe", "bad": "plastic", "domain": "social_agentive"},
            {"prompt": "The parent hugged the child to make him [MASK].", "good": "calm", "bad": "metal", "domain": "social_agentive"},
            {"prompt": "The chef cooked dinner because guests were [MASK].", "good": "hungry", "bad": "square", "domain": "social_agentive"},
            {"prompt": "The guard watched the door to keep people [MASK].", "good": "safe", "bad": "salty", "domain": "social_agentive"},
        ]
    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/multitask_ewok_relational_cloze_v1/data")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--target-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer)
    rng = random.Random(args.seed)
    rows: list[dict] = []
    skipped_multitoken: list[dict] = []
    skipped_ambiguous = 0
    seen = set()
    for item in raw_items():
        good_ids = tokenizer.encode(item["good"], add_special_tokens=False)
        bad_ids = tokenizer.encode(item["bad"], add_special_tokens=False)
        if len(good_ids) != 1 or len(bad_ids) != 1:
            skipped_multitoken.append({**item, "good_ids": good_ids, "bad_ids": bad_ids})
            continue
        if item["good"] == item["bad"]:
            skipped_ambiguous += 1
            continue
        if item["prompt"].count("[MASK]") != 1:
            raise ValueError(item["prompt"])
        key = (item["prompt"], item["good"], item["bad"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"task": "relational_semantic_cloze", "subtask": "relational_semantic_cloze", **item})

    base_rows = list(rows)
    prefixes = [
        "It is true that ",
        "People know that ",
        "Children learn that ",
        "A simple fact is that ",
        "In everyday life, ",
        "Usually, ",
        "As expected, ",
        "The usual fact is that ",
        "In normal situations, ",
        "Most people know that ",
        "A common fact is that ",
        "In ordinary life, ",
        "In the real world, ",
        "People often notice that ",
        "One can see that ",
        "In most cases, ",
        "Normally, ",
        "In a typical scene, ",
        "It makes sense that ",
        "Everyone knows that ",
        "A child can learn that ",
        "In common situations, ",
        "In simple examples, ",
        "It is usually true that ",
        "A familiar fact is that ",
    ]
    for prefix in prefixes:
        for row in base_rows:
            if len(rows) >= args.target_size:
                break
            prompt = prefix + row["prompt"][0].lower() + row["prompt"][1:]
            key = (prompt, row["good"], row["bad"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "prompt": prompt})
        if len(rows) >= args.target_size:
            break

    if len(rows) < args.target_size:
        raise SystemExit(f"Only generated {len(rows)} single-token examples; requested {args.target_size}")

    rng.shuffle(rows)
    rows = rows[: args.target_size]
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "relational_semantic_cloze.jsonl", rows)
    write_jsonl(out_dir / "relational_semantic_cloze_50_samples.jsonl", rows[:50])
    counts = Counter(row["domain"] for row in rows)
    manifest = {
        "task": "relational_semantic_cloze",
        "num_examples": len(rows),
        "num_skipped_multitoken": len(skipped_multitoken),
        "num_skipped_ambiguous": skipped_ambiguous,
        "uses_official_ewok_data": False,
        "uses_official_blimp_data": False,
        "uses_external_datasets": False,
        "domains": dict(sorted(counts.items())),
        "tokenizer": args.tokenizer,
        "target_size": args.target_size,
    }
    (out_dir / "relational_semantic_cloze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
