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

    affordances = [
        ("knife", "cutting", "drinking"), ("pencil", "writing", "sleeping"), ("broom", "sweeping", "reading"),
        ("cup", "drinking", "cutting"), ("spoon", "eating", "driving"), ("key", "opening", "swimming"),
        ("bed", "sleeping", "cooking"), ("chair", "sitting", "flying"), ("hammer", "hitting", "singing"),
        ("brush", "painting", "running"), ("shovel", "digging", "reading"), ("camera", "filming", "eating"),
        ("ruler", "measuring", "sleeping"), ("soap", "washing", "flying"), ("needle", "sewing", "drinking"),
    ]
    for obj, good, bad in affordances:
        noun = obj if obj == "soap" else f"a {obj}"
        items.extend(
            [
                {"prompt": f"{noun.capitalize()} is used for [MASK].", "good": good, "bad": bad, "domain": "affordance"},
                {"prompt": f"People use {noun} for [MASK].", "good": good, "bad": bad, "domain": "affordance"},
                {"prompt": f"The purpose of {noun} is [MASK].", "good": good, "bad": bad, "domain": "affordance"},
            ]
        )

    locations = [
        ("a fish", "water", "cupboard", "lives in"), ("a bird", "nest", "bottle", "lives in a"),
        ("a book", "shelf", "river", "is kept on a"), ("milk", "fridge", "shoe", "is kept in a"),
        ("a car", "road", "pillow", "is driven on a"), ("a boat", "water", "grass", "moves on"),
        ("a shirt", "closet", "lake", "is kept in a"), ("a coin", "pocket", "cloud", "is kept in a"),
        ("an apple", "basket", "garage", "is kept in a"), ("a plane", "sky", "drawer", "flies in the"),
        ("a train", "track", "blanket", "moves on a"), ("a lamp", "desk", "ocean", "stands on a"),
    ]
    for subject, good, bad, phrase in locations:
        items.extend(
            [
                {"prompt": f"{subject.capitalize()} {phrase} [MASK].", "good": good, "bad": bad, "domain": "typical_location"},
                {"prompt": f"The usual place for {subject} is [MASK].", "good": good, "bad": bad, "domain": "typical_location"},
            ]
        )

    properties = [
        ("Fire is usually", "Fire can be", "hot", "cold"),
        ("Ice is usually", "Ice can be", "cold", "hot"),
        ("Glass is often", "Glass can be", "fragile", "hungry"),
        ("Stone is usually", "Stone can be", "hard", "soft"),
        ("Sugar tastes", "Sugar can taste", "sweet", "salty"),
        ("Lemons taste", "Lemons can taste", "sour", "sweet"),
        ("Feathers are usually", "Feathers can be", "light", "heavy"),
        ("Bricks are usually", "Bricks can be", "heavy", "thirsty"),
        ("Honey is usually", "Honey can be", "sticky", "angry"),
        ("Knives are usually", "Knives can be", "sharp", "sleepy"),
        ("Clouds are often", "Clouds can be", "soft", "hungry"),
        ("Snow is usually", "Snow can be", "cold", "hot"),
    ]
    for prompt_a, prompt_b, good, bad in properties:
        items.extend(
            [
                {"prompt": f"{prompt_a} [MASK].", "good": good, "bad": bad, "domain": "physical_property"},
                {"prompt": f"{prompt_b} [MASK].", "good": good, "bad": bad, "domain": "physical_property"},
            ]
        )

    part_wholes = [
        ("car", "wheels", "pages"), ("book", "pages", "wheels"), ("tree", "leaves", "buttons"),
        ("bird", "wings", "wheels"), ("house", "roof", "tail"), ("shirt", "sleeves", "roots"),
        ("camera", "lens", "leaf"), ("flower", "petals", "keys"), ("door", "handle", "wing"),
        ("guitar", "strings", "wheels"), ("clock", "hands", "roots"), ("computer", "keyboard", "tail"),
    ]
    for whole, good, bad in part_wholes:
        article = "an" if whole[0] in "aeiou" else "a"
        items.extend(
            [
                {"prompt": f"A {whole} has [MASK].", "good": good, "bad": bad, "domain": "part_whole"},
                {"prompt": f"Part of {article} {whole} can be [MASK].", "good": good, "bad": bad, "domain": "part_whole"},
            ]
        )

    agency = [
        ("person", "think", "rust"), ("dog", "bark", "read"), ("baby", "cry", "drive"),
        ("teacher", "explain", "melt"), ("cat", "meow", "write"), ("robot", "move", "evaporate"),
        ("doctor", "help", "boil"), ("singer", "sing", "freeze"), ("driver", "steer", "bloom"),
        ("child", "learn", "rust"), ("chef", "cook", "melt"), ("bird", "fly", "read"),
    ]
    for actor, good, bad in agency:
        items.extend(
            [
                {"prompt": f"A {actor} can [MASK].", "good": good, "bad": bad, "domain": "animate_agency"},
                {"prompt": f"The {actor} is able to [MASK].", "good": good, "bad": bad, "domain": "animate_agency"},
            ]
        )

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/multitask_ewok_semantic_cloze_v1/data")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.tokenizer)
    rng = random.Random(args.seed)
    rows = []
    skipped = []
    seen = set()
    for item in raw_items():
        good_ids = tokenizer.encode(item["good"], add_special_tokens=False)
        bad_ids = tokenizer.encode(item["bad"], add_special_tokens=False)
        if len(good_ids) != 1 or len(bad_ids) != 1:
            skipped.append({**item, "good_ids": good_ids, "bad_ids": bad_ids})
            continue
        if item["prompt"].count("[MASK]") != 1:
            raise ValueError(item["prompt"])
        key = (item["prompt"], item["good"], item["bad"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"task": "semantic_cloze_ranking", **item})

    base_rows = list(rows)
    variants = ["It is true that ", "People know that ", "Children learn that ", "A simple fact is that ", "The usual fact is that ", "In everyday life, ", "As expected, "]
    for prefix in variants:
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
        raise SystemExit(f"Only generated {len(rows)} single-token cloze examples; requested {args.target_size}")

    rng.shuffle(rows)
    rows = rows[: args.target_size]
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "semantic_cloze_ranking.jsonl", rows)
    write_jsonl(out_dir / "semantic_cloze_50_samples.jsonl", rows[:50])
    counts = Counter(row["domain"] for row in rows)
    manifest = {
        "task": "semantic_cloze_ranking",
        "num_examples": len(rows),
        "num_skipped_multitoken": len(skipped),
        "domains": dict(sorted(counts.items())),
        "uses_official_ewok_data": False,
        "uses_official_blimp_data": False,
        "tokenizer": args.tokenizer,
        "target_size": args.target_size,
    }
    (out_dir / "semantic_cloze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
