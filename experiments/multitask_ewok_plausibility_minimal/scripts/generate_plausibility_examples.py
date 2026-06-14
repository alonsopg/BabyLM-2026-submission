from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multitask_distributional_bert.scripts.generate_multitask_examples import (
    build_stats,
    gen_collocation,
    gen_connective,
    gen_definiteness,
    gen_mlm,
    gen_rtd,
    gen_substitution,
    write_jsonl,
)


TASKS = [
    "mlm",
    "rtd",
    "connective",
    "definiteness",
    "collocation",
    "conceptual_plausibility_choice",
    "substitution",
    "function_word_recovery",
    "agreement_prediction",
    "grammar_minpair",
]

TASK_PROBABILITIES = {
    "mlm": 0.55,
    "rtd": 0.10,
    "connective": 0.075,
    "definiteness": 0.075,
    "collocation": 0.05,
    "conceptual_plausibility_choice": 0.15,
    "substitution": 0.0,
    "function_word_recovery": 0.0,
    "agreement_prediction": 0.0,
    "grammar_minpair": 0.0,
}


def indef(phrase: str) -> str:
    lower = phrase.lower()
    article = "a" if lower.startswith(("useful", "university", "unit")) else "an" if lower[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {phrase}"


def add_choice(rows: list[dict], rng: random.Random, target: str, good: str, bad: str, domain: str, relation: str) -> None:
    if rng.random() < 0.5:
        context_a, context_b, label = good, bad, 0
    else:
        context_a, context_b, label = bad, good, 1
    rows.append(
        {
            "task": "conceptual_plausibility_choice",
            "input": f"[CLS] Target: {target} [SEP] Context A: {context_a} [SEP] Context B: {context_b} [SEP]",
            "target": label,
            "metadata": {
                "domain": domain,
                "relation": relation,
                "target_sentence": target,
                "good_context": good,
                "bad_context": bad,
                "label_meaning": "0 = Context A is more plausible, 1 = Context B is more plausible",
            },
        }
    )


def gen_selectional_preference(rng: random.Random, limit: int) -> list[dict]:
    frames = [
        ("teacher", "explained", "lesson", "to the student", "gave information about"),
        ("doctor", "treated", "patient", "in the clinic", "provided care for"),
        ("chef", "cooked", "meal", "for the guest", "prepared"),
        ("parent", "comforted", "child", "at home", "helped calm"),
        ("coach", "trained", "athlete", "on the field", "gave practice to"),
        ("librarian", "recommended", "book", "to the reader", "suggested reading for"),
        ("mechanic", "repaired", "car", "in the garage", "fixed"),
        ("gardener", "watered", "plant", "in the garden", "gave water to"),
        ("musician", "played", "song", "for the audience", "performed"),
        ("pilot", "flew", "plane", "from the airport", "controlled"),
        ("artist", "painted", "portrait", "in the studio", "created"),
        ("judge", "heard", "case", "in the court", "listened to"),
        ("writer", "wrote", "story", "for the editor", "created text for"),
        ("farmer", "fed", "animal", "in the barn", "gave food to"),
        ("scientist", "tested", "sample", "in the lab", "examined"),
    ]
    rows: list[dict] = []
    templates: list[tuple[str, str, str]] = []
    agent_mods = ["", "careful", "young", "experienced", "local", "helpful", "busy", "patient", "skilled", "friendly"]
    patient_mods = ["", "small", "new", "old", "important", "quiet", "large", "familiar", "difficult", "simple"]
    adverbs = ["", "carefully", "quickly", "slowly", "politely", "clearly", "patiently", "often"]
    for agent, verb, patient, tail, paraphrase in frames:
        for amod in agent_mods:
            for pmod in patient_mods:
                for adv in adverbs:
                    a = f"{amod} {agent}".strip()
                    p = f"{pmod} {patient}".strip()
                    verb_phrase = f"{adv} {verb}" if adv else verb
                    templates.append(
                        (
                            f"The {a} {verb_phrase} the {p} {tail}.",
                            f"The {a} {paraphrase} the {p}.",
                            f"The {p} {verb_phrase} the {a} {tail}.",
                        )
                    )
    rng.shuffle(templates)
    for target, good, bad in templates[:limit]:
        add_choice(rows, rng, target, good, bad, "selectional_preference", "event_role_plausibility")
    return rows


def gen_affordance(rng: random.Random, limit: int) -> list[dict]:
    tools = [
        ("chef", "cut", "cut", "bread", "knife"),
        ("person", "opened", "open", "door", "key"),
        ("carpenter", "hit", "hit", "nail", "hammer"),
        ("student", "wrote", "write", "note", "pencil"),
        ("painter", "painted", "paint", "wall", "brush"),
        ("gardener", "dug", "dig", "hole", "shovel"),
        ("tailor", "cut", "cut", "cloth", "scissors"),
        ("driver", "started", "start", "car", "key"),
        ("cleaner", "swept", "sweep", "floor", "broom"),
        ("cook", "stirred", "stir", "soup", "spoon"),
        ("photographer", "took", "take", "picture", "camera"),
        ("musician", "played", "play", "song", "guitar"),
        ("doctor", "checked", "check", "temperature", "thermometer"),
        ("camper", "lit", "light", "fire", "match"),
        ("builder", "measured", "measure", "board", "ruler"),
    ]
    rows: list[dict] = []
    templates: list[tuple[str, str, str]] = []
    actor_mods = ["", "careful", "young", "experienced", "local", "busy", "skilled", "friendly"]
    object_mods = ["", "small", "large", "old", "new", "wooden", "metal", "clean", "heavy", "light"]
    tool_mods = ["", "sharp", "small", "large", "metal", "wooden", "clean", "heavy", "useful", "old"]
    for actor, past, base, obj, tool in tools:
        for amod in actor_mods:
            for omod in object_mods:
                for tmod in tool_mods:
                    a = f"{amod} {actor}".strip()
                    o = f"{omod} {obj}".strip()
                    t = f"{tmod} {tool}".strip()
                    templates.append(
                        (
                            f"The {a} {past} the {o} with {indef(t)}.",
                            f"{indef(t).capitalize()} can be used to {base} the {o}.",
                            f"The {o} can be used to {base} {indef(t)}.",
                        )
                    )
    rng.shuffle(templates)
    for target, good, bad in templates[:limit]:
        add_choice(rows, rng, target, good, bad, "affordance", "instrument_plausibility")
    return rows


def gen_spatial_containment(rng: random.Random, limit: int) -> list[dict]:
    containment = [
        ("toy", "box"), ("letter", "envelope"), ("coin", "pocket"), ("book", "bag"), ("apple", "basket"),
        ("key", "drawer"), ("fish", "bowl"), ("shirt", "closet"), ("cookie", "jar"), ("pencil", "case"),
        ("sandwich", "lunchbox"), ("blanket", "basket"), ("ring", "box"), ("water", "bottle"), ("soup", "pot"),
    ]
    support = [
        ("cup", "table"), ("book", "shelf"), ("lamp", "desk"), ("plate", "counter"), ("vase", "stand"),
        ("phone", "chair"), ("picture", "wall"), ("clock", "wall"), ("pillow", "bed"), ("plant", "floor"),
    ]
    rows: list[dict] = []
    templates: list[tuple[str, str, str]] = []
    obj_mods = ["", "small", "large", "old", "new", "red", "blue", "heavy", "light", "clean"]
    container_mods = ["", "small", "large", "old", "new", "wooden", "metal", "open", "empty", "nearby"]
    support_mods = ["", "small", "large", "old", "new", "wooden", "metal", "nearby", "low", "flat"]
    for obj, container in containment:
        for omod in obj_mods:
            for cmod in container_mods:
                o = f"{omod} {obj}".strip()
                c = f"{cmod} {container}".strip()
                templates.extend(
                    [
                        (
                            f"The child put the {o} in the {c}.",
                            f"The {o} is inside the {c}.",
                            f"The {c} is inside the {o}.",
                        ),
                        (
                            f"The parent stored the {o} in the {c}.",
                            f"The {c} contains the {o}.",
                            f"The {o} contains the {c}.",
                        ),
                        (
                            f"The person placed the {o} inside the {c}.",
                            f"The {o} fits inside the {c}.",
                            f"The {c} fits inside the {o}.",
                        ),
                    ]
                )
    for obj, support_obj in support:
        for omod in obj_mods:
            for smod in support_mods:
                o = f"{omod} {obj}".strip()
                s = f"{smod} {support_obj}".strip()
                templates.extend(
                    [
                        (
                            f"The {o} is on the {s}.",
                            f"The {s} supports the {o}.",
                            f"The {o} supports the {s}.",
                        ),
                        (
                            f"The person left the {o} on the {s}.",
                            f"The {o} rests on the {s}.",
                            f"The {s} rests on the {o}.",
                        ),
                    ]
                )
    rng.shuffle(templates)
    for target, good, bad in templates[:limit]:
        add_choice(rows, rng, target, good, bad, "spatial_containment", "containment_or_support")
    return rows


def gen_part_whole(rng: random.Random, limit: int) -> list[dict]:
    pairs = [
        ("wheel", "car"), ("page", "book"), ("leaf", "tree"), ("handle", "door"), ("roof", "house"),
        ("screen", "phone"), ("keyboard", "computer"), ("petal", "flower"), ("leg", "table"), ("wing", "bird"),
        ("engine", "car"), ("button", "shirt"), ("finger", "hand"), ("window", "house"), ("chapter", "book"),
        ("seat", "chair"), ("string", "guitar"), ("lens", "camera"), ("pocket", "coat"), ("branch", "tree"),
    ]
    rows: list[dict] = []
    templates: list[tuple[str, str, str]] = []
    part_mods = ["", "small", "large", "old", "new", "broken", "missing", "important", "outer", "inner"]
    whole_mods = ["", "small", "large", "old", "new", "red", "blue", "complete", "familiar", "ordinary"]
    for part, whole in pairs:
        for pmod in part_mods:
            for wmod in whole_mods:
                p = f"{pmod} {part}".strip()
                w = f"{wmod} {whole}".strip()
                templates.extend(
                    [
                        (
                            f"The {p} is part of the {w}.",
                            f"{indef(w).capitalize()} has {indef(p)}.",
                            f"{indef(p).capitalize()} has {indef(w)}.",
                        ),
                        (
                            f"The {w} includes {indef(p)}.",
                            f"The {p} belongs to the {w}.",
                            f"The {w} belongs to the {p}.",
                        ),
                        (
                            f"The {p} came from the {w}.",
                            f"The {p} can be a component of the {w}.",
                            f"The {w} can be a component of the {p}.",
                        ),
                    ]
                )
    rng.shuffle(templates)
    for target, good, bad in templates[:limit]:
        add_choice(rows, rng, target, good, bad, "part_whole", "part_whole_compatibility")
    return rows


def gen_animate_agency(rng: random.Random, limit: int) -> list[dict]:
    animate = ["child", "teacher", "doctor", "student", "parent", "driver", "singer", "farmer", "artist", "runner", "chef", "pilot"]
    inanimate = ["chair", "stone", "table", "pencil", "blanket", "door", "cloud", "spoon", "box", "lamp", "shoe", "bottle"]
    actions = [
        ("asked a question", "ask a question"),
        ("made a promise", "make a promise"),
        ("told a story", "tell a story"),
        ("answered the phone", "answer the phone"),
        ("chose a book", "choose a book"),
        ("planned a trip", "plan a trip"),
        ("remembered a name", "remember a name"),
        ("explained a rule", "explain a rule"),
        ("apologized politely", "apologize politely"),
        ("laughed at a joke", "laugh at a joke"),
    ]
    rows: list[dict] = []
    templates: list[tuple[str, str, str]] = []
    animate_mods = ["", "young", "old", "careful", "friendly", "busy", "local", "helpful", "curious", "patient"]
    inanimate_mods = ["", "old", "new", "small", "large", "wooden", "metal", "heavy", "broken", "quiet"]
    for person in animate:
        for thing in inanimate:
            for pmod in animate_mods:
                for tmod in inanimate_mods:
                    for past, base in actions:
                        p = f"{pmod} {person}".strip()
                        t = f"{tmod} {thing}".strip()
                        templates.append(
                            (
                                f"The {p} {past}.",
                                f"{indef(p).capitalize()} can {base}.",
                                f"{indef(t).capitalize()} can {base}.",
                            )
                        )
    rng.shuffle(templates)
    for target, good, bad in templates[:limit]:
        add_choice(rows, rng, target, good, bad, "animate_agency", "animate_inanimate_agency")
    return rows


def mix_plausibility_examples(rng: random.Random, max_examples: int) -> list[dict]:
    targets = {
        "selectional_preference": int(max_examples * 0.30),
        "affordance": int(max_examples * 0.20),
        "spatial_containment": int(max_examples * 0.20),
        "part_whole": int(max_examples * 0.15),
        "animate_agency": max_examples,
    }
    pools = {
        "selectional_preference": gen_selectional_preference(rng, targets["selectional_preference"]),
        "affordance": gen_affordance(rng, targets["affordance"]),
        "spatial_containment": gen_spatial_containment(rng, targets["spatial_containment"]),
        "part_whole": gen_part_whole(rng, targets["part_whole"]),
        "animate_agency": gen_animate_agency(rng, max_examples),
    }
    selected: list[dict] = []
    for domain in ["selectional_preference", "affordance", "spatial_containment", "part_whole"]:
        selected.extend(pools[domain][: targets[domain]])
    selected.extend(pools["animate_agency"][: max(0, max_examples - len(selected))])
    if len(selected) < max_examples:
        extras = [row for rows in pools.values() for row in rows]
        rng.shuffle(extras)
        seen = {(row["metadata"]["target_sentence"], row["metadata"]["good_context"], row["metadata"]["bad_context"]) for row in selected}
        for row in extras:
            key = (row["metadata"]["target_sentence"], row["metadata"]["good_context"], row["metadata"]["bad_context"])
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= max_examples:
                break
    rng.shuffle(selected)
    return selected[:max_examples]


def domain_counts(rows: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter(row["metadata"].get("domain", "unknown") for row in rows)
    return {
        "selectional_preference": counts.get("selectional_preference", 0),
        "affordance": counts.get("affordance", 0),
        "spatial_containment": counts.get("spatial_containment", 0),
        "part_whole": counts.get("part_whole", 0),
        "animate_agency": counts.get("animate_agency", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BabyLM-community/BabyLM-2026-Strict-Small")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="experiments/multitask_ewok_plausibility_minimal/data/generated")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--max-examples-per-task", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

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
        "conceptual_plausibility_choice": mix_plausibility_examples(rng, args.max_examples_per_task),
        "substitution": gen_substitution(texts, counts, candidates, rng, args.max_examples_per_task),
        "function_word_recovery": [],
        "agreement_prediction": [],
        "grammar_minpair": [],
    }

    stats = {}
    manifest = {
        "dataset": args.dataset,
        "text_column": args.text_column,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "task_probabilities": TASK_PROBABILITIES,
        "tasks": {},
        "ewok_data_used": False,
    }
    for task in TASKS:
        rows = rows_by_task.get(task, [])
        write_jsonl(out_dir / f"{task}.jsonl", rows)
        write_jsonl(out_dir / "samples" / f"{task}_50_examples.jsonl", rows[:50])
        enabled = TASK_PROBABILITIES.get(task, 0.0) > 0 and len(rows) > 0
        stats[task] = {"num_examples": len(rows), "enabled": enabled, "probability": TASK_PROBABILITIES.get(task, 0.0)}
        if task == "conceptual_plausibility_choice":
            stats[task]["domains"] = domain_counts(rows)
        manifest["tasks"][task] = {
            "path": str(out_dir / f"{task}.jsonl"),
            "sample_path": str(out_dir / "samples" / f"{task}_50_examples.jsonl"),
        }
    (out_dir / "task_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "multitask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
