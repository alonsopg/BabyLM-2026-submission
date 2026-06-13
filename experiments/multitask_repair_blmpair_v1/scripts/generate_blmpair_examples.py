from __future__ import annotations

import argparse
import json
import random
import re
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
    "grammar_minpair",
    "substitution",
    "function_word_recovery",
    "agreement_prediction",
]

TASK_PROBABILITIES = {
    "mlm": 0.60,
    "rtd": 0.10,
    "connective": 0.075,
    "definiteness": 0.075,
    "collocation": 0.05,
    "grammar_minpair": 0.10,
    "substitution": 0.0,
    "function_word_recovery": 0.0,
    "agreement_prediction": 0.0,
}

SG_PRONOUNS = {"he", "she", "it", "this", "that"}
PL_PRONOUNS = {"they", "we", "these", "those"}
SINGULAR_DETS = {"this", "that"}
PLURAL_DETS = {"these", "those"}
DET_SWAP = {"this": "these", "these": "this", "that": "those", "those": "that"}
BE_SWAP = {"is": "are", "are": "is", "was": "were", "were": "was"}
AUX_SWAP = {"has": "have", "have": "has", "does": "do", "do": "does"}
PUNCT = {".", ",", "!", "?", ";", ":"}
IRREGULAR_PLURALS = {"children", "people", "men", "women", "teeth", "feet", "mice"}
SINGULAR_S_EXCEPTIONS = {"news", "series", "species", "means", "physics", "mathematics", "economics"}
BAD_NEXT_WORDS = {"of", "to", "for", "in", "on", "at", "with", "by", "from", "and", "or", "but"}
FUNCTION_OR_NON_NOUN = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our",
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "not", "n't", "no", "yes", "oh", "well", "now", "then", "there", "here",
    "because", "if", "although", "though", "while", "since", "that",
    "later", "see", "out", "away", "back", "round", "along", "through",
}
COMMON_VERB_FORMS = {
    "going", "coming", "doing", "being", "having", "getting", "making", "taking", "saying", "working", "talking",
    "said", "made", "went", "came", "got", "took", "seen", "known", "done",
}


def is_alpha_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token))


def pluralish(token: str) -> bool:
    lower = token.lower()
    if lower in IRREGULAR_PLURALS:
        return True
    if lower in SINGULAR_S_EXCEPTIONS:
        return False
    return lower.endswith("s") and not lower.endswith("ss")


def nounish_for_number(token: str) -> bool:
    lower = token.lower()
    if not is_alpha_token(token):
        return False
    if len(lower) < 3 or lower in FUNCTION_OR_NON_NOUN or lower in BAD_NEXT_WORDS or lower in COMMON_VERB_FORMS:
        return False
    if lower.endswith("ly"):
        return False
    return True


def clean_sentence(toks: list[str]) -> bool:
    alpha = [tok for tok in toks if is_alpha_token(tok)]
    return 5 <= len(toks) <= 32 and len(alpha) >= 4


def replace_one(toks: list[str], idx: int, replacement: str) -> str:
    new = toks[:]
    if toks[idx][:1].isupper():
        replacement = replacement.capitalize()
    new[idx] = replacement
    return render(new)


def sentence_from_tokens(toks: list[str]) -> str:
    sent = render(toks)
    if sent and sent[-1] not in ".!?":
        sent += "."
    return sent


def add_pair(rows: list[dict], rng: random.Random, good: str, bad: str, phenomenon: str, rule: str) -> None:
    good = good.strip()
    bad = bad.strip()
    if not good or not bad or good == bad:
        return
    if rng.random() < 0.5:
        first, second, target = good, bad, 0
    else:
        first, second, target = bad, good, 1
    rows.append(
        {
            "task": "grammar_minpair",
            "input": f"[CLS] {first} [SEP] {second} [SEP]",
            "target": target,
            "metadata": {
                "phenomenon": phenomenon,
                "good": good,
                "bad": bad,
                "generation_rule": rule,
                "label_meaning": "0 = first sentence better, 1 = second sentence better",
            },
        }
    )


def gen_subject_verb_agreement(texts: list[str], rng: random.Random, limit: int) -> list[dict]:
    rows: list[dict] = []
    for text in texts:
        toks = words(text)
        if not clean_sentence(toks):
            continue
        low = [tok.lower() for tok in toks]
        for idx, tok in enumerate(low):
            if tok not in BE_SWAP or idx < 1 or idx + 2 >= len(toks):
                continue
            if idx > 0 and idx + 1 < len(toks) and (low[idx - 1] in BE_SWAP.values() or low[idx + 1] in BE_SWAP.values()):
                continue
            expected_plural: bool | None = None
            if low[idx - 1] in SG_PRONOUNS:
                expected_plural = False
            elif low[idx - 1] in PL_PRONOUNS:
                expected_plural = True
            elif idx >= 2 and low[idx - 2] in {"the", "a", "an", "this", "that", "these", "those"} and nounish_for_number(toks[idx - 1]):
                if low[idx - 2] in SINGULAR_DETS:
                    expected_plural = False
                elif low[idx - 2] in PLURAL_DETS:
                    expected_plural = True
                else:
                    expected_plural = pluralish(low[idx - 1])
            if expected_plural is None:
                continue
            is_plural_verb = tok in {"are", "were"}
            if is_plural_verb != expected_plural:
                continue
            good = sentence_from_tokens(toks)
            bad = replace_one(toks, idx, BE_SWAP[tok])
            add_pair(rows, rng, good, bad, "subject_verb_agreement", "corpus_be_agreement_swap")
            break
        if len(rows) >= limit:
            break
    return rows


def gen_determiner_noun_agreement(texts: list[str], rng: random.Random, limit: int) -> list[dict]:
    rows: list[dict] = []
    for text in texts:
        toks = words(text)
        if not clean_sentence(toks):
            continue
        low = [tok.lower() for tok in toks]
        for idx, tok in enumerate(low[:-1]):
            if tok not in DET_SWAP or idx + 2 >= len(toks):
                continue
            noun = low[idx + 1]
            if not nounish_for_number(toks[idx + 1]):
                continue
            noun_plural = pluralish(noun)
            det_plural = tok in PLURAL_DETS
            if noun_plural != det_plural:
                continue
            good = sentence_from_tokens(toks)
            bad = replace_one(toks, idx, DET_SWAP[tok])
            add_pair(rows, rng, good, bad, "determiner_noun_agreement", "corpus_this_that_number_swap")
            break
        if len(rows) >= limit:
            break
    return rows


def gen_auxiliary_agreement(texts: list[str], rng: random.Random, limit: int) -> list[dict]:
    rows: list[dict] = []
    for text in texts:
        toks = words(text)
        if not clean_sentence(toks):
            continue
        low = [tok.lower() for tok in toks]
        for idx, tok in enumerate(low):
            if tok not in AUX_SWAP or idx < 1 or idx + 1 >= len(toks):
                continue
            cue = low[idx - 1]
            if cue in {"i", "you"}:
                continue
            expected_plural: bool | None = None
            if cue in SG_PRONOUNS:
                expected_plural = False
            elif cue in PL_PRONOUNS:
                expected_plural = True
            if expected_plural is None:
                continue
            is_plural_aux = tok in {"have", "do"}
            if is_plural_aux != expected_plural:
                continue
            good = sentence_from_tokens(toks)
            bad = replace_one(toks, idx, AUX_SWAP[tok])
            add_pair(rows, rng, good, bad, "auxiliary_agreement", "corpus_has_have_does_do_swap")
            break
        if len(rows) >= limit:
            break
    return rows


def gen_subject_verb_templates(rng: random.Random, limit: int) -> list[dict]:
    noun_pairs = [
        ("dog", "dogs"), ("child", "children"), ("student", "students"), ("teacher", "teachers"), ("artist", "artists"),
        ("doctor", "doctors"), ("neighbor", "neighbors"), ("parent", "parents"), ("driver", "drivers"), ("writer", "writers"),
        ("worker", "workers"), ("visitor", "visitors"), ("friend", "friends"), ("member", "members"), ("player", "players"),
        ("singer", "singers"), ("reader", "readers"), ("customer", "customers"), ("patient", "patients"), ("speaker", "speakers"),
        ("garden", "gardens"), ("window", "windows"), ("machine", "machines"), ("letter", "letters"), ("picture", "pictures"),
    ]
    adjectives = ["", "young", "old", "careful", "quiet", "happy", "tired", "new", "local", "friendly", "important", "small"]
    singular_subjects = ["he", "she", "it", "this", "that"]
    plural_subjects = ["they", "we", "these", "those"]
    for singular, plural in noun_pairs:
        for adj in adjectives:
            adj_part = f"{adj} " if adj else ""
            singular_subjects.append(f"the {adj_part}{singular}")
            plural_subjects.append(f"the {adj_part}{plural}")
    present_predicates = ["barking", "playing", "waiting", "working", "sleeping", "running", "talking", "reading", "listening", "arriving", "leaving", "laughing", "standing", "walking", "singing", "moving", "helping", "watching"]
    complements = ["ready", "quiet", "nearby", "outside", "inside", "available", "careful", "happy", "late", "early", "important", "visible", "safe", "useful", "popular"]
    templates: list[tuple[str, str]] = []
    for subj in singular_subjects:
        cap = subj.capitalize()
        for pred in present_predicates:
            templates.append((f"{cap} is {pred}.", f"{cap} are {pred}."))
        for comp in complements:
            templates.append((f"{cap} was {comp}.", f"{cap} were {comp}."))
    for subj in plural_subjects:
        cap = subj.capitalize()
        for pred in present_predicates:
            templates.append((f"{cap} are {pred}.", f"{cap} is {pred}."))
        for comp in complements:
            templates.append((f"{cap} were {comp}.", f"{cap} was {comp}."))
    rng.shuffle(templates)
    rows: list[dict] = []
    for good, bad in templates[:limit]:
        add_pair(rows, rng, good, bad, "subject_verb_agreement", "template_be_subject_number_swap")
    return rows


def gen_determiner_noun_templates(rng: random.Random, limit: int) -> list[dict]:
    nouns = [
        ("book", "books"), ("dog", "dogs"), ("student", "students"), ("teacher", "teachers"), ("car", "cars"),
        ("house", "houses"), ("letter", "letters"), ("idea", "ideas"), ("song", "songs"), ("chair", "chairs"),
        ("garden", "gardens"), ("river", "rivers"), ("window", "windows"), ("table", "tables"), ("picture", "pictures"),
        ("question", "questions"), ("answer", "answers"), ("story", "stories"), ("city", "cities"), ("family", "families"),
    ]
    adjectives = ["old", "new", "small", "large", "quiet", "bright", "careful", "familiar", "important", "simple", ""]
    predicates = ["ready", "nearby", "available", "interesting", "useful", "missing", "expensive", "popular"]
    templates: list[tuple[str, str]] = []
    for singular, plural in nouns:
        for adj in adjectives:
            adj_part = f"{adj} " if adj else ""
            for pred in predicates:
                templates.append((f"This {adj_part}{singular} is {pred}.", f"These {adj_part}{singular} is {pred}."))
                templates.append((f"That {adj_part}{singular} was {pred}.", f"Those {adj_part}{singular} was {pred}."))
                templates.append((f"These {adj_part}{plural} are {pred}.", f"This {adj_part}{plural} are {pred}."))
                templates.append((f"Those {adj_part}{plural} were {pred}.", f"That {adj_part}{plural} were {pred}."))
    rng.shuffle(templates)
    rows: list[dict] = []
    for good, bad in templates[:limit]:
        add_pair(rows, rng, good, bad, "determiner_noun_agreement", "template_this_these_that_those_number_swap")
    return rows


def gen_auxiliary_templates(rng: random.Random, limit: int) -> list[dict]:
    noun_pairs = [
        ("student", "students"), ("teacher", "teachers"), ("child", "children"), ("doctor", "doctors"), ("artist", "artists"),
        ("neighbor", "neighbors"), ("parent", "parents"), ("driver", "drivers"), ("writer", "writers"), ("worker", "workers"),
        ("visitor", "visitors"), ("friend", "friends"), ("member", "members"), ("player", "players"), ("speaker", "speakers"),
    ]
    adjectives = ["", "young", "old", "careful", "quiet", "local", "friendly", "important"]
    singular_subjects = ["he", "she", "it", "this student", "that teacher"]
    plural_subjects = ["they", "we", "these students", "those teachers"]
    for singular, plural in noun_pairs:
        for adj in adjectives:
            adj_part = f"{adj} " if adj else ""
            singular_subjects.append(f"the {adj_part}{singular}")
            plural_subjects.append(f"the {adj_part}{plural}")
    objects = ["a car", "a book", "a question", "a plan", "the answer", "the ticket", "some time", "several ideas", "a reason", "a choice", "the address", "the report"]
    verbs = ["like it", "need help", "want more", "read well", "work here", "arrive early", "listen carefully", "agree today", "speak clearly", "move quickly", "wait outside", "help often"]
    templates: list[tuple[str, str]] = []
    for subj in singular_subjects:
        cap = subj.capitalize()
        for obj in objects:
            templates.append((f"{cap} has {obj}.", f"{cap} have {obj}."))
        for verb in verbs:
            templates.append((f"{cap} does {verb}.", f"{cap} do {verb}."))
    for subj in plural_subjects:
        cap = subj.capitalize()
        for obj in objects:
            templates.append((f"{cap} have {obj}.", f"{cap} has {obj}."))
        for verb in verbs:
            templates.append((f"{cap} do {verb}.", f"{cap} does {verb}."))
    rng.shuffle(templates)
    rows: list[dict] = []
    for good, bad in templates[:limit]:
        add_pair(rows, rng, good, bad, "auxiliary_agreement", "template_has_have_does_do_subject_number_swap")
    return rows


def gen_npi_templates(rng: random.Random, limit: int) -> list[dict]:
    singular_nouns = ["student", "teacher", "child", "doctor", "artist", "neighbor", "driver", "writer", "parent", "friend"]
    plural_nouns = ["students", "teachers", "children", "doctors", "artists", "neighbors", "drivers", "writers", "parents", "friends"]
    verb_phrases = ["complained", "arrived", "called", "noticed", "objected", "returned", "visited", "agreed", "answered", "laughed"]
    rows: list[dict] = []
    templates: list[tuple[str, str]] = []
    for noun in singular_nouns:
        for vp in verb_phrases:
            templates.append((f"No {noun} has ever {vp}.", f"A {noun} has ever {vp}."))
    for noun in plural_nouns:
        for vp in verb_phrases:
            templates.append((f"No {noun} have ever {vp}.", f"Some {noun} have ever {vp}."))
    for subj, bad_subj in [("Nobody", "Somebody"), ("Nothing", "Something")]:
        for vp in verb_phrases:
            templates.append((f"{subj} has ever {vp}.", f"{bad_subj} has ever {vp}."))
    rng.shuffle(templates)
    for good, bad in templates[:limit]:
        add_pair(rows, rng, good, bad, "npi_licensing", "template_no_nobody_nothing_ever")
    return rows


def gen_reflexive_templates(rng: random.Random, limit: int) -> list[dict]:
    female_subjects = ["Alice", "Mary", "Anna", "Sarah", "The girl", "The woman", "The mother"]
    male_subjects = ["John", "Bob", "Tom", "Peter", "The boy", "The man", "The father"]
    verbs = ["saw", "hurt", "helped", "washed", "blamed", "introduced"]
    rows: list[dict] = []
    templates: list[tuple[str, str]] = []
    for subj in female_subjects:
        for verb in verbs:
            templates.append((f"{subj} {verb} herself.", f"{subj} {verb} himself."))
    for subj in male_subjects:
        for verb in verbs:
            templates.append((f"{subj} {verb} himself.", f"{subj} {verb} herself."))
    rng.shuffle(templates)
    for good, bad in templates[:limit]:
        add_pair(rows, rng, good, bad, "reflexive_binding", "template_gender_reflexive_swap")
    return rows


def mix_grammar_minpairs(texts: list[str], rng: random.Random, max_examples: int) -> list[dict]:
    targets = {
        "subject_verb_agreement": int(max_examples * 0.35),
        "determiner_noun_agreement": int(max_examples * 0.25),
        "auxiliary_agreement": int(max_examples * 0.20),
        "npi_licensing": int(max_examples * 0.10),
        "reflexive_binding": max_examples,
    }
    pools = {
        "subject_verb_agreement": gen_subject_verb_templates(rng, targets["subject_verb_agreement"]),
        "determiner_noun_agreement": gen_determiner_noun_templates(rng, targets["determiner_noun_agreement"]),
        "auxiliary_agreement": gen_auxiliary_templates(rng, targets["auxiliary_agreement"]),
        "npi_licensing": gen_npi_templates(rng, targets["npi_licensing"]),
        "reflexive_binding": gen_reflexive_templates(rng, max_examples),
    }
    selected: list[dict] = []
    for phenomenon in ["subject_verb_agreement", "determiner_noun_agreement", "auxiliary_agreement", "npi_licensing"]:
        selected.extend(pools[phenomenon][: targets[phenomenon]])
    remaining = max(0, max_examples - len(selected))
    selected.extend(pools["reflexive_binding"][:remaining])
    if len(selected) < max_examples:
        extras = [row for rows in pools.values() for row in rows]
        rng.shuffle(extras)
        seen = {(row["metadata"]["good"], row["metadata"]["bad"]) for row in selected}
        for row in extras:
            key = (row["metadata"]["good"], row["metadata"]["bad"])
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= max_examples:
                break
    rng.shuffle(selected)
    return selected[:max_examples]


def phenomenon_counts(rows: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter(row["metadata"].get("phenomenon", "unknown") for row in rows)
    return {
        "subject_verb_agreement": counts.get("subject_verb_agreement", 0),
        "determiner_noun_agreement": counts.get("determiner_noun_agreement", 0),
        "auxiliary_agreement": counts.get("auxiliary_agreement", 0),
        "npi_licensing": counts.get("npi_licensing", 0),
        "reflexive_binding": counts.get("reflexive_binding", 0),
    }


def empty_disabled_rows(task: str) -> list[dict]:
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BabyLM-community/BabyLM-2026-Strict-Small")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="experiments/multitask_repair_blmpair_v1/data/generated")
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
        "grammar_minpair": mix_grammar_minpairs(texts, rng, args.max_examples_per_task),
        "substitution": gen_substitution(texts, counts, candidates, rng, args.max_examples_per_task),
        "function_word_recovery": empty_disabled_rows("function_word_recovery"),
        "agreement_prediction": empty_disabled_rows("agreement_prediction"),
    }

    stats = {}
    manifest = {
        "dataset": args.dataset,
        "text_column": args.text_column,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "task_probabilities": TASK_PROBABILITIES,
        "tasks": {},
    }
    for task in TASKS:
        rows = rows_by_task.get(task, [])
        write_jsonl(out_dir / f"{task}.jsonl", rows)
        write_jsonl(out_dir / "samples" / f"{task}_50_examples.jsonl", rows[:50])
        enabled = TASK_PROBABILITIES.get(task, 0.0) > 0 and len(rows) > 0
        stats[task] = {"num_examples": len(rows), "enabled": enabled, "probability": TASK_PROBABILITIES.get(task, 0.0)}
        if task == "grammar_minpair":
            stats[task]["phenomena"] = phenomenon_counts(rows)
        manifest["tasks"][task] = {
            "path": str(out_dir / f"{task}.jsonl"),
            "sample_path": str(out_dir / "samples" / f"{task}_50_examples.jsonl"),
        }
    (out_dir / "task_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "multitask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
