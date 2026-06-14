from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


TASKS = ["mlm", "rtd", "substitution", "connective", "definiteness", "collocation"]
CONNECTIVES = ["because", "so", "but", "although", "however", "therefore", "when", "while", "if", "then", "before", "after", "since", "though", "unless"]
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "while", "when", "to", "of", "in", "on",
    "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "i", "you",
    "he", "she", "it", "we", "they", "this", "that", "these", "those",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_class(token: str) -> str:
    t = token.lower()
    if not t.isalpha() or t in STOPWORDS:
        return "other"
    if t.endswith("ly"):
        return "adv"
    if t.endswith(("ing", "ed", "ize", "ise")):
        return "verb"
    if t.endswith(("ous", "ful", "ive", "able", "al", "ic", "less")):
        return "adj"
    return "noun"


def freq_bucket(count: int) -> str:
    if count < 10:
        return "rare"
    if count < 100:
        return "low"
    if count < 1000:
        return "medium"
    return "high"


def render(tokens: list[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_negative(token: str, candidates: dict[tuple[str, str], list[str]], counts: Counter[str], rng: random.Random) -> str | None:
    cls = word_class(token)
    if cls == "other":
        return None
    bucket = freq_bucket(counts[token.lower()])
    pool = [w for w in candidates.get((cls, bucket), []) if w != token.lower()]
    if not pool:
        pool = [w for (c, _), vals in candidates.items() if c == cls for w in vals if w != token.lower()]
    if not pool:
        return None
    return rng.choice(pool)


def build_stats(texts: list[str]) -> tuple[Counter[str], dict[tuple[str, str], list[str]], Counter[tuple[str, str]], Counter[tuple[str, str, str]]]:
    counts: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    trigrams: Counter[tuple[str, str, str]] = Counter()
    for text in texts:
        toks = [t.lower() for t in words(text) if t.isalpha()]
        counts.update(toks)
        bigrams.update(zip(toks, toks[1:]))
        trigrams.update(zip(toks, toks[1:], toks[2:]))
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for token, count in counts.items():
        cls = word_class(token)
        if cls != "other":
            buckets[(cls, freq_bucket(count))].append(token)
    for key in buckets:
        buckets[key].sort(key=lambda w: counts[w], reverse=True)
        buckets[key] = buckets[key][:2000]
    return counts, buckets, bigrams, trigrams


def gen_mlm(texts: list[str], rng: random.Random, max_examples: int) -> list[dict]:
    rows = []
    for text in texts:
        toks = words(text)
        valid = [i for i, tok in enumerate(toks) if tok.isalpha() and tok.lower() not in STOPWORDS]
        if not valid:
            continue
        n_mask = max(1, round(len(valid) * 0.15))
        chosen = sorted(rng.sample(valid, min(n_mask, len(valid))))
        masked = toks[:]
        targets = []
        for idx in chosen:
            targets.append(toks[idx])
            masked[idx] = "[MASK]"
        rows.append({"task": "mlm", "input": render(masked), "target": {"tokens": targets}, "metadata": {"original": text, "generation_rule": "offline_random_15_percent_mlm"}})
        if len(rows) >= max_examples:
            break
    return rows


def gen_rtd(texts: list[str], counts: Counter[str], candidates: dict[tuple[str, str], list[str]], rng: random.Random, max_examples: int) -> list[dict]:
    rows = []
    for text in texts:
        toks = words(text)
        valid = [i for i, tok in enumerate(toks) if word_class(tok) in {"noun", "verb", "adj", "adv"}]
        if not valid:
            continue
        idx = rng.choice(valid)
        neg = choose_negative(toks[idx], candidates, counts, rng)
        if not neg:
            continue
        corrupted = toks[:]
        old = corrupted[idx]
        corrupted[idx] = neg
        labels = [0] * len(corrupted)
        labels[idx] = 1
        rows.append({
            "task": "rtd",
            "input": render(corrupted),
            "target": {"token_labels": labels},
            "metadata": {"original": text, "replacements": [{"from": old, "to": neg}], "generation_rule": "same_class_frequency_bucket_negative"},
        })
        if len(rows) >= max_examples:
            break
    return rows


def gen_connective(texts: list[str], max_examples: int) -> list[dict]:
    rows = []
    conn_re = re.compile(r"\b(" + "|".join(map(re.escape, CONNECTIVES)) + r")\b", re.IGNORECASE)
    for text in texts:
        for m in conn_re.finditer(text):
            before = text[:m.start()].strip(" ,;:")
            after = text[m.end():].strip(" ,;:")
            if not before or not after:
                continue
            if m.start() < 3:
                continue
            inp = text[:m.start()] + "[unused1]" + text[m.end():]
            rows.append({"task": "connective", "input": inp, "target": m.group(1).lower(), "metadata": {"original": text, "generation_rule": "fixed_connective_inventory_unused1"}})
            break
        if len(rows) >= max_examples:
            break
    return rows


def gen_definiteness(texts: list[str], max_examples: int) -> list[dict]:
    rows = []
    art_re = re.compile(r"\b(a|an|the)\s+([A-Za-z][A-Za-z'-]*)\b", re.IGNORECASE)
    for text in texts:
        matches = list(art_re.finditer(text))
        if not matches:
            continue
        out = []
        last = 0
        targets = []
        for m in matches[:4]:
            noun = m.group(2).lower()
            if noun in {"school", "home", "bed", "piano"}:
                continue
            out.append(text[last:m.start(1)])
            out.append("[unused2]")
            targets.append({"label": "definite" if m.group(1).lower() == "the" else "indefinite", "original": m.group(1)})
            last = m.end(1)
        if not targets:
            continue
        out.append(text[last:])
        rows.append({"task": "definiteness", "input": "".join(out), "target": targets, "metadata": {"original": text, "generation_rule": "a_an_the_binary_unused2"}})
        if len(rows) >= max_examples:
            break
    return rows


def gen_collocation(counts: Counter[str], bigrams: Counter[tuple[str, str]], candidates: dict[tuple[str, str], list[str]], rng: random.Random, max_examples: int) -> list[dict]:
    rows = []
    frequent = [(bg, c) for bg, c in bigrams.items() if c >= 3 and all(word_class(t) != "other" for t in bg)]
    frequent.sort(key=lambda item: item[1], reverse=True)
    for (w1, w2), _ in frequent:
        replace_idx = rng.choice([0, 1])
        original = [w1, w2]
        neg = choose_negative(original[replace_idx], candidates, counts, rng)
        if not neg:
            continue
        corrupted = original[:]
        corrupted[replace_idx] = neg
        if bigrams.get(tuple(corrupted), 0) >= 2:
            continue
        positive = " ".join(original)
        negative = " ".join(corrupted)
        if rng.random() < 0.5:
            inp, target = f"[CLS] {positive} [SEP] {negative} [SEP]", 0
        else:
            inp, target = f"[CLS] {negative} [SEP] {positive} [SEP]", 1
        rows.append({"task": "collocation", "input": inp, "target": target, "metadata": {"positive": positive, "negative": negative, "generation_rule": "frequent_bigram_same_class_replacement"}})
        if len(rows) >= max_examples:
            break
    return rows


def gen_substitution(texts: list[str], counts: Counter[str], candidates: dict[tuple[str, str], list[str]], rng: random.Random, max_examples: int) -> list[dict]:
    rows = []
    context_to_words: dict[tuple[str, str], set[str]] = defaultdict(set)
    tokenized = []
    for text in texts:
        toks = [t.lower() for t in words(text) if t.isalpha()]
        tokenized.append((text, toks))
        for i in range(1, len(toks) - 1):
            if word_class(toks[i]) != "other":
                context_to_words[(toks[i - 1], toks[i + 1])].add(toks[i])
    positives = [(ctx, sorted(vals)) for ctx, vals in context_to_words.items() if 2 <= len(vals) <= 8]
    rng.shuffle(positives)
    for ctx, vals in positives:
        target = rng.choice(vals)
        candidate = rng.choice([v for v in vals if v != target])
        sent = f"{ctx[0]} {target} {ctx[1]}"
        rows.append({"task": "substitution", "input": f"[CLS] {sent} [SEP] target: {target} [SEP] candidate: {candidate} [SEP]", "target": 1, "metadata": {"target_word": target, "candidate": candidate, "generation_rule": "shared_local_context_positive"}})
        neg = choose_negative(target, candidates, counts, rng)
        if neg:
            rows.append({"task": "substitution", "input": f"[CLS] {sent} [SEP] target: {target} [SEP] candidate: {neg} [SEP]", "target": 0, "metadata": {"target_word": target, "candidate": neg, "generation_rule": "same_class_frequency_bucket_negative"}})
        if len(rows) >= max_examples:
            break
    return rows[:max_examples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BabyLM-community/BabyLM-2026-Strict-Small")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="experiments/multitask_distributional_bert/data/generated")
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
        "substitution": gen_substitution(texts, counts, candidates, rng, args.max_examples_per_task),
    }

    stats = {}
    manifest = {"dataset": args.dataset, "text_column": args.text_column, "max_rows": args.max_rows, "seed": args.seed, "tasks": {}}
    for task in TASKS:
        rows = rows_by_task.get(task, [])
        write_jsonl(out_dir / f"{task}.jsonl", rows)
        write_jsonl(out_dir / "samples" / f"{task}_50_examples.jsonl", rows[:50])
        enabled = len(rows) >= 1000 or task in {"mlm", "rtd", "definiteness", "connective"}
        stats[task] = {"num_examples": len(rows), "enabled": enabled}
        manifest["tasks"][task] = {"path": str(out_dir / f"{task}.jsonl"), "sample_path": str(out_dir / "samples" / f"{task}_50_examples.jsonl")}
    (out_dir / "task_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "multitask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
