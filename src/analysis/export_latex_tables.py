from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def mean(values):
    vals = [float(v) for v in values if v not in {"", None}]
    return sum(vals) / len(vals) if vals else None


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="paper_exports/main_results.csv")
    parser.add_argument("--latex", default="../babylm-sketchselect/paper_draft/tables/main_results.tex")
    args = parser.parse_args()
    grouped = defaultdict(list)
    for row in csv.DictReader(Path(args.csv).open()):
        grouped[row["Model"]].append(row)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Seeds} & \\textbf{Val. Loss} & \\textbf{BabyLM} & \\textbf{BLiMP} & \\textbf{EWoK} & \\textbf{Entity} \\\\",
        "\\midrule",
    ]
    for model, rows in sorted(grouped.items()):
        val = mean([r["Validation loss"] for r in rows])
        babylm = mean([r["Overall BabyLM score"] for r in rows])
        blimp = mean([r["BLiMP"] for r in rows])
        ewok = mean([r["EWoK"] for r in rows])
        entity = mean([r["Entity tracking"] for r in rows])
        lines.append(f"{model.replace('_', '-')} & {len(rows)} & {fmt(val)} & {fmt(babylm)} & {fmt(blimp)} & {fmt(ewok)} & {fmt(entity)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Main BabyLM-style results. Validation loss is populated by the local MLM training pipeline; official BabyLM scores should be filled from the 2026 evaluation pipeline outputs.}",
        "\\label{tab:main-results}",
        "\\end{table*}",
    ]
    Path(args.latex).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
