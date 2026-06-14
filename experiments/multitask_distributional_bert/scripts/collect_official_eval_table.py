from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TASK_PATHS = {
    "BLiMP": ("blimp", "blimp_fast", "best_temperature_report.txt"),
    "BLiMP Supplement": ("blimp", "supplement_fast", "best_temperature_report.txt"),
    "EWoK": ("ewok", "ewok_fast", "best_temperature_report.txt"),
    "Entity Tracking": ("entity_tracking", "entity_tracking_fast", "best_temperature_report.txt"),
}


def average_accuracy(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"### AVERAGE ACCURACY\s+([0-9.]+)", text)
    return match.group(1) if match else ""


def reading_scores(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    text = path.read_text(encoding="utf-8")
    eye = re.search(r"EYE TRACKING SCORE:\s*([0-9.\-]+)", text)
    spr = re.search(r"SELF-PACED READING SCORE:\s*([0-9.\-]+)", text)
    return (eye.group(1) if eye else "", spr.group(1) if spr else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results-root", default="/home/paperspace/babylm-hhm/resources/babylm-eval/strict/results")
    parser.add_argument("--revision-name", default="main")
    parser.add_argument("--output-dir", default="experiments/multitask_distributional_bert/metrics")
    parser.add_argument("run_names", nargs="+")
    args = parser.parse_args()

    results_root = Path(args.eval_results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_name in args.run_names:
        base = results_root / run_name / args.revision_name / "zero_shot" / "mlm"
        row = {"System": run_name}
        for label, parts in TASK_PATHS.items():
            row[label] = average_accuracy(base / parts[0] / parts[1] / parts[2])
        row["Reading Eye"] = ""
        row["Reading SPR"] = ""
        eye, spr = reading_scores(base / "reading" / "report.txt")
        row["Reading Eye"] = eye
        row["Reading SPR"] = spr
        rows.append(row)

    fieldnames = ["System", "BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "Reading Eye", "Reading SPR"]
    csv_path = output_dir / "official_fast_eval_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Official Fast Eval Results", ""]
    md_lines.append("| " + " | ".join(fieldnames) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(row.get(name, "") for name in fieldnames) + " |")
    md_path = output_dir / "official_fast_eval_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_path = Path("paper/tables/multitask_distributional_eval_fast.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{System} & \\textbf{BLiMP} & \\textbf{Sup.} & \\textbf{EWoK} & \\textbf{Entity} & \\textbf{Eye} & \\textbf{SPR} \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = [row["System"], row["BLiMP"], row["BLiMP Supplement"], row["EWoK"], row["Entity Tracking"], row["Reading Eye"], row["Reading SPR"]]
        vals = [v if v else "--" for v in vals]
        lines.append(" & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Official BabyLM strict fast-evaluation results for the multi-task distributional BERT experiment. Accuracy-style scores are percentages; reading scores are official predictive-power scores.}",
        "\\label{tab:multitask-distributional-fast-eval}",
        "\\end{table*}",
    ]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
