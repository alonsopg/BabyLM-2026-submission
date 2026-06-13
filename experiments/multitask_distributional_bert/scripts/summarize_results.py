from __future__ import annotations

import csv
import json
from pathlib import Path


def read_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        return []
    with log_path.open(newline="") as f:
        return list(csv.DictReader(f))


def validation_value(row: dict) -> float | None:
    raw = row.get("val_loss") or row.get("val_mlm_loss")
    if raw in {None, ""}:
        return None
    return float(raw)


def main() -> None:
    root = Path("experiments/multitask_distributional_bert")
    normal_root = root / "checkpoints/normal_bert"
    multitask_root = root / "checkpoints/multitask_bert"
    summary = {"normal_bert": [], "multitask_bert": []}
    for label, base in [("normal_bert", normal_root), ("multitask_bert", multitask_root)]:
        for run_dir in sorted(base.glob("*/*")):
            rows = read_log(run_dir)
            if not rows:
                continue
            scored = [row for row in rows if validation_value(row) is not None]
            best = min(scored, key=validation_value) if scored else None
            final = rows[-1]
            summary[label].append({
                "run_dir": str(run_dir),
                "final": final,
                "best": best,
            })
    out_json = root / "metrics/comparison_summary.json"
    out_md = root / "metrics/comparison_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    lines = ["# Comparison Summary", ""]
    for label, rows in summary.items():
        lines += [f"## {label}", ""]
        if not rows:
            lines += ["No completed logs found.", ""]
            continue
        for item in rows:
            row = item["final"]
            best = item["best"]
            val = row.get("val_loss") or row.get("val_mlm_loss")
            train = row.get("train_loss") or row.get("loss")
            best_text = "best_val=None"
            if best:
                best_text = f"best_step={best.get('step')}, best_val={validation_value(best)}"
            lines += [
                f"- `{item['run_dir']}`: final_step={row.get('step')}, "
                f"train_loss={train}, final_val={val}, {best_text}, early_stop={row.get('early_stop')}",
            ]
        lines.append("")
    out_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
