from __future__ import annotations

import argparse
import csv
from pathlib import Path


def last_row(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    return rows[-1] if rows else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="paper_exports/main_results.csv")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    rows = []
    for log in sorted(Path(args.outputs).glob("*/seed_*/train_log.csv")):
        model = log.parents[1].name
        if model.startswith("smoke") and not args.include_smoke:
            continue
        seed = log.parent.name.replace("seed_", "")
        final = last_row(log)
        if not final:
            continue
        rows.append({
            "Model": model,
            "Seed": seed,
            "Validation loss": final["val_loss"],
            "Overall BabyLM score": "",
            "BLiMP": "",
            "EWoK": "",
            "Entity tracking": "",
            "Training time": final["elapsed_sec"],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Seed", "Validation loss", "Overall BabyLM score", "BLiMP", "EWoK", "Entity tracking", "Training time"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
