from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="paper_exports/validation_loss.png")
    args = parser.parse_args()
    for log in sorted(Path(args.outputs).glob("*/seed_*/train_log.csv")):
        rows = list(csv.DictReader(log.open()))
        if not rows:
            continue
        label = f"{log.parents[1].name}/{log.parent.name}"
        plt.plot([int(r["step"]) for r in rows], [float(r["val_loss"]) for r in rows], label=label)
    plt.xlabel("Training step")
    plt.ylabel("Validation MLM loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
