from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--babylm-eval", default="../babylm-eval")
    args = parser.parse_args()
    note = {
        "checkpoint": args.checkpoint,
        "status": "Use the official BabyLM 2026 eval repo for held-out scores.",
        "expected_repo": args.babylm_eval,
    }
    out = Path(args.checkpoint).parent / "evaluation_note.json"
    out.write_text(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
