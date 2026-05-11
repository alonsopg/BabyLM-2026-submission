from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "main": ["configs/baseline_random_mlm.yaml", "configs/hhm_entity_error.yaml"],
    "ablations": [
        "configs/hh_entity_only.yaml",
        "configs/hh_error_only.yaml",
        "configs/random_sketch_control.yaml",
        "configs/frequency_only.yaml",
    ],
    "smoke": ["configs/smoke_baseline.yaml", "configs/smoke_hh_entity.yaml"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "main", "ablations", "all"], default="main")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configs = []
    if args.suite == "all":
        configs = CONFIGS["main"] + CONFIGS["ablations"]
    else:
        configs = CONFIGS[args.suite]
    for config in configs:
        for seed in args.seeds:
            cmd = [args.python, "-m", "src.training.train", "--config", config, "--seed", str(seed)]
            if args.max_steps is not None:
                cmd += ["--max-steps", str(args.max_steps)]
            if args.max_train_rows is not None:
                cmd += ["--max-train-rows", str(args.max_train_rows)]
            if args.name_suffix:
                cmd += ["--name-suffix", args.name_suffix]
            if args.resume:
                cmd += ["--resume"]
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
