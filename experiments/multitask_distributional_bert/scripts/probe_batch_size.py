from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/multitask_distributional_bert/configs/batch_size_probe.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    probe_cfg = yaml.safe_load((root / args.config).read_text())
    base_path = root / probe_cfg["base_config"]
    base_cfg = yaml.safe_load(base_path.read_text())
    results = []

    for batch_size in probe_cfg["candidate_batch_sizes"]:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = f"{base_cfg['name']}_batch_probe_bs{batch_size}"
        cfg["training"]["batch_size"] = int(batch_size)
        cfg["training"]["max_steps"] = int(probe_cfg.get("probe_steps", 10))
        cfg["training"]["eval_every"] = int(probe_cfg.get("probe_steps", 10))
        tmp_config = root / "experiments/multitask_distributional_bert/configs" / f"_tmp_batch_probe_bs{batch_size}.yaml"
        tmp_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd = [
            sys.executable,
            "-m",
            "src.training.train",
            "--config",
            str(tmp_config.relative_to(root)),
            "--seed",
            str(probe_cfg.get("seed", 1)),
            "--max-train-rows",
            str(probe_cfg.get("max_train_rows", 2000)),
        ]
        print("PROBE", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=root)
        ok = proc.returncode == 0
        results.append({"batch_size": batch_size, "ok": ok, "returncode": proc.returncode})
        if not ok:
            print(f"batch_size={batch_size} failed; stopping probe", flush=True)
            break

    out = root / "experiments/multitask_distributional_bert/metrics/batch_size_probe_results.yaml"
    out.write_text(yaml.safe_dump({"results": results}, sort_keys=False))
    good = [row["batch_size"] for row in results if row["ok"]]
    if good:
        print(f"largest_successful_batch_size={max(good)}")
    else:
        raise SystemExit("No candidate batch size succeeded.")


if __name__ == "__main__":
    main()

