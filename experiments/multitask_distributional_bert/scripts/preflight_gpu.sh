#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

source experiments/multitask_distributional_bert/scripts/env.sh

echo "== nvidia-smi =="
nvidia-smi

echo
echo "== torch cuda check =="
python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to launch training on CPU.")
print("gpu:", torch.cuda.get_device_name(0))
print("cuda_device_count:", torch.cuda.device_count())
PY
