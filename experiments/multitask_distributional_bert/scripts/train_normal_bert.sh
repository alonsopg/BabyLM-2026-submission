#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

experiments/multitask_distributional_bert/scripts/preflight_gpu.sh
source experiments/multitask_distributional_bert/scripts/env.sh

python -m src.training.train \
  --config experiments/multitask_distributional_bert/configs/normal_bert_mlm.yaml \
  "$@"
