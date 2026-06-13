#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT_DIR [BABYLM_EVAL_DIR]" >&2
  exit 2
fi

cd "$(dirname "$0")/../../.."

source experiments/multitask_distributional_bert/scripts/env.sh

checkpoint="$1"
babylm_eval="${2:-resources/babylm-eval}"

python -m src.evaluation.evaluate \
  --checkpoint "$checkpoint" \
  --babylm-eval "$babylm_eval"
