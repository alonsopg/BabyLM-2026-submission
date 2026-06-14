#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_NAME CHECKPOINT_DIR [REVISION_NAME]" >&2
  exit 2
fi

cd "$(dirname "$0")/../../.."

run_name="$1"
checkpoint_dir="$(realpath "$2")"
revision_name="${3:-main}"
eval_root="/home/paperspace/babylm-hhm/resources/babylm-eval/strict"
model_link_root="$(pwd)/experiments/multitask_distributional_bert/eval_models"
model_link="${model_link_root}/${run_name}"

source experiments/multitask_distributional_bert/scripts/env.sh

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run official eval on CPU.")
print(torch.cuda.get_device_name(0))
PY

mkdir -p "$model_link_root"
rm -f "$model_link"
ln -s "$checkpoint_dir" "$model_link"

cd "$eval_root"
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name "$model_link" \
  --backend mlm \
  --task blimp \
  --data_path evaluation_data/fast_eval/blimp_fast \
  --save_predictions \
  --revision_name "$revision_name"
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name "$model_link" \
  --backend mlm \
  --task blimp \
  --data_path evaluation_data/fast_eval/supplement_fast \
  --save_predictions \
  --revision_name "$revision_name"
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name "$model_link" \
  --backend mlm \
  --task ewok \
  --data_path evaluation_data/fast_eval/ewok_fast \
  --save_predictions \
  --revision_name "$revision_name"
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name "$model_link" \
  --backend mlm \
  --task entity_tracking \
  --data_path evaluation_data/fast_eval/entity_tracking_fast \
  --save_predictions \
  --revision_name "$revision_name"
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.reading.run \
  --model_path_or_name "$model_link" \
  --backend mlm \
  --data_path evaluation_data/fast_eval/reading/reading_data.csv \
  --revision_name "$revision_name"
