# Inspection notes

## Branch

- Branch: `ewok-plausibility-minimal`
- Base commit inspected: `e57124c Add semantic cloze ranking EWoK experiment`

## Base semantic cloze run

- Base experiment folder: `experiments/multitask_ewok_semantic_cloze_v1/`
- Base config: `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml`
- Previous semantic cloze weight: 0.05
- Previous final EWoK: 52.00
- Previous best-validation EWoK: 50.73

## Semantic cloze data path

- `experiments/multitask_ewok_semantic_cloze_v1/data/semantic_cloze_ranking.jsonl`
- Manifest says the data is synthetic and does not use official EWoK or BLiMP data.

## Base config path

- `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml`

## Existing GPU command

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml \
  --seed 1
```

CUDA check through the BabyLM env:

```text
torch 2.5.1+cu121
True NVIDIA RTX A6000
```

## Existing calibration command

Final MLM-only calibration is configured inside training:

```yaml
training:
  calibration_mlm_steps: 2000
```

No separate calibration script is used for this experiment family.

## Existing evaluation command

```bash
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  <run_name> <checkpoint_dir> main
```

The evaluation script refuses to run if CUDA is unavailable.

## Files changed

- Added sweep configs for 2.5%, 7.5%, and 10.0% semantic cloze.
- Added this inspection note.
- Planned lightweight outputs: run records, official fast-eval tables, comparison summary JSON/Markdown.
- Checkpoints and eval-model symlinks remain ignored.
