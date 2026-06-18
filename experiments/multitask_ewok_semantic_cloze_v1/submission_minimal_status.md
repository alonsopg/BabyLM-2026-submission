# Minimal Submission Status

## Final System Choice

Use `multitask_ewok_semantic_cloze_v1_final` as the minimal BabyLM multi-task
submission candidate.

Reason: it is the strongest EWoK-focused checkpoint in the isolated multi-task series.

| System | BLiMP | Supplement | EWoK | Entity | Eye | SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLM baseline | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| Semantic cloze best | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 |
| Semantic cloze final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |
| Relational cloze best | 54.78 | 53.20 | 51.00 | 40.68 | 7.69 | 3.29 |
| Relational cloze final | 54.83 | 52.00 | 50.82 | 26.80 | 8.09 | 3.26 |

## Checkpoint Verification

Verified locally with:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

path = "experiments/multitask_ewok_semantic_cloze_v1/checkpoints/multitask_ewok_semantic_cloze_v1/seed_1/mlm_compatible_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForMaskedLM.from_pretrained(path)
```

Result:

```text
tokenizer: BertTokenizer
model: BertForMaskedLM
parameters: 11,201,338
```

## Hugging Face Model

Public model:

```text
https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-strict-small
```

Hub commit:

```text
4530b3e6be2c264f339a59a2223c35bf3e12798a
```

Verified from the public repo with:

```python
repo = "alonsopg/babylm-2026-semantic-cloze-strict-small"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForMaskedLM.from_pretrained(repo)
```

Uploaded files:

```text
README.md
config.json
model.safetensors
results_summary.json
tokenizer.json
tokenizer_config.json
training_config.yaml
```

## Completed Minimal Submission Items

- Final system selected.
- Final checkpoint verified as Hugging Face `BertForMaskedLM`.
- Public Hugging Face model uploaded.
- Model card added with training setup, task mixture, results, and caveats.

## Remaining Items

- Run the official BabyLM evaluation pipeline on the public model.
- Generate the official leaderboard/submission artifact.
- Write the focused multi-task semantic cloze paper.
