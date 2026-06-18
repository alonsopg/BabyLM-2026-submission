# Smoke Test

Command:

```bash
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_relational_cloze_v1/configs/multitask_ewok_relational_cloze_v1.yaml \
  --seed 1 \
  --max-steps 300 \
  --name-suffix smoke300
```

Result: passed.

The 300-step smoke run used CUDA on the NVIDIA RTX A6000, sampled both cloze tasks,
logged the shared MLM-head cloze ranking metrics, and exported an MLM-compatible
checkpoint.

Task counts:

```json
{
  "mlm": 175,
  "rtd": 32,
  "connective": 30,
  "definiteness": 22,
  "collocation": 16,
  "substitution": 0,
  "function_word_recovery": 0,
  "agreement_prediction": 0,
  "grammar_minpair": 11,
  "conceptual_plausibility_choice": 0,
  "mlm_pair_ranking": 0,
  "semantic_cloze_ranking": 7,
  "relational_semantic_cloze": 7
}
```

Final smoke eval row:

```text
step=300
task=connective
val_mlm_loss=7.796850204467773
semantic_cloze_ranking_margin=1.6915591955184937
semantic_cloze_ranking_accuracy=0.90625
early_stop=False
```
