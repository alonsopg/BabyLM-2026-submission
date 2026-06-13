# Official Fast Eval and Light-Aux Follow-Up

## What was missing

The final eval-dataset table was missing. The local `src/evaluation/evaluate.py` script only writes an `evaluation_note.json`; it does not run the official BabyLM benchmark datasets.

The official strict eval bundle is available locally at:

```text
/home/paperspace/babylm-hhm/resources/babylm-eval/strict
```

## What was added

Two scripts were added:

```text
scripts/run_official_fast_eval.sh
scripts/collect_official_eval_table.py
```

The first runs the official BabyLM strict fast zero-shot/reading evaluation with CUDA. The second collects the report files into:

```text
metrics/official_fast_eval_table.csv
metrics/official_fast_eval_table.md
paper/tables/multitask_distributional_eval_fast.tex
```

## Official fast eval results

```text
normal_bert_mlm_final          BLiMP 64.33  Sup. 54.00  EWoK 50.55  Entity 15.56  Eye 2.47  SPR 2.42
multitask_bert_calibrated_best BLiMP 54.80  Sup. 54.80  EWoK 50.27  Entity 34.52  Eye 7.46  SPR 3.19
multitask_bert_light_aux_best  BLiMP 54.59  Sup. 52.00  EWoK 50.09  Entity 29.60  Eye 8.07  SPR 3.34
```

## Light auxiliary run

A lighter auxiliary schedule was added:

```text
configs/multitask_bert_light_aux.yaml
```

Mixture:

```text
MLM 0.85
RTD 0.10
connective 0.025
definiteness 0.025
```

This run used CUDA, `max_steps=6000`, and early stopping enabled from step 3000 with patience 2. It finished at the max step because step 5000 improved before the last two evals.

Best local MLM validation:

```text
step 5000
val_mlm_loss = 6.804625988006592
```

This is the best multi-task validation loss so far, but still far behind the normal MLM baseline:

```text
normal_bert best val_loss = 3.0248489379882812
```

## Interpretation

The multi-task objective is not improving general grammatical preference under BLiMP. However, it consistently improves Entity Tracking and Reading fast scores over the normal MLM baseline. The result is not a clean win, but it is a real trade-off: auxiliary tasks seem to bias the model toward discourse/reference and processing-alignment signals while damaging core MLM/BLiMP behavior.

For shorter future runs, the most sensible optimization is staged triage:

```text
1. run 4k-6k steps
2. enable early stopping from step 3k
3. run official fast eval only for checkpoints that improve local validation or target metrics
4. reserve full 10k/full official eval for promising variants
```

Changing optimizer or mixed precision may reduce wall-clock time, but it also changes the comparison. Staged runs with early stopping are the safer speedup for this experiment.
