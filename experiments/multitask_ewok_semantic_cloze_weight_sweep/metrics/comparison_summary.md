# Semantic Cloze Weight Sweep Summary

## Goal

Sweep only the semantic cloze auxiliary weight while preserving the model, tokenizer,
corpus, training schedule, calibration, and official fast-evaluation pipeline from the
successful `multitask_ewok_semantic_cloze_v1` run.

## Variants

| Variant | MLM | RTD | Conn. | Def. | Colloc. | Grammar | Semantic cloze |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `semantic_cloze_w025` | 0.625 | 0.100 | 0.075 | 0.075 | 0.050 | 0.050 | 0.025 |
| `semantic_cloze_w075` | 0.575 | 0.100 | 0.075 | 0.075 | 0.050 | 0.050 | 0.075 |
| `semantic_cloze_w100` | 0.550 | 0.100 | 0.075 | 0.075 | 0.050 | 0.050 | 0.100 |

All runs used the existing synthetic semantic cloze data and did not use official EWoK
or BLiMP examples, external datasets, or classifier heads.

## GPU And Smoke Checks

CUDA was available through the BabyLM training environment:

```text
torch 2.5.1+cu121
True NVIDIA RTX A6000
```

All smoke runs completed at 300 steps, saved checkpoints, sampled MLM, grammar-minpair,
and semantic-cloze batches, and logged semantic margin/accuracy.

| Smoke run | MLM steps | Grammar steps | Semantic steps | Semantic acc. | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| `semantic_cloze_w025_smoke300` | 177 | 20 | 8 | 0.9375 | ok |
| `semantic_cloze_w075_smoke300` | 164 | 17 | 25 | 0.9271 | ok |
| `semantic_cloze_w100_smoke300` | 162 | 16 | 34 | 0.9167 | ok |

## Full Training

| Run | Stop step | Early stop | Best val step | Best val MLM loss | Semantic steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `semantic_cloze_w025` | 9500 | yes | 7000 | 6.811737 | 202 |
| `semantic_cloze_w075` | 9500 | yes | 5000 | 6.898015 | 604 |
| `semantic_cloze_w100` | 10000 | no | 9000 | 6.830217 | 857 |

## Official Fast-Eval Results

| System | BLiMP | Sup. | EWoK | Entity | Eye | SPR | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLM baseline | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 | baseline |
| BLiMP-pair repair best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 | old EWoK best before semantic cloze |
| Semantic cloze 5 best | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 | old semantic best |
| Semantic cloze 5 final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 | current EWoK best |
| Semantic cloze 2.5 best | 53.93 | 52.80 | 50.82 | 29.34 | 8.81 | 3.10 | new sweep |
| Semantic cloze 2.5 final | 54.34 | 54.80 | 50.73 | 22.79 | 8.82 | 3.06 | new sweep |
| Semantic cloze 7.5 best | 53.57 | 52.80 | 51.00 | 40.48 | 7.85 | 3.13 | new sweep |
| Semantic cloze 7.5 final | 55.17 | 55.60 | 50.82 | 25.13 | 8.12 | 3.10 | new sweep |
| Semantic cloze 10 best | 54.72 | 52.40 | 51.27 | 34.77 | 7.92 | 3.25 | new sweep |
| Semantic cloze 10 final | 55.09 | 52.80 | 51.64 | 33.59 | 8.01 | 3.39 | new sweep |

## Best By Metric

| Metric | Best system | Score |
| --- | --- | ---: |
| EWoK | Semantic cloze 5 final | 52.00 |
| Entity | Semantic cloze 7.5 best | 40.48 |
| Eye | Semantic cloze 2.5 final | 8.82 |
| SPR | Semantic cloze 5 final | 3.71 |
| Sweep-only EWoK | Semantic cloze 10 final | 51.64 |
| Sweep-only balanced | Semantic cloze 10 final | EWoK 51.64, Entity 33.59, Eye 8.01, SPR 3.39 |

## Interpretation

The sweep did not improve over the current EWoK best. The original 5% semantic cloze
final checkpoint remains the strongest EWoK model at 52.00.

The sweep does clarify the trade-off:

- Lower semantic cloze weight did not recover EWoK. The 2.5% final checkpoint reached
  only 50.73 EWoK, although it produced the best Eye score.
- Slightly higher semantic cloze weight did not improve EWoK. The 7.5% best checkpoint
  matched the old BLiMP-pair EWoK score of 51.00 and gave the best Entity score, 40.48.
- Higher semantic cloze weight gave the best sweep-only EWoK, 51.64, and the best
  sweep-only balance, but still did not beat the 5% final checkpoint.

## Decision

Keep `multitask_ewok_semantic_cloze_v1_final` as the EWoK-focused checkpoint. It remains
the only run above 52.00 and is still +1.45 over the MLM baseline.

Use `semantic_cloze_w100_final` only if the priority is a more balanced sweep-only
checkpoint with EWoK above 51.5 and Entity above 30. It misses the SPR threshold from
the proposed balanced criterion, so it is not strictly balanced by the full stated rule.

Use `semantic_cloze_w075_best` as evidence that entity tracking can be recovered by
changing the weight, but not as the primary result because its EWoK is only 51.00.

## Conclusion

The 5% semantic cloze mixture appears to be the best setting found so far for EWoK.
The EWoK improvement is real but narrow: moving semantic cloze weight down or up does
not improve the target score.
