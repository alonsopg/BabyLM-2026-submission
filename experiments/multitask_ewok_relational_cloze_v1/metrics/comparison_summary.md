# Relational Semantic Cloze Comparison

## Question

Does adding relational semantic cloze examples improve EWoK beyond the current best
semantic cloze result?

## Result

No. The relational cloze variant trained and evaluated correctly, but it did not beat
`multitask_ewok_semantic_cloze_v1_final` on EWoK.

| System | BLiMP | Supplement | EWoK | Entity | Eye | SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLM baseline | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| BLiMP-pair repair best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 |
| Semantic cloze 5% best | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 |
| Semantic cloze 5% final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |
| Relational cloze best | 54.78 | 53.20 | 51.00 | 40.68 | 7.69 | 3.29 |
| Relational cloze final | 54.83 | 52.00 | 50.82 | 26.80 | 8.09 | 3.26 |

## Interpretation

The hypothesis is not supported in this form. Splitting the 5% cloze pressure into
2.5% original semantic cloze plus 2.5% relational semantic cloze reduced EWoK relative
to the 5% semantic cloze final checkpoint:

```text
semantic cloze final:     EWoK 52.00
relational cloze best:    EWoK 51.00
relational cloze final:   EWoK 50.82
```

The useful positive signal is Entity Tracking. The relational best checkpoint reached
Entity 40.68, higher than the semantic cloze best at 36.61 and much higher than the
semantic cloze final at 27.44. That suggests the relational examples may help some
structured tracking behavior, but the benefit did not transfer to EWoK.

## Training Notes

The run used the existing MLM-head pairwise ranking loss, with no classifier head and
no CLS pooling. The relational task was added as a separate task name,
`relational_semantic_cloze`, but shares the same single-mask good-vs-bad token scoring
path as `semantic_cloze_ranking`.

The configured active task probabilities were:

```text
mlm                         0.600
rtd                         0.100
connective                  0.075
definiteness                0.075
collocation                 0.050
grammar_minpair             0.050
semantic_cloze_ranking      0.025
relational_semantic_cloze   0.025
```

Early stopping triggered at step 9500. The validation best checkpoint was selected
from step 5000.

Final full-run task counts:

```json
{
  "mlm": 6316,
  "rtd": 783,
  "connective": 564,
  "definiteness": 618,
  "collocation": 426,
  "grammar_minpair": 395,
  "semantic_cloze_ranking": 199,
  "relational_semantic_cloze": 199
}
```

## Recommendation

Keep `multitask_ewok_semantic_cloze_v1_final` as the best EWoK-focused checkpoint.
Treat relational cloze as a negative EWoK result and a possible lead only if the next
objective is improving Entity Tracking rather than EWoK.
