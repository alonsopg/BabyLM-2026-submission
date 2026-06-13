# Multitask Repair BLiMP-Pair v1 Summary

## Question

Does replacing the weak function-word/agreement MLM repair tasks with a direct BLiMP-style grammatical minimal-pair preference task recover BLiMP while preserving Entity Tracking and reading gains?

## Result

No. The experiment trained and evaluated correctly, but the BLiMP-pair objective did not recover BLiMP.

| System | BLiMP | Supplement | EWoK | Entity | Eye | SPR | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 | MLM-only baseline |
| multitask_bert_calibrated_best | 54.80 | 54.80 | 50.27 | 34.52 | 7.46 | 3.19 | previous calibrated multi-task |
| multitask_repair_syntax_v1_final | 55.51 | 52.40 | 49.55 | 32.76 | 8.28 | 3.63 | previous weak syntax repair |
| multitask_repair_blmpair_v1_best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 | validation-best checkpoint |
| multitask_repair_blmpair_v1_final | 54.81 | 52.40 | 50.73 | 25.99 | 7.96 | 3.11 | final checkpoint |

## Deltas

Using `multitask_repair_blmpair_v1_final`:

| Comparison | BLiMP | Entity | Eye | SPR |
| --- | ---: | ---: | ---: | ---: |
| minus MLM-only baseline | -9.52 | +10.43 | +5.49 | +0.69 |
| minus prior calibrated multi-task | +0.01 | -8.53 | +0.50 | -0.08 |
| minus repair syntax final | -0.70 | -6.77 | -0.32 | -0.52 |

Using `multitask_repair_blmpair_v1_best`:

| Comparison | BLiMP | Entity | Eye | SPR |
| --- | ---: | ---: | ---: | ---: |
| minus MLM-only baseline | -9.53 | +11.98 | +5.55 | +0.63 |
| minus prior calibrated multi-task | +0.00 | -6.98 | +0.56 | -0.14 |
| minus repair syntax final | -0.71 | -5.22 | -0.26 | -0.58 |

## Training Notes

- Run: `multitask_repair_blmpair_v1`, seed 1.
- Full training completed 10000 steps.
- Final 2000 steps were MLM-only calibration.
- Validation-best checkpoint: step 9500, validation MLM loss 6.7981.
- Final checkpoint: step 10000, validation MLM loss 6.8184.
- Mixed-training task counts at step 8000 matched the target mixture well:
  - MLM 4817
  - RTD 749
  - connective 605
  - definiteness 609
  - collocation 406
  - grammar_minpair 814
- `grammar_minpair` loss became very easy by mid-training, reaching near zero around step 6500.

## Generator Notes

The first corpus-corruption generator was rejected before training because the determiner-number examples were too noisy, mostly due to `that` as complementizer or pronoun. The final training generator used high-precision templates for:

- subject-verb agreement
- determiner-noun agreement
- auxiliary agreement
- NPI licensing
- simple reflexive binding

Final `grammar_minpair` count: 30820 examples.

## Interpretation

The direct BLiMP-style pair objective was operationally successful but scientifically unsuccessful for the target. It did not improve BLiMP beyond the older calibrated multi-task baseline and performed worse than the previous syntax repair final checkpoint.

This suggests the BLiMP degradation is probably not fixed by adding an easy template-based grammatical preference task. The degradation may instead come from broader multi-task interference, reduced effective MLM exposure, the auxiliary task mix, or a mismatch between template pair discrimination and the BLiMP evaluation distribution.

## Decision

Do not keep `multitask_repair_blmpair_v1` as the main repaired model.

Recommended next follow-up, if continuing: increase MLM share while keeping only a small grammar-pair dose, for example 70 MLM, 7.5 RTD, 5 connective, 5 definiteness, 2.5 collocation, 10 grammar_minpair, with the same MLM calibration.
