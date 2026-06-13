# Multitask Repair Syntax v1 Summary

## Question

Can a minimal syntax-oriented repair to the multi-task BERT setup recover BLiMP performance while preserving the Entity Tracking and reading gains from auxiliary supervision?

## Result

Partial success, but not a BLiMP repair.

The repair run trains correctly on GPU, samples the intended task mixture, and improves the auxiliary-side metrics. It does not recover BLiMP close to the MLM-only baseline.

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| multitask_bert_calibrated_best | 54.80 | 54.80 | 50.27 | 34.52 | 7.46 | 3.19 |
| multitask_bert_light_aux_best | 54.59 | 52.00 | 50.09 | 29.60 | 8.07 | 3.34 |
| multitask_repair_syntax_v1_best | 55.28 | 52.80 | 50.09 | 34.92 | 8.13 | 3.50 |
| multitask_repair_syntax_v1_final | 55.51 | 52.40 | 49.55 | 32.76 | 8.28 | 3.63 |

## Interpretation

- BLiMP recovery is minimal: the best observed repair checkpoint reaches 55.51, only +0.71 over the calibrated multi-task baseline and still -8.82 behind MLM-only.
- Entity Tracking remains much better than MLM-only: 34.92 for the validation-best repair checkpoint versus 15.56 for MLM-only.
- Reading improves beyond previous multi-task runs: final checkpoint reaches 8.28 Eye and 3.63 SPR.
- The final checkpoint is best for BLiMP and reading, while the validation-best checkpoint is better for Entity Tracking and EWoK.

## Training Notes

- Run: `multitask_repair_syntax_v1`, seed 1.
- GPU path verified through PyTorch on `NVIDIA RTX A6000`.
- `nvidia-smi` failed with an NVML driver/library mismatch, but PyTorch CUDA was available and the trainer refuses CPU execution.
- Full training completed 10000 steps.
- Best validation MLM checkpoint occurred at step 9000 with validation MLM loss 6.8174.
- Early stopping was active from step 9000 and reached `bad_evals=2` at the natural final step.

## Conclusion

The syntax repair works operationally and improves the auxiliary behavior, but it does not solve the BLiMP regression. The current evidence says the multi-task setup is learning something useful for Entity Tracking and reading, while still disrupting the grammatical preference behavior measured by BLiMP.
