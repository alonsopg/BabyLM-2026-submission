# Current Situation

## Main Finding

The current BabyLM multi-task experiments show a stable trade-off:

- MLM-only pretraining is much better on BLiMP.
- Multi-task pretraining is much better on Entity Tracking and reading-prediction metrics.
- The targeted repair attempts did not recover BLiMP.

## Best Systems By Metric

- BLiMP: MLM-only baseline, 64.33.
- Entity Tracking: syntax repair best checkpoint, 34.92.
- Eye: syntax repair final checkpoint, 8.28.
- SPR: syntax repair final checkpoint, 3.63.

## Failed Repair Attempts

Syntax repair:

- Added function-word recovery and agreement prediction as MLM-head tasks.
- BLiMP improved only from 54.80 to 55.51.
- Entity/Reading remained strong.

BLiMP-pair repair:

- Replaced weak syntax MLM tasks with a direct grammatical minimal-pair preference task.
- First corpus-corruption generator was rejected as too noisy.
- Final generator used high-precision templates.
- BLiMP stayed at 54.80/54.81.
- Entity dropped relative to previous multi-task systems.

## Current Interpretation

The BLiMP degradation is probably not caused simply by the absence of explicit syntax supervision. It appears more likely to arise from broader multi-task interference, reduced effective MLM exposure, calibration limits, or a mismatch between the auxiliary objectives and BLiMP's sentence-level acceptability distribution.

## Paper Angle

This should be written as a diagnostic paper rather than a method-wins paper:

> Multi-task objectives can improve discourse-like and reading-prediction behavior in compact BabyLM-scale BERT models, but these gains come with a persistent loss in grammatical minimal-pair performance that simple syntax-oriented repairs do not fix.
