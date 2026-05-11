# Paper Framing

## Research Question

Can bounded-memory heavy-hitter sketches improve small-data language model pretraining by reallocating a fixed masked-language-modeling budget toward repeated discourse entities and model-specific hard examples?

## Core Claim

Heavy-Hitter Masking is a lightweight masking-only modification for MLM pretraining. It changes the distribution of masked positions while holding the corpus, tokenizer, model, optimizer, training schedule, and total mask rate fixed.

## Main Comparisons

- Baseline Random MLM
- HHM Entity+Error
- HH-Entity only
- HH-Error only
- Random sketch control
- Frequency-only control

## Main Analyses

- Validation MLM loss
- Official BabyLM score
- BLiMP-style grammatical generalization
- Entity tracking
- Mask distribution diagnostics
- Training time and overhead
