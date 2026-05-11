# Heavy-Hitter Masking

Heavy-Hitter Masking (HHM) is a masking-only modification for BabyLM-style masked language modeling. The baseline and HHM use the same dataset, tokenizer, model, optimizer, batch size, training schedule, and total mask rate. HHM changes only which valid token positions are selected for prediction.

The default HHM mixture masks 15 percent of valid non-special tokens:

- 50 percent uniform random positions
- 25 percent repeated local entity-heavy-hitter positions
- 25 percent online error-heavy-hitter positions

Entity heavy hitters are extracted with a lightweight heuristic over decoded token strings. A candidate is alphabetic, not a stopword, at least two characters long, and either capitalized or content-word-like. Candidates are counted with a bounded Space-Saving sketch per sequence/window; positions belonging to repeated top entities are preferred for the entity share.

Error heavy hitters are tracked online with a Space-Saving sketch over target token strings. After warmup, the training loop computes per-token MLM losses for masked positions and updates the sketch with the top-loss masked tokens in each batch. Later batches preferentially mask tokens whose strings appear in the error sketch.

Fallbacks preserve fairness: if a sequence lacks enough entity or error positions, the collator fills the remaining budget with uniform random valid positions. Special and padding tokens are never masked.

The first paper milestone is implemented in `src/masking`: `RandomMaskingCollator`, `SpaceSavingSketch`, `EntityExtractor`, and `HeavyHitterMaskingCollator`. Online error tracking is also implemented through `update_error_sketch`.
