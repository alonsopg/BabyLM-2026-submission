# MLM Pair-Ranking Sample Inspection

Generated on 2026-06-14 from:

`experiments/multitask_ewok_plausibility_minimal/data/generated/conceptual_plausibility_choice.jsonl`

## Counts

- `mlm_pair_ranking`: 50,000 examples
- Domain mix:
  - affordance: 10,000
  - animate_agency: 15,500
  - part_whole: 6,000
  - selectional_preference: 12,000
  - spatial_containment: 6,500

The manifest records `ewok_data_used: false` and `blimp_data_used: false`.

## Inspection

The first 50 converted examples were inspected. Each row contains:

- `good_sequence`: `Target: ... Context: ...`
- `bad_sequence`: `Target: ... Context: ...`
- `metadata.scoring_tokens_good`
- `metadata.scoring_tokens_bad`

The inspected rows had no empty scoring-token lists. Scoring tokens come from the
context sentence rather than the shared target sentence. Good contexts are the same
plausible contexts used by the previous classifier-head conceptual-plausibility run;
bad contexts are the corresponding implausible contexts.

This conversion does not regenerate conceptual examples and does not use official EWoK
or BLiMP evaluation data.
