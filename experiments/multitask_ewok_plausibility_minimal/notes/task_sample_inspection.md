# Task Sample Inspection

Generated on 2026-06-14 for `multitask_ewok_plausibility_minimal`.

## Counts

- `conceptual_plausibility_choice`: 50,000 examples
- Domain mix:
  - selectional_preference: 12,000
  - affordance: 10,000
  - spatial_containment: 6,500
  - part_whole: 6,000
  - animate_agency: 15,500

The remaining enabled tasks reuse the same generated BabyLM-derived data pipeline as
the calibrated multi-task runs: MLM, RTD, connective choice, definiteness, and
collocation.

## Inspection Notes

The first 50 conceptual-plausibility examples were manually inspected after template
cleanup. Choice order is randomized, with labels using `0 = Context A is more
plausible` and `1 = Context B is more plausible`. The inspected sample contained both
label values and all five conceptual domains.

The final cleanup pass fixed article choice (`a useful`, `an old`), removed noisy
selectional-preference location modifiers, moved adverbs before verbs for more natural
word order, and separated support-object modifiers from containment-object modifiers.

The generated examples are intentionally high-precision synthetic contrasts. They do
not use EWoK evaluation items or EWoK training data. The manifest records
`ewok_data_used: false`.
