# Grammar Minpair Sample Inspection

## Generated Counts

```json
{
  "grammar_minpair": {
    "num_examples": 30820,
    "phenomena": {
      "subject_verb_agreement": 17500,
      "determiner_noun_agreement": 7040,
      "auxiliary_agreement": 5976,
      "npi_licensing": 220,
      "reflexive_binding": 84
    }
  }
}
```

## First Corpus-Based Attempt

The first generator used corpus corruptions for subject-verb, determiner-noun, and auxiliary agreement. It was rejected before training because manual inspection found too many noisy determiner examples:

- `that` used as a complementizer rather than a determiner.
- deictic/pronominal `that` cases such as `that way` and `that later`.
- occasional false noun cues before `is/was`.

This exceeded the precision target for a BLiMP-style repair objective.

## Final Generator Used For Training

The final generator uses high-precision templates for all five grammar-minpair phenomena:

- subject-verb agreement
- determiner-noun agreement
- auxiliary agreement
- NPI licensing
- simple reflexive binding

The other multi-task examples remain generated from the same BabyLM corpus by the existing generator functions.

## 50-Example Sample Check

Sample path:

```text
experiments/multitask_repair_blmpair_v1/data/generated/samples/grammar_minpair_50_examples.jsonl
```

Manual inspection result:

- Good sentence is consistently better than bad sentence.
- Bad sentence is minimally corrupted.
- Labels match randomized order.
- Target order is balanced in the inspected sample: 27 first-sentence-good, 23 second-sentence-good.
- Phenomena present in the inspected sample: subject-verb agreement, determiner-noun agreement, auxiliary agreement, NPI licensing.
- Reflexive binding is present in the full generated file but only one example appeared in the first 50-example sample because that pool is intentionally small.

Known caveat:

- Templates are sometimes semantically plain or odd, such as inanimate plural subjects with progressive predicates. They are still grammatical minimal pairs, and this is preferable to training on noisy corpus-derived complementizer/pronoun corruptions.

Decision:

- Proceed to GPU smoke test.
