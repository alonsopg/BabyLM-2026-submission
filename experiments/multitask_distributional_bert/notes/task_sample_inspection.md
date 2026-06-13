# Task Sample Inspection

Generated on a 5,000-row pilot slice of `BabyLM-community/BabyLM-2026-Strict-Small`.

## Decision

Use these tasks for the first multi-task smoke/full run:

- MLM
- RTD
- connective prediction
- definiteness prediction

Disable these tasks for the first run:

- substitution compatibility
- collocation naturalness

## Rationale

The substitution samples are too noisy under the first heuristic generator. Examples include local-context positives such as `we c there` with candidate `got`, which is not a high-confidence compatibility example.

The collocation samples are also noisy. Examples include `lot number` vs `there number` and `hundred pounds` vs `going pounds`; these are detectable as odd strings but are not clean collocation-naturalness contrasts.

Following the experiment plan, their probability is set to 0 for the first run and redistributed to MLM:

```text
60 MLM, 20 RTD, 10 connective, 10 definiteness
```

The generated files are retained for inspection but not sampled by `configs/multitask_bert.yaml`.
