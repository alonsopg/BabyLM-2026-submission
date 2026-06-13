# Task Sample Inspection

Generated files:

```text
experiments/multitask_repair_syntax_v1/data/generated/
```

## Counts

```text
mlm: 48889
rtd: 48889
connective: 7777
definiteness: 19007
collocation: 8570
function_word_recovery: 19180
agreement_prediction: 10285
substitution: 40990, disabled with probability 0.0
```

## Function-word recovery

The examples are usable for a first repair run. They mask syntactically important items such as:

```text
who
to
in
with
from
are
was
have
```

The task uses the MLM head and labels only the `[MASK]` position.

## Agreement prediction

The first generator pass admitted some noisy demonstrative examples, so the filters were tightened before training. The regenerated examples are mostly high precision for:

```text
has/have
is/are/was/were
this/these/that/those when used determiner-like
```

The corpus itself contains conversational fragments, so a few examples remain informal or not fully sentence-like. This is acceptable for the minimal repair run because the task share is only 5 percent and labels are high-confidence function/agreement tokens.

The task uses the MLM head and labels only the `[MASK]` position.

## Collocation

Collocation remains noisy, as already observed in the previous experiment. It is included at 5 percent because the repair plan requested preserving it as a possible contributor to the Entity/Reading gains. Substitution remains disabled.

## Decision

Proceed to GPU smoke test, then full `multitask_repair_syntax_v1` if the smoke run has finite loss, task sampling works, and checkpoints load as MLM-compatible.
