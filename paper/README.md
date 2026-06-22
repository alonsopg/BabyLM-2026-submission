# Semantic Cloze Multi-Task BabyLM Paper Draft

This folder contains a first ACL-style technical-description draft for the BabyLM 2026 Strict-Small submission.

Build:

```bash
cd paper
latexmk -pdf semantic-cloze-multitask-babylm.tex
```

The paper intentionally focuses on the multi-task semantic cloze experiment and does not include the older morph experiments.

Status:

- Draft created from the current submission artifact and experiment configs.
- Results tables are populated from the official eval reports and local fast-eval comparison.
- The known checkpoint provenance caveat is stated explicitly.
- Final polishing, bibliography expansion, and venue-specific OpenReview/ARR metadata are still needed before submission.
