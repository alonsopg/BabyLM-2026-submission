# BabyLM Leaderboard Submission Status

Date: 2026-06-22

## Current State

The official Strict-Small predictions artifact is complete and locally validated:

- Artifact: `artifacts/all_full_preds_and_fast_scores_mlm.json`
- Track: `strict-small`
- Backend: `mlm`
- Hugging Face model: `alonsopg/babylm-2026-semantic-cloze-strict-small`
- Model revision: `main`
- Artifact SHA256: `ce458a181ef04dc47284da51dc7bb6e5b1902666a2d7e725cdee226bb23608d7`
- AoA results: populated, 152095 rows
- Checkpoint fast eval: populated for all 19 required `chck_*` revisions with no `null` entries

The public Hugging Face model repo is available and contains all required revisions:

- `main`
- `chck_1M` through `chck_9M`
- `chck_10M` through `chck_100M` by tens

## Leaderboard Form Payload

The exact prepared leaderboard fields are saved in:

- `leaderboard_form_payload.json`
- `LEADERBOARD_FORM_PAYLOAD.md`
- `leaderboard_other_hyperparameters.json`

These files are intended to make a browser submission reproducible if the Space API cannot be used.

## API Submission Attempt

The BabyLM leaderboard Space exposes a Gradio API endpoint:

```text
https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026
api_name="/submit_and_refresh"
```

Two API submission attempts were made from this machine on 2026-06-22:

1. Full payload with `leaderboard_other_hyperparameters.json`.
2. Simplified payload using only built-in dropdown values and no optional hyperparameter file.

Both attempts failed with a server-side Gradio exception:

```text
AppError: The upstream Gradio app has raised an exception but has not enabled verbose error reporting.
```

The current public leaderboard table was checked after the failures, and no matching row for
`semantic-cloze-multitask-bert-strict-small` or `alonsopg/babylm-2026-semantic-cloze-strict-small`
was visible in the embedded table data. This indicates the failed API calls did not visibly register.

## Manual Submission Result

Manual browser submission succeeded after changing the FLOPs field from the original decimal PFLOP estimate to the integer FLOP value required by the form:

```text
8000000000000000
```

The submitted row is visible on the `Strict-small` tab as:

```text
semantic-cloze-multitask-bert-strict-small
```

Live row observed after submission:

| Metric | Score |
|---|---:|
| Text Average | 33.0 |
| BLiMP | 55.36 |
| BLiMP Supplement | 53.29 |
| EWoK | 0.0 |
| Entity Tracking | 27.55 |
| COMPS | 50.63 |
| Reading | 5.94 |
| AoA | 11.25 |
| SuperGLUE | 60.0 |

## EWoK Display Issue

The live leaderboard currently displays EWoK as `0.0`, but the submitted JSON contains EWoK predictions and the local official eval report produced:

```text
EWoK average accuracy: 50.45
```

Local artifact checks:

- `ewok` is present in `artifacts/all_full_preds_and_fast_scores_mlm.json`.
- It contains 11 EWoK subtasks.
- Checkpoint `fast_eval_results.ewok` contains 19 entries with no `null` values.

This may be a leaderboard scoring/schema issue and should be checked with the organizers if the displayed `0.0` persists.

## Manual Submission Path

If a resubmission is needed, submit through the leaderboard browser UI:

```text
https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026
```

Use the values from `LEADERBOARD_FORM_PAYLOAD.md`.

Upload this results file:

```text
shared_task_submission/semantic_cloze/artifacts/all_full_preds_and_fast_scores_mlm.json
```

For Strict-Small, the multilingual predictions file is not required.

## Remaining Items

- Fill the BabyLM hyperparameter/details form using `LEADERBOARD_FORM_PAYLOAD.md`.
- Investigate or report the leaderboard EWoK `0.0` display issue.
- Prepare and submit the paper through the announced OpenReview/ARR path.

The paper is intentionally left as the final item.
