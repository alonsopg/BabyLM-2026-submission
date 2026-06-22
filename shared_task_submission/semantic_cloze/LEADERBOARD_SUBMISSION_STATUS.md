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

## Manual Submission Path

If the API remains unavailable, submit through the leaderboard browser UI:

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

- Submit the results through the leaderboard UI once the Space accepts the form.
- Fill the BabyLM hyperparameter/details form using `LEADERBOARD_FORM_PAYLOAD.md`.
- Prepare and submit the paper through the announced OpenReview/ARR path.

The paper is intentionally left as the final item.
