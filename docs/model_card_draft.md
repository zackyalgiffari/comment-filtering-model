# Model Card Draft: Qwen3.5-2B Comment Filtering

## Model Details

- Base checkpoint for fine-tuning: `Qwen/Qwen3.5-2B`
- Base reference: `Qwen/Qwen3.5-2B-Base`
- License: Apache-2.0
- Task: Indonesian/English OTT live-chat moderation
- Output: compact JSON moderation decision

## Intended Use

This model is intended for shadow-mode trust-and-safety analysis of live-stream comments.
It should not be used as the sole basis for user blocking, account penalties, or appeals
until internal OTT chat data has been labeled and evaluated.

## Taxonomy

Labels:

- `safe`
- `hate_or_harassment`
- `sexual`
- `violence_or_threat`
- `spam_or_scam`
- `profanity`
- `self_harm`

Severity values: `low`, `medium`, `high`.

Action in v0: `shadow_log`.

## Training Data

The initial repository includes only bootstrap sample data for pipeline validation. A
production model should be trained and evaluated with:

- Public English moderation datasets for bootstrap coverage.
- Curated Indonesian and mixed Indonesian/English examples.
- Real OTT chat samples labeled by human reviewers.

## Limitations

- Public moderation data may not match OTT live-chat slang or local language patterns.
- The model may over-flag profanity used casually in sports or entertainment chat.
- Self-harm and threat labels require careful human review before enforcement.
- Confidence values are model outputs and must be calibrated against validation data.

## Evaluation

Use `scripts/evaluate_predictions.py` and prioritize high precision for any label that
could later trigger enforcement. Report metrics separately for Indonesian, English, and
mixed-language comments.
