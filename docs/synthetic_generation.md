# Synthetic Moderation Data Prompt Pack

Use this prompt pack with any LLM provider to create bootstrap examples. Generated data
must be reviewed before import and must pass `scripts/validate_synthetic_batch.py`.

Do not generate real personal data, usernames, phone numbers, addresses, or URLs. Use
fictional, generic live-chat text only.

## System Prompt

You create fictional OTT live-stream chat moderation examples for a safety dataset.
Return JSONL only. Each line must be a JSON object matching the requested schema.
Do not include real names, real accounts, phone numbers, addresses, or real URLs.
Use Indonesian, English, and mixed Indonesian/English live-chat style.

## Row Schema

Each JSONL row:

```json
{
  "synthetic_id": "batch001-0001",
  "text": "fictional live-chat comment",
  "language": "id|en|mixed",
  "decision": {
    "flagged": true,
    "labels": ["spam_or_scam"],
    "severity": "low|medium|high",
    "confidence": 0.85,
    "action": "shadow_log"
  },
  "review": {
    "reviewed": true,
    "reviewer": "initials-or-team",
    "notes": "short review note"
  }
}
```

## Generation Request

Create 100 fictional OTT live-chat comments as JSONL.

Coverage requirements:

- 20 safe comments about sports, concerts, movies, or stream quality.
- 15 hate_or_harassment comments without using protected-class slurs.
- 15 profanity comments where the label is profanity rather than hate.
- 15 spam_or_scam comments using fake promotions and `[URL]`.
- 10 violence_or_threat comments without graphic detail.
- 10 sexual comments without explicit sexual detail.
- 10 self_harm comments phrased as concerning user distress.
- 5 mixed-label unsafe comments using two unsafe labels only when both are clearly present.

Language requirements:

- At least 40 Indonesian rows.
- At least 30 English rows.
- At least 20 mixed Indonesian/English rows.

Review requirements:

- Set `review.reviewed` to `false` during generation.
- A human reviewer must set it to `true` before import.
- Do not output any markdown wrapper.
