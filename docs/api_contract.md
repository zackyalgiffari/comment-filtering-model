# Shadow-Mode Moderation API Contract

The OTT backend should call moderation before displaying or storing a live chat message
for trust-and-safety analytics. The first rollout is shadow mode, so the product action
remains logging only.

## Request

`POST /moderate-comment`

```json
{
  "comment_id": "chat-123",
  "stream_id": "stream-456",
  "user_id_hash": "sha256-user-id",
  "text": "Komentar live chat",
  "language": "id",
  "timestamp": "2026-05-26T10:00:00Z"
}
```

## Response

```json
{
  "comment_id": "chat-123",
  "model_version": "qwen3.5-2b-comment-filtering-v0",
  "latency_ms": 84,
  "decision": {
    "flagged": true,
    "labels": ["spam_or_scam"],
    "severity": "medium",
    "confidence": 0.89,
    "action": "shadow_log"
  }
}
```

## Serving Notes

- Use vLLM OpenAI-compatible chat completions.
- Set temperature to `0.0`.
- Keep `max_tokens` small because the target output is compact JSON.
- Store model version, prompt version, raw output, parsed decision, and latency for audits.
- Do not block users until shadow-mode metrics and human review approve enforcement.
