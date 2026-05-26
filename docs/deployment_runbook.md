# Deployment Runbook

Use this after a fine-tuned model has been pushed to Hugging Face Hub and approved for
shadow-mode traffic.

## Prerequisites

- Fine-tuned model repo or merged checkpoint.
- GPU host with vLLM installed.
- OTT backend can reach the vLLM OpenAI-compatible endpoint.
- Logging sink for decisions, latency, model version, prompt version, and raw model output.

## Start vLLM

```bash
vllm serve your-org/qwen3.5-2b-comment-filtering \
  --max-model-len 1024
```

Keep the first deployment in shadow mode. Do not wire the response to user blocking until
offline evaluation and human review have approved enforcement.

## Smoke Test

```bash
PYTHONPATH=src python3 scripts/smoke_vllm.py \
  --base-url http://localhost:8000/v1 \
  --model your-org/qwen3.5-2b-comment-filtering
```

Expected result: valid moderation JSON with `action` set to `shadow_log`.

## Rollback

- Route moderation calls back to the previous model version or disable the moderation call.
- Keep logging enabled if the model is disabled so the team can compare missed events.
- Mark the failed model version as blocked in deployment notes.
