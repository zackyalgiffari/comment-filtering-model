# Shadow-Mode Rollout Checklist

## Before Training

- Taxonomy reviewed by product and trust-and-safety owners.
- Dataset contains Indonesian, English, and mixed-language examples.
- Dataset validation passes.
- Training run config records model checkpoint, data version, and prompt version.

## Before Shadow Deployment

- Model card draft updated with actual data sources and evaluation results.
- Offline evaluation completed with precision-first threshold notes.
- vLLM smoke test returns valid JSON.
- Backend logs model version, parsed decision, raw output, latency, and final product action.

## During Shadow Mode

- Sample flagged and safe comments for human review.
- Track false positives by label and language.
- Track p50, p95, and p99 latency.
- Keep action as `shadow_log`.

## Before Enforcement

- Human review approves precision for each enforced label.
- Appeals and audit process are defined.
- Rollback path is tested.
- Enforcement thresholds are versioned and reviewed.
