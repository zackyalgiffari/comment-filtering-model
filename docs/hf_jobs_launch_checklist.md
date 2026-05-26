# Hugging Face Jobs Launch Checklist

Use this checklist before submitting the paid demo fine-tuning job.

## Required Inputs

- Hugging Face account: `zackyalgiffari`
- Public model target: `zackyalgiffari/comment-filtering-qwen3.5-2b`
- Dataset uploaded to Hub, matching `configs/demo_training.yaml`
- `HF_TOKEN` with write access available to the job
- Dataset manifest reviewed for source licenses and label counts

## Preflight

```bash
PYTHONPATH=src python3 scripts/print_hf_job_payload.py --dry-run
python3 scripts/hf_jobs_train_sft.py --dry-run
```

The generated job payload is intended for an HF Jobs `uv` run. The job script is
self-contained and declares dependencies in its PEP 723 header.

## Submission Notes

- Keep timeout above the expected run time; default is `2h`.
- Keep `push_to_hub=True`; HF Jobs environments are ephemeral.
- Use Trackio logs to inspect training loss before using the model artifact.
- Do not publish as production-ready. The first artifact is a demo fine-tune.

## After Completion

- Confirm the model repo contains adapter or merged weights and a model card.
- Run the vLLM smoke test against the new model.
- Generate evaluation predictions and run `scripts/evaluate_predictions.py`.
- Update `docs/model_card_draft.md` with actual dataset and metric details.
