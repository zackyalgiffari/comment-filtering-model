# comment-filtering-model

Open-weight comment-filtering pipeline for Indonesian/English OTT live-stream chat.

The target model is `Qwen/Qwen3.5-2B`, fine-tuned with LoRA SFT for compact JSON
moderation decisions. The first production mode is shadow logging: the model records
labels and confidence for live comments but does not block users.

## Target Model

- Primary checkpoint: `Qwen/Qwen3.5-2B`
- Base reference: `Qwen/Qwen3.5-2B-Base`
- Parameter count: about 2.27B
- License: Apache-2.0
- Training method: TRL `SFTTrainer` + PEFT LoRA
- Serving target: vLLM on a GPU-backed OTT backend service

Do not fine-tune from GGUF, AWQ, or other quantized deployment variants. Those are
deployment artifacts, not training checkpoints.

## Moderation Output

Every model response should be valid JSON:

```json
{
  "flagged": true,
  "labels": ["hate_or_harassment"],
  "severity": "medium",
  "confidence": 0.82,
  "action": "shadow_log"
}
```

Supported labels are configured in `configs/taxonomy.yaml`.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,serving]"
python -m compileall src scripts tests
pytest
```

Training and serving scripts are designed to be reproducible, but the initial repo
implementation does not launch a paid Hugging Face GPU job by default.

## Dataset Preparation

```bash
PYTHONPATH=src python3 scripts/validate_dataset.py data/bootstrap_sample.jsonl
PYTHONPATH=src python3 scripts/prepare_sft_dataset.py \
  data/bootstrap_sample.jsonl outputs/bootstrap_sft.jsonl
```

## Training Dry Run

```bash
PYTHONPATH=src python3 scripts/train_lora_sft.py \
  --train-jsonl outputs/bootstrap_sft.jsonl \
  --dry-run
```

## Offline Evaluation

```bash
PYTHONPATH=src python3 scripts/evaluate_predictions.py \
  --gold-jsonl data/bootstrap_sample.jsonl \
  --predictions-jsonl data/bootstrap_predictions_sample.jsonl
```

## vLLM Smoke Test

Dry-run the request payload without needing a running server:

```bash
PYTHONPATH=src python3 scripts/smoke_vllm.py --dry-run
```

When vLLM is running with the fine-tuned model:

```bash
PYTHONPATH=src python3 scripts/smoke_vllm.py \
  --base-url http://localhost:8000/v1 \
  --model your-org/qwen3.5-2b-comment-filtering
```

## Demo Dataset And HF Jobs

Build ignored local artifacts from enabled demo sources:

```bash
PYTHONPATH=src python3 scripts/build_demo_dataset.py
PYTHONPATH=src python3 scripts/split_dataset.py \
  outputs/datasets/demo_v1/raw.jsonl outputs/datasets/demo_v1/splits
PYTHONPATH=src python3 scripts/prepare_hf_dataset_upload.py outputs/datasets/demo_v1/splits
```

Print the self-contained HF Jobs payload without submitting a paid job:

```bash
PYTHONPATH=src python3 scripts/print_hf_job_payload.py --dry-run
python3 scripts/hf_jobs_train_sft.py --dry-run
```

## Docs

- Training: `docs/training_runbook.md`
- Deployment: `docs/deployment_runbook.md`
- API contract: `docs/api_contract.md`
- Thresholding: `docs/thresholding.md`
- Rollout: `docs/rollout_checklist.md`
- Model card draft: `docs/model_card_draft.md`
- Synthetic data: `docs/synthetic_generation.md`
- HF Jobs launch: `docs/hf_jobs_launch_checklist.md`
