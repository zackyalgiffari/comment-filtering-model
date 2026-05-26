# Training Runbook

Use this runbook to prepare data and start a Qwen3.5 LoRA SFT run.

## Prerequisites

- Python 3.10 or newer.
- Dependencies installed with `pip install -e ".[dev,serving]"`.
- Labeled moderation data in the raw JSONL format used by `data/bootstrap_sample.jsonl`.
- Hugging Face write token if pushing to Hub.
- GPU training environment for actual fine-tuning.

## Prepare Data

```bash
PYTHONPATH=src python3 scripts/validate_dataset.py data/bootstrap_sample.jsonl
PYTHONPATH=src python3 scripts/prepare_sft_dataset.py \
  data/bootstrap_sample.jsonl outputs/bootstrap_sft.jsonl
```

Replace `data/bootstrap_sample.jsonl` with the real labeled dataset before production
training.

## Dry Run

```bash
PYTHONPATH=src python3 scripts/train_lora_sft.py \
  --train-jsonl outputs/bootstrap_sft.jsonl \
  --dry-run
```

Dry run validates the SFT file and prints the training configuration without importing
GPU training libraries.

## Fine-Tune

```bash
PYTHONPATH=src python3 scripts/train_lora_sft.py \
  --train-jsonl outputs/train_sft.jsonl \
  --eval-jsonl outputs/eval_sft.jsonl \
  --hub-model-id your-org/qwen3.5-2b-comment-filtering \
  --push-to-hub
```

The script trains LoRA adapters against `Qwen/Qwen3.5-2B` by default. Do not use GGUF,
AWQ, or other quantized checkpoints for training.

## After Training

- Save the exact git commit, dataset version, model repo, and training logs.
- Generate predictions on the evaluation set.
- Run `scripts/evaluate_predictions.py`.
- Update `docs/model_card_draft.md` with actual metrics and limitations.
