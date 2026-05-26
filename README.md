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
