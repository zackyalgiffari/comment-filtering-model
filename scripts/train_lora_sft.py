#!/usr/bin/env python3
"""Fine-tune Qwen/Qwen3.5-2B for comment moderation with LoRA SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from comment_filtering.dataset import iter_jsonl


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_OUTPUT_DIR = "outputs/qwen3_5_2b-comment-filtering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True, help="SFT JSONL with messages")
    parser.add_argument("--eval-jsonl", type=Path, default=None, help="Optional eval SFT JSONL")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print config only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count = validate_sft_jsonl(args.train_jsonl)
    eval_count = validate_sft_jsonl(args.eval_jsonl) if args.eval_jsonl else 0

    config = {
        "model_name": args.model_name,
        "train_rows": train_count,
        "eval_rows": eval_count,
        "output_dir": args.output_dir,
        "hub_model_id": args.hub_model_id,
        "push_to_hub": args.push_to_hub,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    print(json.dumps(config, indent=2, sort_keys=True))

    if args.dry_run:
        print("Dry run complete; training was not started.")
        return

    run_training(args)


def validate_sft_jsonl(path: Path | None) -> int:
    if path is None:
        return 0

    count = 0
    for line_number, record in iter_jsonl(path):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{path}:{line_number}: messages must contain system, user, assistant")
        expected_roles = ["system", "user", "assistant"]
        for index, expected_role in enumerate(expected_roles):
            message = messages[index]
            if not isinstance(message, dict):
                raise ValueError(f"{path}:{line_number}: messages[{index}] must be an object")
            if message.get("role") != expected_role:
                raise ValueError(f"{path}:{line_number}: messages[{index}].role must be {expected_role}")
            if not isinstance(message.get("content"), str) or not message["content"]:
                raise ValueError(f"{path}:{line_number}: messages[{index}].content must be text")
        count += 1

    if count == 0:
        raise ValueError(f"{path}: no SFT rows found")
    return count


def run_training(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    train_dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_dataset = (
        load_dataset("json", data_files=str(args.eval_jsonl), split="train")
        if args.eval_jsonl
        else None
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        save_steps=100,
        report_to=["trackio"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if processor is not None:
        trainer_kwargs["processing_class"] = processor

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
