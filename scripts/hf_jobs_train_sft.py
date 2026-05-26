#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=2.20.0",
#   "peft>=0.12.0",
#   "torch>=2.3.0",
#   "trackio",
#   "transformers>=4.57.3",
#   "trl>=0.25.0",
# ]
# ///
"""Self-contained HF Jobs LoRA SFT training script for Qwen3 moderation."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset-name", default="zackyalgiffari/comment-filtering-demo-dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--output-dir", default="qwen3_1_7b_comment_filtering_demo")
    parser.add_argument("--hub-model-id", default="zackyalgiffari/comment-filtering-qwen3-1.7b")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--trackio-project", default="comment-filtering")
    parser.add_argument("--run-name", default="qwen3-1.7b-demo-sft")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    print(json.dumps(config, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete; HF Jobs training was not started.")
        return

    run_training(args)


def run_training(args: argparse.Namespace) -> None:
    import torch
    import trackio
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    trackio.init(project=args.trackio_project, name=args.run_name)

    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
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
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        report_to=["trackio"],
        project=args.trackio_project,
        run_name=args.run_name,
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
