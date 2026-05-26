#!/usr/bin/env python3
"""Print the HF Jobs UV payload for the demo fine-tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo_training.yaml"))
    parser.add_argument("--script", type=Path, default=Path("scripts/hf_jobs_train_sft.py"))
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print payload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["demo_training"]
    job_config = config["hf_jobs"]
    script = args.script.read_text(encoding="utf-8")
    payload = {
        "type": "uv",
        "flavor": job_config["flavor"],
        "timeout": job_config["timeout"],
        "secrets": job_config["secrets"],
        "script": script,
        "args": [
            "--model-name",
            config["model_name"],
            "--dataset-name",
            config["dataset_name"],
            "--train-split",
            config["train_split"],
            "--eval-split",
            config["eval_split"],
            "--output-dir",
            config["output_dir"],
            "--hub-model-id",
            config["hub_model_id"],
            "--max-length",
            str(config["max_length"]),
            "--num-train-epochs",
            str(config["num_train_epochs"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--per-device-train-batch-size",
            str(config["per_device_train_batch_size"]),
            "--gradient-accumulation-steps",
            str(config["gradient_accumulation_steps"]),
            "--lora-r",
            str(config["lora_r"]),
            "--lora-alpha",
            str(config["lora_alpha"]),
            "--lora-dropout",
            str(config["lora_dropout"]),
            "--trackio-project",
            config["trackio_project"],
            "--run-name",
            config["run_name"],
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete; payload was printed only.")


if __name__ == "__main__":
    main()
