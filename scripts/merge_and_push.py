#!/usr/bin/env python3
"""Merge LoRA adapter into base model and push the merged weights to HuggingFace Hub."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


MODEL_CARD_TEMPLATE = textwrap.dedent("""\
    ---
    language:
      - id
      - en
    license: apache-2.0
    base_model: Qwen/Qwen3-1.7B
    tags:
      - text-classification
      - content-moderation
      - hate-speech-detection
      - indonesian
      - lora
      - sft
    datasets:
      - nahiar/hate_speech_detection
      - haipradana/indonesian-twitter-hate-speech-cleaned
    ---

    # Qwen3-1.7B Comment Filtering

    Fine-tuned [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) for Indonesian/English
    live-stream chat moderation using LoRA SFT. The model classifies comments into a 7-label
    moderation taxonomy and returns a structured JSON decision.

    ## Usage

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, json, re

    model_id = "{hub_repo}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    SYSTEM_PROMPT = (
        "You are a live-stream chat moderation model for an OTT platform. "
        "Classify the user comment using only the supported taxonomy and return only valid JSON "
        "with keys flagged, labels, severity, confidence, and action."
    )

    def moderate(text: str, language: str = "id") -> dict:
        messages = [
            {{"role": "system", "content": SYSTEM_PROMPT}},
            {{"role": "user", "content": f"Comment language: {{language}}\\nComment: {{text}}"}},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        with torch.inference_mode():
            output = model.generate(inputs.to(model.device), max_new_tokens=96, temperature=None, do_sample=False)
        text_out = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        text_out = re.sub(r"<think>.*?</think>", "", text_out, flags=re.DOTALL).strip()
        return json.loads(text_out)

    print(moderate("Hei semuanya, selamat datang!"))
    ```

    ## Taxonomy

    | Label | Description |
    |---|---|
    | `safe` | Compliant comment |
    | `hate_or_harassment` | Hate speech or targeted harassment |
    | `sexual` | Sexual or explicit content |
    | `violence_or_threat` | Threats or violent content |
    | `spam_or_scam` | Spam, advertising, or scam |
    | `profanity` | Offensive language without targeting |
    | `self_harm` | Self-harm or suicide-related content |

    Severity values: `low`, `medium`, `high`. Action: `shadow_log`.

    ## Training Data

    Trained on publicly available Indonesian hate-speech datasets:
    - [nahiar/hate_speech_detection](https://huggingface.co/datasets/nahiar/hate_speech_detection) (MIT)
    - [haipradana/indonesian-twitter-hate-speech-cleaned](https://huggingface.co/datasets/haipradana/indonesian-twitter-hate-speech-cleaned) (Apache-2.0)

    ## Limitations

    - Public training data may not match OTT live-chat slang or regional language patterns.
    - Low-resource labels (`self_harm`, `sexual`) require careful review before enforcement.
    - Confidence values must be calibrated against your own validation data.
    - Initial deployment in shadow-log mode is strongly recommended.

    ## License

    Apache-2.0
""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B", help="Base model ID on HF Hub")
    parser.add_argument("--hub-repo", required=True, help="HF Hub repo ID to push to (org/model-name)")
    parser.add_argument("--private", action="store_true", help="Push as a private repository")
    parser.add_argument("--dry-run", action="store_true", help="Merge weights but do not push")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    print(f"Loading base model: {args.base_model}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    if args.dry_run:
        print("Dry run: merge successful. Skipping Hub push.")
        return

    print(f"Pushing merged model to: {args.hub_repo}")
    model.push_to_hub(args.hub_repo, private=args.private)
    tokenizer.push_to_hub(args.hub_repo, private=args.private)

    from huggingface_hub import HfApi

    api = HfApi()
    card_content = MODEL_CARD_TEMPLATE.format(hub_repo=args.hub_repo)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=args.hub_repo,
        repo_type="model",
    )
    print(f"Done. Model available at: https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
