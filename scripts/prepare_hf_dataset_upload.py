#!/usr/bin/env python3
"""Prepare a local dataset directory for Hugging Face Hub upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


README_TEMPLATE = """---
license: apache-2.0
task_categories:
- text-classification
language:
- id
- en
pretty_name: Qwen3.5 Comment Filtering Demo Dataset
---

# Qwen3.5 Comment Filtering Demo Dataset

This dataset artifact is prepared for a demo fine-tune of
`Qwen/Qwen3.5-2B` for OTT live-chat moderation.

The GitHub repository intentionally does not commit raw toxic training examples.
Review `manifest.json` for source counts, labels, licenses, and artifact hashes.

Files:

- `train.jsonl`
- `eval.jsonl`
- `test.jsonl`
- `manifest.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory containing split JSONL files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = ["train.jsonl", "eval.jsonl", "test.jsonl", "manifest.json"]
    missing = [name for name in required if not (args.dataset_dir / name).exists()]
    if missing:
        raise ValueError(f"Dataset directory is missing: {', '.join(missing)}")

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    readme = README_TEMPLATE + "\n## Manifest Summary\n\n```json\n"
    readme += json.dumps(
        {
            "dataset_version": manifest.get("dataset_version"),
            "row_count": manifest.get("row_count"),
            "source_licenses": manifest.get("source_licenses"),
            "label_counts": manifest.get("label_counts"),
        },
        indent=2,
        sort_keys=True,
    )
    readme += "\n```\n"
    (args.dataset_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {(args.dataset_dir / 'README.md')}")


if __name__ == "__main__":
    main()
