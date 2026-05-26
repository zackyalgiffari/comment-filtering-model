#!/usr/bin/env python3
"""Create deterministic train/eval/test splits for moderation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comment_filtering.demo_dataset import (
    build_manifest,
    read_moderation_rows,
    split_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Moderation raw JSONL")
    parser.add_argument("output_dir", type=Path, help="Output split directory")
    parser.add_argument("--dataset-version", default="demo_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_moderation_rows(args.input)
    splits = split_rows(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = []
    for split_name, split_rows_value in splits.items():
        path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(path, [row.to_raw_record() for row in split_rows_value])
        artifact_paths.append(path)
        print(f"Wrote {len(split_rows_value)} rows to {path}")

    manifest = build_manifest(
        dataset_version=args.dataset_version,
        source_rows=rows,
        artifact_paths=artifact_paths,
    )
    manifest["split_counts"] = {name: len(value) for name, value in splits.items()}
    manifest["seed"] = args.seed
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
