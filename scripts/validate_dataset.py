#!/usr/bin/env python3
"""Validate moderation dataset JSONL before fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from comment_filtering.dataset import iter_jsonl, validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Moderation JSONL file to validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = 0
    for line_number, record in iter_jsonl(args.path):
        validate_record(record, line_number=line_number)
        count += 1
    print(f"Validated {count} rows from {args.path}")


if __name__ == "__main__":
    main()
