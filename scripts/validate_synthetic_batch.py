#!/usr/bin/env python3
"""Validate an LLM-assisted synthetic moderation JSONL batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from comment_filtering.synthetic import validate_synthetic_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Synthetic JSONL batch")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow review.reviewed=false for pre-review generation checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = validate_synthetic_batch(args.path, require_review=not args.allow_unreviewed)
    print(f"Validated {count} synthetic rows from {args.path}")


if __name__ == "__main__":
    main()
