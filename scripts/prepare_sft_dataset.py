#!/usr/bin/env python3
"""Convert validated moderation records into chat-style SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comment_filtering.dataset import iter_jsonl, record_to_sft_messages, validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw moderation JSONL file")
    parser.add_argument("output", type=Path, help="Output SFT JSONL file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for line_number, record in iter_jsonl(args.input):
            validated = validate_record(record, line_number=line_number)
            handle.write(json.dumps(record_to_sft_messages(validated), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
