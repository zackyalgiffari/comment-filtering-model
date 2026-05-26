#!/usr/bin/env python3
"""Import a reviewed synthetic JSONL batch into moderation dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path

from comment_filtering.dataset import ModerationRecord, iter_jsonl
from comment_filtering.demo_dataset import DemoDatasetRow, sanitize_text, write_jsonl
from comment_filtering.synthetic import validate_synthetic_batch
from comment_filtering.taxonomy import ModerationDecision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Reviewed synthetic JSONL batch")
    parser.add_argument("output", type=Path, help="Output moderation JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_synthetic_batch(args.input, require_review=True)
    rows = []
    for _, row in iter_jsonl(args.input):
        decision = row["decision"]
        record = ModerationRecord(
            comment_id=f"synthetic-{row['synthetic_id']}",
            text=sanitize_text(row["text"]),
            language=row["language"],
            source="synthetic_llm_reviewed",
            decision=ModerationDecision(
                flagged=decision["flagged"],
                labels=tuple(decision["labels"]),
                severity=decision["severity"],
                confidence=float(decision["confidence"]),
                action=decision["action"],
            ),
        )
        rows.append(DemoDatasetRow(record, "synthetic_llm_reviewed", "synthetic-reviewed"))

    count = write_jsonl(args.output, [row.to_raw_record() for row in rows])
    print(f"Imported {count} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
