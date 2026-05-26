#!/usr/bin/env python3
"""Evaluate moderation predictions against labeled comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comment_filtering.dataset import iter_jsonl, validate_record
from comment_filtering.evaluation import evaluate_predictions, load_prediction_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-jsonl", type=Path, required=True, help="Labeled moderation JSONL")
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        required=True,
        help="Prediction JSONL with comment_id and prediction object",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = {}
    for line_number, record in iter_jsonl(args.gold_jsonl):
        validated = validate_record(record, line_number=line_number)
        gold[validated.comment_id] = validated
    predictions = load_prediction_records(args.predictions_jsonl)
    report = evaluate_predictions(gold, predictions)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
