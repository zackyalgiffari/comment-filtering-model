"""Offline metrics for moderation predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comment_filtering.dataset import ModerationRecord, iter_jsonl
from comment_filtering.taxonomy import ModerationDecision, UNSAFE_LABELS


def load_prediction_records(path: Path) -> dict[str, ModerationDecision]:
    predictions: dict[str, ModerationDecision] = {}
    for line_number, record in iter_jsonl(path):
        comment_id = record.get("comment_id")
        if not isinstance(comment_id, str) or not comment_id:
            raise ValueError(f"{path}:{line_number}: comment_id must be a non-empty string")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"{path}:{line_number}: prediction must be an object")
        if comment_id in predictions:
            raise ValueError(f"{path}:{line_number}: duplicate prediction for {comment_id}")
        predictions[comment_id] = _decision_from_prediction(prediction, path, line_number)
    if not predictions:
        raise ValueError(f"{path}: no predictions found")
    return predictions


def evaluate_predictions(
    gold: dict[str, ModerationRecord],
    predictions: dict[str, ModerationDecision],
) -> dict[str, Any]:
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing:
        raise ValueError(f"Missing predictions for: {', '.join(missing[:10])}")
    if extra:
        raise ValueError(f"Predictions without gold labels: {', '.join(extra[:10])}")

    per_label = {}
    for label in UNSAFE_LABELS:
        tp = fp = fn = 0
        for comment_id, gold_record in gold.items():
            gold_labels = _unsafe_set(gold_record.decision.labels)
            predicted_labels = _unsafe_set(predictions[comment_id].labels)
            if label in predicted_labels and label in gold_labels:
                tp += 1
            elif label in predicted_labels and label not in gold_labels:
                fp += 1
            elif label not in predicted_labels and label in gold_labels:
                fn += 1
        per_label[label] = _metric_dict(tp, fp, fn)

    flagged_tp = flagged_fp = flagged_fn = 0
    for comment_id, gold_record in gold.items():
        gold_flagged = gold_record.decision.flagged
        predicted_flagged = predictions[comment_id].flagged
        if predicted_flagged and gold_flagged:
            flagged_tp += 1
        elif predicted_flagged and not gold_flagged:
            flagged_fp += 1
        elif not predicted_flagged and gold_flagged:
            flagged_fn += 1

    return {
        "rows": len(gold),
        "flagged": _metric_dict(flagged_tp, flagged_fp, flagged_fn),
        "per_label": per_label,
        "optimization_target": "precision_first",
    }


def _decision_from_prediction(
    prediction: dict[str, Any],
    path: Path,
    line_number: int,
) -> ModerationDecision:
    required = ("flagged", "labels", "severity", "confidence", "action")
    missing = [field for field in required if field not in prediction]
    if missing:
        raise ValueError(f"{path}:{line_number}: prediction missing fields: {', '.join(missing)}")
    if not isinstance(prediction["labels"], list):
        raise ValueError(f"{path}:{line_number}: prediction.labels must be a list")
    return ModerationDecision(
        flagged=prediction["flagged"],
        labels=tuple(prediction["labels"]),
        severity=prediction["severity"],
        confidence=float(prediction["confidence"]),
        action=prediction["action"],
    )


def _unsafe_set(labels: tuple[str, ...]) -> set[str]:
    return {label for label in labels if label != "safe"}


def _metric_dict(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
