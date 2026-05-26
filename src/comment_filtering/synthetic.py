"""Validation for LLM-assisted synthetic moderation batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comment_filtering.dataset import LANGUAGES, iter_jsonl
from comment_filtering.taxonomy import ModerationDecision


SYNTHETIC_LANGUAGES = {"id", "en", "mixed"}


def validate_synthetic_batch(path: Path, *, require_review: bool = True) -> int:
    count = 0
    seen_ids: set[str] = set()
    for line_number, row in iter_jsonl(path):
        validate_synthetic_row(row, line_number=line_number, require_review=require_review)
        synthetic_id = row["synthetic_id"]
        if synthetic_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate synthetic_id {synthetic_id}")
        seen_ids.add(synthetic_id)
        count += 1
    if count == 0:
        raise ValueError(f"{path}: no synthetic rows found")
    return count


def validate_synthetic_row(
    row: dict[str, Any],
    *,
    line_number: int | None = None,
    require_review: bool = True,
) -> None:
    prefix = f"line {line_number}: " if line_number else ""
    required = ("synthetic_id", "text", "language", "decision", "review")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{prefix}missing required fields: {', '.join(missing)}")

    _expect_string(row["synthetic_id"], f"{prefix}synthetic_id")
    text = _expect_string(row["text"], f"{prefix}text")
    language = _expect_string(row["language"], f"{prefix}language")
    if not text.strip():
        raise ValueError(f"{prefix}text cannot be empty")
    if language not in LANGUAGES or language not in SYNTHETIC_LANGUAGES:
        raise ValueError(f"{prefix}language must be one of: {', '.join(sorted(SYNTHETIC_LANGUAGES))}")
    if not isinstance(row["decision"], dict):
        raise ValueError(f"{prefix}decision must be an object")

    decision = row["decision"]
    ModerationDecision(
        flagged=_expect_bool(decision.get("flagged"), f"{prefix}decision.flagged"),
        labels=tuple(_expect_string_list(decision.get("labels"), f"{prefix}decision.labels")),
        severity=_expect_string(decision.get("severity"), f"{prefix}decision.severity"),
        confidence=float(_expect_number(decision.get("confidence"), f"{prefix}decision.confidence")),
        action=_expect_string(decision.get("action"), f"{prefix}decision.action"),
    )

    review = row["review"]
    if not isinstance(review, dict):
        raise ValueError(f"{prefix}review must be an object")
    reviewed = _expect_bool(review.get("reviewed"), f"{prefix}review.reviewed")
    _expect_string(review.get("reviewer"), f"{prefix}review.reviewer", allow_empty=not require_review)
    _expect_string(review.get("notes"), f"{prefix}review.notes", allow_empty=True)
    if require_review and not reviewed:
        raise ValueError(f"{prefix}review.reviewed must be true before import")


def _expect_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _expect_number(value: Any, field_name: str) -> int | float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return value


def _expect_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value
