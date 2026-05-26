"""Dataset validation and SFT conversion helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from comment_filtering.taxonomy import ModerationDecision


LANGUAGES = ("id", "en", "mixed", "other")

SYSTEM_PROMPT = (
    "You are a live-stream chat moderation model for an OTT platform. "
    "Classify the user comment using only the supported taxonomy and return only valid JSON "
    "with keys flagged, labels, severity, confidence, and action."
)


@dataclass(frozen=True)
class ModerationRecord:
    comment_id: str
    text: str
    language: str
    source: str
    decision: ModerationDecision


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, value


def validate_record(record: dict[str, Any], *, line_number: int | None = None) -> ModerationRecord:
    prefix = f"line {line_number}: " if line_number else ""
    required = ("comment_id", "text", "language", "decision")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{prefix}missing required fields: {', '.join(missing)}")

    comment_id = _expect_string(record["comment_id"], f"{prefix}comment_id")
    text = _expect_string(record["text"], f"{prefix}text")
    language = _expect_string(record["language"], f"{prefix}language")
    source = _expect_string(record.get("source", "unknown"), f"{prefix}source")

    if not text.strip():
        raise ValueError(f"{prefix}text cannot be empty")
    if language not in LANGUAGES:
        raise ValueError(f"{prefix}language must be one of: {', '.join(LANGUAGES)}")
    if not isinstance(record["decision"], dict):
        raise ValueError(f"{prefix}decision must be an object")

    decision = _decision_from_dict(record["decision"], prefix=prefix)
    return ModerationRecord(
        comment_id=comment_id,
        text=text,
        language=language,
        source=source,
        decision=decision,
    )


def record_to_sft_messages(record: ModerationRecord) -> dict[str, Any]:
    assistant_json = json.dumps(record.decision.to_dict(), ensure_ascii=False, sort_keys=True)
    return {
        "comment_id": record.comment_id,
        "language": record.language,
        "source": record.source,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Comment language: {record.language}\nComment: {record.text}"},
            {"role": "assistant", "content": assistant_json},
        ],
    }


def _decision_from_dict(value: dict[str, Any], *, prefix: str) -> ModerationDecision:
    required = ("flagged", "labels", "severity", "confidence", "action")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{prefix}decision missing fields: {', '.join(missing)}")
    if not isinstance(value["flagged"], bool):
        raise ValueError(f"{prefix}decision.flagged must be a boolean")
    if not isinstance(value["labels"], list) or not all(
        isinstance(label, str) for label in value["labels"]
    ):
        raise ValueError(f"{prefix}decision.labels must be a list of strings")
    if not isinstance(value["confidence"], (int, float)):
        raise ValueError(f"{prefix}decision.confidence must be a number")

    return ModerationDecision(
        flagged=value["flagged"],
        labels=tuple(value["labels"]),
        severity=_expect_string(value["severity"], f"{prefix}decision.severity"),
        confidence=float(value["confidence"]),
        action=_expect_string(value["action"], f"{prefix}decision.action"),
    )


def _expect_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    return value
