"""Shared moderation taxonomy and output validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


LABELS = (
    "safe",
    "hate_or_harassment",
    "sexual",
    "violence_or_threat",
    "spam_or_scam",
    "profanity",
    "self_harm",
)

UNSAFE_LABELS = tuple(label for label in LABELS if label != "safe")
SEVERITIES = ("low", "medium", "high")
ACTIONS = ("shadow_log",)


def validate_labels(labels: Iterable[str]) -> list[str]:
    """Return labels as a list after checking taxonomy membership and safe mixing."""

    normalized = list(labels)
    unknown = sorted(set(normalized) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(unknown)}")
    if not normalized:
        raise ValueError("At least one label is required")
    if "safe" in normalized and len(normalized) > 1:
        raise ValueError("'safe' cannot be combined with unsafe labels")
    return normalized


@dataclass(frozen=True)
class ModerationDecision:
    """Canonical model output for shadow-mode moderation."""

    flagged: bool
    labels: tuple[str, ...]
    severity: str
    confidence: float
    action: str = "shadow_log"

    def __post_init__(self) -> None:
        validate_labels(self.labels)
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {self.severity}")
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown action: {self.action}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        expected_flagged = "safe" not in self.labels
        if self.flagged != expected_flagged:
            raise ValueError("flagged must be false only for the safe label")

    def to_dict(self) -> dict[str, object]:
        return {
            "flagged": self.flagged,
            "labels": list(self.labels),
            "severity": self.severity,
            "confidence": self.confidence,
            "action": self.action,
        }
