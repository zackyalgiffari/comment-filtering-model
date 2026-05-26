import pytest

from comment_filtering.taxonomy import ModerationDecision, validate_labels


def test_safe_label_cannot_mix_with_unsafe_label() -> None:
    with pytest.raises(ValueError):
        validate_labels(["safe", "spam_or_scam"])


def test_flagged_must_match_labels() -> None:
    with pytest.raises(ValueError):
        ModerationDecision(
            flagged=False,
            labels=("hate_or_harassment",),
            severity="low",
            confidence=0.8,
        )
