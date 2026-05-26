import unittest

from comment_filtering.taxonomy import ModerationDecision, validate_labels


class TaxonomyTest(unittest.TestCase):
    def test_safe_label_cannot_mix_with_unsafe_label(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["safe", "spam_or_scam"])


    def test_flagged_must_match_labels(self) -> None:
        with self.assertRaises(ValueError):
            ModerationDecision(
                flagged=False,
                labels=("hate_or_harassment",),
                severity="low",
                confidence=0.8,
            )
