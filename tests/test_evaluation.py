import unittest

from comment_filtering.dataset import validate_record
from comment_filtering.evaluation import evaluate_predictions
from comment_filtering.taxonomy import ModerationDecision


class EvaluationTest(unittest.TestCase):
    def test_evaluate_predictions_reports_flagged_precision(self) -> None:
        gold = {
            "1": validate_record(
                {
                    "comment_id": "1",
                    "text": "Klik hadiah gratis",
                    "language": "id",
                    "decision": {
                        "flagged": True,
                        "labels": ["spam_or_scam"],
                        "severity": "medium",
                        "confidence": 0.9,
                        "action": "shadow_log",
                    },
                }
            ),
            "2": validate_record(
                {
                    "comment_id": "2",
                    "text": "Great match",
                    "language": "en",
                    "decision": {
                        "flagged": False,
                        "labels": ["safe"],
                        "severity": "low",
                        "confidence": 0.99,
                        "action": "shadow_log",
                    },
                }
            ),
        }
        predictions = {
            "1": ModerationDecision(
                flagged=True,
                labels=("spam_or_scam",),
                severity="medium",
                confidence=0.9,
            ),
            "2": ModerationDecision(
                flagged=False,
                labels=("safe",),
                severity="low",
                confidence=0.99,
            ),
        }

        report = evaluate_predictions(gold, predictions)

        self.assertEqual(report["flagged"]["precision"], 1.0)
        self.assertEqual(report["per_label"]["spam_or_scam"]["recall"], 1.0)
