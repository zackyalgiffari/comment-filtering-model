import unittest

from comment_filtering.dataset import record_to_sft_messages, validate_record


class DatasetTest(unittest.TestCase):
    def test_validate_record_accepts_safe_comment(self) -> None:
        record = validate_record(
            {
                "comment_id": "1",
                "text": "Great match tonight",
                "language": "en",
                "decision": {
                    "flagged": False,
                    "labels": ["safe"],
                    "severity": "low",
                    "confidence": 0.99,
                    "action": "shadow_log",
                },
            }
        )

        self.assertEqual(record.comment_id, "1")
        self.assertFalse(record.decision.flagged)

    def test_record_to_sft_messages_contains_assistant_json(self) -> None:
        record = validate_record(
            {
                "comment_id": "2",
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
        )

        sft_row = record_to_sft_messages(record)

        self.assertEqual(sft_row["messages"][0]["role"], "system")
        self.assertEqual(sft_row["messages"][1]["role"], "user")
        self.assertEqual(sft_row["messages"][2]["role"], "assistant")
        self.assertIn('"spam_or_scam"', sft_row["messages"][2]["content"])
