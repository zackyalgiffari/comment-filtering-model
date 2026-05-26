import json
import unittest

from comment_filtering.serving import MODERATION_JSON_SCHEMA, build_messages, parse_decision_json


class ServingTest(unittest.TestCase):
    def test_build_messages_uses_two_turn_prompt(self) -> None:
        messages = build_messages("Mantap!", "id")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Mantap!", messages[1]["content"])

    def test_parse_decision_json(self) -> None:
        decision = parse_decision_json(
            json.dumps(
                {
                    "flagged": True,
                    "labels": ["spam_or_scam"],
                    "severity": "medium",
                    "confidence": 0.91,
                    "action": "shadow_log",
                }
            )
        )

        self.assertTrue(decision.flagged)
        self.assertEqual(decision.labels, ("spam_or_scam",))

    def test_schema_defines_required_output_keys(self) -> None:
        self.assertEqual(
            set(MODERATION_JSON_SCHEMA["required"]),
            {"flagged", "labels", "severity", "confidence", "action"},
        )
