from comment_filtering.dataset import record_to_sft_messages, validate_record


def test_validate_record_accepts_safe_comment() -> None:
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

    assert record.comment_id == "1"
    assert record.decision.flagged is False


def test_record_to_sft_messages_contains_assistant_json() -> None:
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

    assert sft_row["messages"][0]["role"] == "system"
    assert sft_row["messages"][1]["role"] == "user"
    assert sft_row["messages"][2]["role"] == "assistant"
    assert '"spam_or_scam"' in sft_row["messages"][2]["content"]
