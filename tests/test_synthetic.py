import tempfile
import unittest
from pathlib import Path

from comment_filtering.demo_dataset import write_jsonl
from comment_filtering.synthetic import validate_synthetic_batch, validate_synthetic_row


def valid_row() -> dict:
    return {
        "synthetic_id": "batch-0001",
        "text": "Komentar fiksi aman.",
        "language": "id",
        "decision": {
            "flagged": False,
            "labels": ["safe"],
            "severity": "low",
            "confidence": 0.95,
            "action": "shadow_log",
        },
        "review": {
            "reviewed": True,
            "reviewer": "qa",
            "notes": "checked",
        },
    }


class SyntheticTest(unittest.TestCase):
    def test_validate_synthetic_row_accepts_reviewed_row(self) -> None:
        validate_synthetic_row(valid_row())

    def test_validate_synthetic_row_rejects_unreviewed_import(self) -> None:
        row = valid_row()
        row["review"]["reviewed"] = False

        with self.assertRaises(ValueError):
            validate_synthetic_row(row)

    def test_validate_synthetic_batch_counts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synthetic.jsonl"
            write_jsonl(path, [valid_row()])

            self.assertEqual(validate_synthetic_batch(path), 1)
