import tempfile
import unittest
from pathlib import Path

from comment_filtering.demo_dataset import (
    DatasetSource,
    build_manifest,
    map_public_row,
    sanitize_text,
    write_jsonl,
)


class DemoDatasetTest(unittest.TestCase):
    def test_sanitize_text_removes_urls_and_handles(self) -> None:
        self.assertEqual(
            sanitize_text("hai @someone cek https://example.com now"),
            "hai @user cek [URL] now",
        )

    def test_map_nahiar_row_maps_hate_label(self) -> None:
        source = DatasetSource(
            source_id="nahiar_hate_speech_detection",
            kind="hf_dataset",
            license="mit",
            repo_id="nahiar/hate_speech_detection",
        )
        row = map_public_row(source, {"Tweet": "contoh komentar", "HS": 1}, index=7)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.record.decision.labels, ("hate_or_harassment",))
        self.assertEqual(row.record.comment_id, "nahiar_hate_speech_detection-00000007")

    def test_manifest_records_artifact_hashes(self) -> None:
        source = DatasetSource(
            source_id="haipradana_indonesian_twitter_hate_speech_cleaned",
            kind="hf_dataset",
            license="apache-2.0",
            repo_id="haipradana/indonesian-twitter-hate-speech-cleaned",
        )
        row = map_public_row(source, {"text": "aman", "label": "neutral"}, index=1)
        self.assertIsNotNone(row)
        assert row is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            write_jsonl(path, [row.to_raw_record()])
            manifest = build_manifest(
                dataset_version="demo_v1",
                source_rows=[row],
                artifact_paths=[path],
            )

        self.assertEqual(manifest["row_count"], 1)
        self.assertEqual(manifest["source_licenses"][source.source_id], "apache-2.0")
        self.assertIn("safe", manifest["label_counts"])
