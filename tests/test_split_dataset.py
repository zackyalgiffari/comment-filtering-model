import unittest

from comment_filtering.demo_dataset import DatasetSource, map_public_row, split_rows


class SplitDatasetTest(unittest.TestCase):
    def test_split_rows_is_deterministic(self) -> None:
        source = DatasetSource(
            source_id="haipradana_indonesian_twitter_hate_speech_cleaned",
            kind="hf_dataset",
            license="apache-2.0",
            repo_id="haipradana/indonesian-twitter-hate-speech-cleaned",
        )
        rows = [
            map_public_row(source, {"text": f"row {index}", "label": "neutral"}, index=index)
            for index in range(10)
        ]
        typed_rows = [row for row in rows if row is not None]

        first = split_rows(typed_rows, seed=42, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1)
        second = split_rows(typed_rows, seed=42, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1)

        self.assertEqual(
            [row.record.comment_id for row in first["train"]],
            [row.record.comment_id for row in second["train"]],
        )
        self.assertEqual(len(first["train"]), 8)
        self.assertEqual(len(first["eval"]), 1)
        self.assertEqual(len(first["test"]), 1)
