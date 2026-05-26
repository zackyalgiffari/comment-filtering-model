"""Demo dataset source adapters and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from comment_filtering.dataset import ModerationRecord, iter_jsonl, validate_record
from comment_filtering.taxonomy import ModerationDecision


PERMISSIVE_LICENSES = {"apache-2.0", "mit", "project-sample"}


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    kind: str
    license: str
    language: str = "id"
    path: str | None = None
    repo_id: str | None = None
    enabled_by_default: bool = False

    @property
    def is_permissive(self) -> bool:
        return self.license.lower() in PERMISSIVE_LICENSES


@dataclass(frozen=True)
class DemoDatasetRow:
    record: ModerationRecord
    source_id: str
    source_license: str

    def to_raw_record(self) -> dict[str, Any]:
        return {
            "comment_id": self.record.comment_id,
            "text": self.record.text,
            "language": self.record.language,
            "source": self.source_id,
            "source_license": self.source_license,
            "decision": self.record.decision.to_dict(),
        }


def parse_sources(config: dict[str, Any]) -> list[DatasetSource]:
    source_values = config.get("demo_dataset", {}).get("sources", [])
    if not isinstance(source_values, list):
        raise ValueError("demo_dataset.sources must be a list")
    sources = []
    for value in source_values:
        if not isinstance(value, dict):
            raise ValueError("Each source must be an object")
        sources.append(
            DatasetSource(
                source_id=_required_string(value, "id"),
                kind=_required_string(value, "kind"),
                license=_required_string(value, "license"),
                language=value.get("language", "id"),
                path=value.get("path"),
                repo_id=value.get("repo_id"),
                enabled_by_default=bool(value.get("enabled_by_default", False)),
            )
        )
    return sources


def load_source_rows(source: DatasetSource, *, limit: int | None = None) -> Iterator[DemoDatasetRow]:
    if not source.is_permissive:
        raise ValueError(f"Source {source.source_id} has non-permissive license: {source.license}")
    if source.kind == "local_jsonl":
        yield from _load_local_jsonl(source, limit=limit)
        return
    if source.kind == "hf_dataset":
        yield from _load_hf_dataset(source, limit=limit)
        return
    raise ValueError(f"Unsupported source kind for {source.source_id}: {source.kind}")


def sanitize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_manifest(
    *,
    dataset_version: str,
    source_rows: Iterable[DemoDatasetRow],
    artifact_paths: Iterable[Path],
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    licenses: dict[str, str] = {}
    row_count = 0

    for row in source_rows:
        row_count += 1
        source_counts[row.source_id] += 1
        language_counts[row.record.language] += 1
        licenses[row.source_id] = row.source_license
        for label in row.record.decision.labels:
            label_counts[label] += 1

    return {
        "dataset_version": dataset_version,
        "row_count": row_count,
        "source_counts": dict(sorted(source_counts.items())),
        "source_licenses": dict(sorted(licenses.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "artifacts": {
            str(path): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
            if path.exists()
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_jsonl(source: DatasetSource, *, limit: int | None) -> Iterator[DemoDatasetRow]:
    if source.path is None:
        raise ValueError(f"Local source {source.source_id} is missing path")
    count = 0
    for line_number, record in iter_jsonl(Path(source.path)):
        validated = validate_record(record, line_number=line_number)
        yield DemoDatasetRow(validated, source.source_id, source.license)
        count += 1
        if limit is not None and count >= limit:
            return


def _load_hf_dataset(source: DatasetSource, *, limit: int | None) -> Iterator[DemoDatasetRow]:
    if source.repo_id is None:
        raise ValueError(f"HF source {source.source_id} is missing repo_id")

    from datasets import load_dataset

    dataset = load_dataset(source.repo_id, split="train", streaming=limit is not None)
    count = 0
    for raw_row in dataset:
        mapped = map_public_row(source, raw_row, index=count)
        if mapped is not None:
            yield mapped
            count += 1
        if limit is not None and count >= limit:
            return


def map_public_row(
    source: DatasetSource,
    raw_row: dict[str, Any],
    *,
    index: int,
) -> DemoDatasetRow | None:
    text = _first_string(raw_row, ("text", "tweet", "Tweet", "comment_text", "content"))
    if not text:
        return None

    decision = _map_decision(source, raw_row)
    record = ModerationRecord(
        comment_id=f"{source.source_id}-{index:08d}",
        text=sanitize_text(text),
        language=source.language,
        source=source.source_id,
        decision=decision,
    )
    return DemoDatasetRow(record, source.source_id, source.license)


def _map_decision(source: DatasetSource, raw_row: dict[str, Any]) -> ModerationDecision:
    if source.source_id == "nahiar_hate_speech_detection":
        hate = _truthy(raw_row, ("HS", "hate_speech", "is_hate_speech"))
        abusive = _truthy(raw_row, ("Abusive", "abusive", "is_abusive"))
        strong = _truthy(raw_row, ("HS_Strong", "strong"))
        moderate = _truthy(raw_row, ("HS_Moderate", "moderate"))
        if hate or abusive:
            labels = ("hate_or_harassment",) if hate else ("profanity",)
            severity = "high" if strong else "medium" if moderate else "low"
            return ModerationDecision(True, labels, severity, 0.8)
        return ModerationDecision(False, ("safe",), "low", 0.9)

    if source.source_id == "haipradana_indonesian_twitter_hate_speech_cleaned":
        label = _first_string(raw_row, ("label", "Label", "class", "sentiment"))
        if label and label.lower() in {"hate", "hateful", "1", "toxic"}:
            return ModerationDecision(True, ("hate_or_harassment",), "medium", 0.8)
        return ModerationDecision(False, ("safe",), "low", 0.9)

    raise ValueError(f"No public row mapper configured for {source.source_id}")


def _first_string(raw_row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw_row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and key.lower() == "label":
            return str(value)
    return None


def _truthy(raw_row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = raw_row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return value > 0
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "hate", "hateful", "abusive"}
    return False


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Missing required string source field: {key}")
    return item
