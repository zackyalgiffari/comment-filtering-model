#!/usr/bin/env python3
"""Build the local demo moderation dataset from configured sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from comment_filtering.demo_dataset import (
    build_manifest,
    load_source_rows,
    parse_sources,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo_dataset.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source ID to include. Defaults to enabled_by_default sources.",
    )
    parser.add_argument("--limit-per-source", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_config = config["demo_dataset"]
    output_dir = args.output_dir or Path(dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = set(args.source or [])
    sources = [
        source
        for source in parse_sources(config)
        if source.source_id in selected_sources or (not selected_sources and source.enabled_by_default)
    ]
    if not sources:
        raise ValueError("No dataset sources selected")

    rows = []
    for source in sources:
        rows.extend(load_source_rows(source, limit=args.limit_per_source))

    raw_path = output_dir / "raw.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(raw_path, [row.to_raw_record() for row in rows])
    manifest = build_manifest(
        dataset_version=dataset_config["version"],
        source_rows=rows,
        artifact_paths=[raw_path],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {raw_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
