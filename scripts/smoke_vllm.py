#!/usr/bin/env python3
"""Smoke-test a vLLM OpenAI-compatible moderation endpoint."""

from __future__ import annotations

import argparse
import json

from comment_filtering.serving import build_messages, moderate_comment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--comment", default="Klik link hadiah gratis sekarang")
    parser.add_argument("--language", default="id")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--dry-run", action="store_true", help="Print request payload only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        payload = {
            "model": args.model,
            "messages": build_messages(args.comment, args.language),
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    decision = moderate_comment(
        comment=args.comment,
        language=args.language,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
