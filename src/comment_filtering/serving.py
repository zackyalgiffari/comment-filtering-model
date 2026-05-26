"""Serving helpers for vLLM OpenAI-compatible moderation."""

from __future__ import annotations

import json
import re
from typing import Any

from comment_filtering.dataset import SYSTEM_PROMPT
from comment_filtering.taxonomy import ACTIONS, LABELS, SEVERITIES, ModerationDecision


MODERATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "flagged": {"type": "boolean"},
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": list(LABELS)},
            "minItems": 1,
            "uniqueItems": True,
        },
        "severity": {"type": "string", "enum": list(SEVERITIES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "action": {"type": "string", "enum": list(ACTIONS)},
    },
    "required": ["flagged", "labels", "severity", "confidence", "action"],
    "additionalProperties": False,
}


def build_messages(comment: str, language: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Comment language: {language}\nComment: {comment}"},
    ]


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks before JSON parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_decision_json(content: str) -> ModerationDecision:
    content = strip_think_tags(content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return ModerationDecision(
        flagged=value.get("flagged"),
        labels=tuple(value.get("labels", [])),
        severity=value.get("severity"),
        confidence=float(value.get("confidence")),
        action=value.get("action", "shadow_log"),
    )


def moderate_comment(
    *,
    comment: str,
    language: str,
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    max_tokens: int = 96,
) -> ModerationDecision:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=build_messages(comment, language),
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "moderation_decision",
                "strict": True,
                "schema": MODERATION_JSON_SCHEMA,
            },
        },
    )
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Model response did not include message content")
    return parse_decision_json(content)
