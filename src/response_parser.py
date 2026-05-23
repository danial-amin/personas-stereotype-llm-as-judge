from __future__ import annotations

import json
import re
from typing import Any


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Best-effort parse of model JSON output."""
    cleaned = _strip_markdown_fence(text.strip())
    if not cleaned:
        return None

    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _strip_markdown_fence(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(text)
    add(_repair_common_model_mistakes(text))

    balanced = _extract_balanced_object(text)
    if balanced:
        add(balanced)
        add(_repair_common_model_mistakes(balanced))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        add(text[start : end + 1])
        add(_repair_common_model_mistakes(text[start : end + 1]))

    return candidates


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _repair_common_model_mistakes(text: str) -> str:
    """Fix frequent LLM JSON formatting mistakes."""
    repaired = text

    # Model closes a string field with `"},` instead of `",`
    repaired = re.sub(
        r'"\s*\n(\s*)\},\s*\n(\s*)"',
        r'",\n\1\2"',
        repaired,
    )

    # Literal true/false from prompt examples
    repaired = re.sub(r"\btrue/false\b", "false", repaired, flags=re.IGNORECASE)

    # Trailing commas before } or ]
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    return repaired
