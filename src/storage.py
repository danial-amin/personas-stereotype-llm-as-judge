from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.models import EvaluationResult, Persona


def persona_output_dir(results_dir: Path, persona_id: str) -> Path:
    return results_dir / persona_id


def save_persona_json(
    output_dir: Path,
    persona: Persona,
    evaluations: list[EvaluationResult],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "persona": persona.to_dict(),
        "evaluations": [item.to_dict() for item in evaluations],
    }
    path = output_dir / "evaluations.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_persona_csv(output_dir: Path, evaluations: list[EvaluationResult]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluations.csv"

    fieldnames = [
        "persona_id",
        "model_key",
        "model_display_name",
        "model_id",
        "run_index",
        "timestamp",
        "latency_ms",
        "error",
        "contains_stereotype",
        "stereotype_severity",
        "stereotype_categories",
        "text_stereotype_score",
        "image_stereotype_score",
        "text_image_consistency_score",
        "confidence",
        "reasoning",
        "specific_concerns",
        "response_raw",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in evaluations:
            parsed = item.parsed or {}
            writer.writerow(
                {
                    "persona_id": item.persona_id,
                    "model_key": item.model_key,
                    "model_display_name": item.model_display_name,
                    "model_id": item.model_id,
                    "run_index": item.run_index,
                    "timestamp": item.timestamp,
                    "latency_ms": item.latency_ms,
                    "error": item.error or "",
                    "contains_stereotype": parsed.get("contains_stereotype", ""),
                    "stereotype_severity": parsed.get("stereotype_severity", ""),
                    "stereotype_categories": _join_list(parsed.get("stereotype_categories")),
                    "text_stereotype_score": parsed.get("text_stereotype_score", ""),
                    "image_stereotype_score": parsed.get("image_stereotype_score", ""),
                    "text_image_consistency_score": parsed.get(
                        "text_image_consistency_score", ""
                    ),
                    "confidence": parsed.get("confidence", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "specific_concerns": _join_list(parsed.get("specific_concerns")),
                    "response_raw": item.response_raw,
                }
            )

    return path


def save_aggregate_json(
    results_dir: Path,
    personas: list[Persona],
    all_evaluations: dict[str, list[EvaluationResult]],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "persona_count": len(personas),
        "personas": {},
    }
    for persona in personas:
        payload["personas"][persona.persona_id] = {
            "persona": persona.to_dict(),
            "evaluations": [
                item.to_dict() for item in all_evaluations.get(persona.persona_id, [])
            ],
        }

    path = results_dir / "all_evaluations.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_existing_evaluations(output_dir: Path) -> list[EvaluationResult]:
    json_path = output_dir / "evaluations.json"
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    results: list[EvaluationResult] = []
    for item in data.get("evaluations", []):
        results.append(
            EvaluationResult(
                persona_id=item["persona_id"],
                model_key=item["model_key"],
                model_display_name=item["model_display_name"],
                model_id=item["model_id"],
                run_index=item["run_index"],
                timestamp=item["timestamp"],
                prompt_text=item["prompt_text"],
                response_raw=item.get("response_raw", ""),
                parsed=item.get("parsed"),
                latency_ms=item.get("latency_ms", 0),
                error=item.get("error"),
            )
        )
    return results


def _join_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)
