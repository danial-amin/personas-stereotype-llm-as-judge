from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.models import EvaluationResult, Persona
from src.response_parser import parse_json_response


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
        "image_stereotype_answer",
        "image_stereotype_explanation",
        "description_stereotype_answer",
        "description_stereotype_explanation",
        "stereotype_age",
        "stereotype_gender",
        "stereotype_occupation",
        "stereotype_other",
        "stereotype_other_text",
        "why_assessment",
        "understand_group_answer",
        "understand_group_reason",
        "relatable_answer",
        "relatable_reason",
        "llm_confidence_score",
        "llm_confidence_explanation",
        "response_raw",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in evaluations:
            parsed = item.parsed or {}
            image = parsed.get("image_stereotype") or {}
            description = parsed.get("persona_description_stereotype") or {}
            stereotype_type = parsed.get("stereotype_type") or {}
            understand = parsed.get("this_persona_helped_me_understand_this_group_of_people") or {}
            relatable = parsed.get("i_find_this_persona_relatable") or {}
            confidence = parsed.get("llm_confidence") or {}

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
                    "image_stereotype_answer": image.get("answer", ""),
                    "image_stereotype_explanation": image.get(
                        "what_about_the_image_appears_stereotype_to_you", ""
                    ),
                    "description_stereotype_answer": description.get("answer", ""),
                    "description_stereotype_explanation": description.get(
                        "what_about_the_persona_description_appears_stereotype_to_you", ""
                    ),
                    "stereotype_age": stereotype_type.get("age", ""),
                    "stereotype_gender": stereotype_type.get("gender", ""),
                    "stereotype_occupation": stereotype_type.get("occupation", ""),
                    "stereotype_other": stereotype_type.get("other", ""),
                    "stereotype_other_text": stereotype_type.get("other_text", ""),
                    "why_assessment": parsed.get("why_did_you_give_this_assessment", ""),
                    "understand_group_answer": understand.get("answer", ""),
                    "understand_group_reason": understand.get("reason", ""),
                    "relatable_answer": relatable.get("answer", ""),
                    "relatable_reason": relatable.get("reason", ""),
                    "llm_confidence_score": confidence.get("score", ""),
                    "llm_confidence_explanation": confidence.get("explanation", ""),
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


def reparse_evaluation(item: EvaluationResult) -> EvaluationResult:
    if not item.response_raw.strip():
        return item

    parsed = parse_json_response(item.response_raw)
    if parsed is not None:
        return EvaluationResult(
            persona_id=item.persona_id,
            model_key=item.model_key,
            model_display_name=item.model_display_name,
            model_id=item.model_id,
            run_index=item.run_index,
            timestamp=item.timestamp,
            prompt_text=item.prompt_text,
            response_raw=item.response_raw,
            parsed=parsed,
            latency_ms=item.latency_ms,
            error=None,
        )

    if item.error and "parse" in item.error.lower():
        return item

    return EvaluationResult(
        persona_id=item.persona_id,
        model_key=item.model_key,
        model_display_name=item.model_display_name,
        model_id=item.model_id,
        run_index=item.run_index,
        timestamp=item.timestamp,
        prompt_text=item.prompt_text,
        response_raw=item.response_raw,
        parsed=None,
        latency_ms=item.latency_ms,
        error=item.error or "Failed to parse JSON response",
    )
