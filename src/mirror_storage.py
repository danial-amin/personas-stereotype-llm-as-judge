from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.models import EvaluationResult


def _mirror_record(result: EvaluationResult, assignment: dict[str, Any]) -> dict[str, Any]:
    return {
        **result.to_dict(),
        "participant_id": assignment["participant_id"],
        "participant_index": assignment["participant_index"],
        "slot_index": assignment["slot_index"],
        "condition": assignment["condition"],
        "evaluation_index": assignment["evaluation_index"],
    }


def save_mirror_results(
    output_dir: Path,
    schedule_rows: list[dict[str, Any]],
    results: list[EvaluationResult],
    model_key: str,
    model_display_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [_mirror_record(result, assignment) for result, assignment in zip(results, schedule_rows)]

    json_path = output_dir / "evaluations.json"
    payload = {
        "experiment": "human_mirror_distribution",
        "model_key": model_key,
        "model_display_name": model_display_name,
        "participant_count": len({r["participant_id"] for r in records}),
        "evaluation_count": len(records),
        "evaluations": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "evaluations.csv"
    _save_mirror_csv(csv_path, records)
    return json_path, csv_path


def _save_mirror_csv(path: Path, records: list[dict[str, Any]]) -> None:
    base_fieldnames = [
        "evaluation_index",
        "participant_id",
        "participant_index",
        "slot_index",
        "condition",
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
        writer = csv.DictWriter(f, fieldnames=base_fieldnames)
        writer.writeheader()

        for record in records:
            parsed = record.get("parsed") or {}
            image = parsed.get("image_stereotype") or {}
            description = parsed.get("persona_description_stereotype") or {}
            stereotype_type = parsed.get("stereotype_type") or {}
            understand = parsed.get("this_persona_helped_me_understand_this_group_of_people") or {}
            relatable = parsed.get("i_find_this_persona_relatable") or {}
            confidence = parsed.get("llm_confidence") or {}

            writer.writerow(
                {
                    "evaluation_index": record["evaluation_index"],
                    "participant_id": record["participant_id"],
                    "participant_index": record["participant_index"],
                    "slot_index": record["slot_index"],
                    "condition": record["condition"],
                    "persona_id": record["persona_id"],
                    "model_key": record["model_key"],
                    "model_display_name": record["model_display_name"],
                    "model_id": record["model_id"],
                    "run_index": record["run_index"],
                    "timestamp": record["timestamp"],
                    "latency_ms": record["latency_ms"],
                    "error": record.get("error") or "",
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
                    "response_raw": record.get("response_raw", ""),
                }
            )


def load_mirror_results(output_dir: Path) -> list[dict[str, Any]]:
    json_path = output_dir / "evaluations.json"
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    return list(data.get("evaluations", []))


def completed_mirror_keys(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (record["participant_id"], record["persona_id"])
        for record in records
        if not record.get("error")
    }
