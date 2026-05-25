#!/usr/bin/env python3
"""Run a human-mirror distribution experiment with a single LLM.

Simulates 85 independent participants, each rating 6 personas (3 stereotyped,
3 non-stereotyped) using the exact assignment schedule from the Prolific study.
Each rating is a fresh, independent API call.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.evaluator import PersonaEvaluator
from src.human_assignments import DEFAULT_ASSIGNMENTS_PATH, load_mirror_schedule
from src.mirror_storage import (
    completed_mirror_keys,
    load_mirror_results,
    save_mirror_results,
)
from src.persona_loader import load_personas


DEFAULT_MODEL = "sonnet-4.6"
DEFAULT_OUTPUT_DIR = Path("results/human_mirror_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 85 independent LLM evaluations mirroring the human study "
            "assignment schedule (510 total API calls by default)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project config.yaml).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Single model key to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for experiment results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help="Path to human assignment CSV (built from study data if missing).",
    )
    parser.add_argument(
        "--participants",
        type=int,
        default=85,
        help="Number of virtual participants to simulate (default: 85).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned evaluations without calling the API.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and re-run all evaluations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only the first participant (6 API calls).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _records_to_results(records: list[dict]) -> list:
    from src.models import EvaluationResult

    return [
        EvaluationResult(
            persona_id=record["persona_id"],
            model_key=record["model_key"],
            model_display_name=record["model_display_name"],
            model_id=record["model_id"],
            run_index=record["run_index"],
            timestamp=record["timestamp"],
            prompt_text=record["prompt_text"],
            response_raw=record.get("response_raw", ""),
            parsed=record.get("parsed"),
            latency_ms=record.get("latency_ms", 0),
            error=record.get("error"),
        )
        for record in records
    ]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    model = next((m for m in config.models if m.key == args.model), None)
    if model is None:
        available = ", ".join(m.key for m in config.models)
        raise SystemExit(f"Unknown model {args.model!r}. Available: {available}")

    config = replace(config, models=[model], runs_per_model=1)

    schedule = load_mirror_schedule(args.assignments)
    if args.test:
        first_pid = schedule.iloc[0]["participant_id"]
        schedule = schedule[schedule["participant_id"] == first_pid].copy()
        logging.info("Test mode: first participant only (%s rows)", len(schedule))
    elif args.participants < 85:
        participant_ids = schedule["participant_id"].drop_duplicates().head(args.participants)
        schedule = schedule[schedule["participant_id"].isin(participant_ids)].copy()
        logging.info("Limited to %s participant(s) (%s rows)", args.participants, len(schedule))

    personas = load_personas(config.personas_csv, images_dir=config.images_dir)
    persona_by_id = {persona.persona_id: persona for persona in personas}

    missing_personas = sorted(set(schedule["persona_id"]) - set(persona_by_id))
    if missing_personas:
        raise SystemExit(f"Personas missing from CSV: {missing_personas}")

    schedule_rows = schedule.to_dict(orient="records")
    total_calls = len(schedule_rows)

    logging.info(
        "Human-mirror experiment: model=%s (%s), %s evaluation(s)",
        model.key,
        model.display_name,
        total_calls,
    )
    logging.info(
        "Assignment balance: %s stereo + %s non-stereo per participant",
        schedule.groupby("participant_id")["condition"]
        .apply(lambda values: (values == "stereo").sum())
        .iloc[0],
        schedule.groupby("participant_id")["condition"]
        .apply(lambda values: (values == "non_stereo").sum())
        .iloc[0],
    )

    if args.dry_run:
        for row in schedule_rows:
            logging.info(
                "Would evaluate participant %s slot %s: %s (%s)",
                row["participant_index"],
                row["slot_index"],
                row["persona_id"],
                row["condition"],
            )
        logging.info("Dry run complete: %s API call(s) planned", total_calls)
        return 0

    existing_records = [] if args.no_resume else load_mirror_results(args.output_dir)
    completed = completed_mirror_keys(existing_records)
    if completed:
        logging.info("Resuming with %s completed evaluation(s)", len(completed))

    evaluator = PersonaEvaluator(config)
    results = _records_to_results(existing_records)
    schedule_for_save = [
        {
            "participant_id": record["participant_id"],
            "participant_index": record["participant_index"],
            "slot_index": record["slot_index"],
            "condition": record["condition"],
            "evaluation_index": record["evaluation_index"],
        }
        for record in existing_records
    ]

    failures = 0
    for row in schedule_rows:
        key = (row["participant_id"], row["persona_id"])
        if key in completed:
            logging.info(
                "Skipping participant %s / %s (already completed)",
                row["participant_index"],
                row["persona_id"],
            )
            continue

        persona = persona_by_id[row["persona_id"]]
        logging.info(
            "Evaluating participant %s/%s slot %s: %s (%s) [eval %s/%s]",
            row["participant_index"],
            schedule["participant_id"].nunique(),
            row["slot_index"],
            row["persona_id"],
            row["condition"],
            row["evaluation_index"],
            total_calls,
        )

        result = evaluator.evaluate_once(
            persona=persona,
            model_key=model.key,
            run_index=int(row["evaluation_index"]),
        )
        if result.error:
            failures += 1

        results.append(result)
        schedule_for_save.append(row)
        save_mirror_results(
            args.output_dir,
            schedule_for_save,
            results,
            model_key=model.key,
            model_display_name=model.display_name,
        )

    logging.info(
        "Finished human-mirror experiment: %s evaluation(s), %s failure(s). Saved to %s",
        len(results),
        failures,
        args.output_dir,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
