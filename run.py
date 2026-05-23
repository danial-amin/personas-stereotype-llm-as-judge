#!/usr/bin/env python3
"""Run LLM-as-judge stereotype evaluations for all personas."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluator import PersonaEvaluator
from src.persona_loader import load_personas
from src.storage import (
    load_existing_evaluations,
    persona_output_dir,
    save_aggregate_json,
    save_persona_csv,
    save_persona_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate personas for stereotypes using 5 LLMs, 5 runs each."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    parser.add_argument(
        "--persona-id",
        action="append",
        dest="persona_ids",
        help="Run only specific persona_id(s). Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_keys",
        help="Run only specific model key(s) from config.yaml. Can be repeated.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and re-run everything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned work without calling APIs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    personas = load_personas(config.personas_csv, validate_images=not args.dry_run)

    if args.persona_ids:
        allowed = set(args.persona_ids)
        personas = [p for p in personas if p.persona_id in allowed]
        if not personas:
            raise SystemExit(f"No personas matched: {args.persona_ids}")

    if args.model_keys:
        allowed_models = set(args.model_keys)
        config.models = [m for m in config.models if m.key in allowed_models]
        if not config.models:
            raise SystemExit(f"No models matched: {args.model_keys}")

    total_runs = len(personas) * len(config.models) * config.runs_per_model
    logging.info(
        "Loaded %s persona(s), %s model(s), %s run(s) each => %s total API calls",
        len(personas),
        len(config.models),
        config.runs_per_model,
        total_runs,
    )

    if args.dry_run:
        for persona in personas:
            logging.info("Would evaluate: %s (%s)", persona.persona_id, persona.name)
        return 0

    evaluator = PersonaEvaluator(config)
    all_evaluations: dict[str, list] = {}

    for persona in personas:
        output_dir = persona_output_dir(config.results_dir, persona.persona_id)
        existing = [] if args.no_resume else load_existing_evaluations(output_dir)

        def on_result(results_so_far: list) -> None:
            save_persona_json(output_dir, persona, results_so_far)
            save_persona_csv(output_dir, results_so_far)

        results = evaluator.evaluate_persona(
            persona,
            existing=existing,
            on_result=on_result,
        )
        all_evaluations[persona.persona_id] = results
        logging.info("Finished persona %s (%s evaluations)", persona.persona_id, len(results))

    aggregate_path = save_aggregate_json(config.results_dir, personas, all_evaluations)
    logging.info("Saved aggregate results to %s", aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
