#!/usr/bin/env python3
"""Run LLM-as-judge stereotype evaluations for all personas."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluator import PersonaEvaluator
from src.persona_loader import load_personas
from src.image_preprocessor import is_http_url
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
        "--refresh-images",
        action="store_true",
        help="Re-download persona images from HTTP URLs even if cached locally.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test: 1 persona, all models, 1 run each.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs per model (default: from config.yaml).",
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


def _print_test_summary(persona, results: list, output_dir: Path) -> int:
    print("\n" + "=" * 60)
    print("TEST RUN SUMMARY")
    print("=" * 60)
    print(f"Persona:  {persona.persona_id} ({persona.name})")
    print(f"Models:   {len(results)} evaluation(s)")
    print()

    failures = 0
    for item in results:
        print(f"--- {item.model_display_name} ({item.model_id}) run {item.run_index} ---")
        print(f"Latency:  {item.latency_ms} ms")
        if item.error:
            failures += 1
            print("Status:   FAILED")
            print(f"Error:    {item.error}")
        else:
            print("Status:   OK")
            if item.parsed:
                image_answer = item.parsed.get("image_stereotype", {}).get("answer", "?")
                desc_answer = item.parsed.get("persona_description_stereotype", {}).get("answer", "?")
                print(f"Parsed:   image_stereotype={image_answer}, description_stereotype={desc_answer}")
            else:
                print("Parsed:   (could not parse JSON from response)")
                print(f"Raw:      {item.response_raw[:300]}...")
        print()

    print(f"Saved to: {output_dir / 'evaluations.json'}")
    print("=" * 60 + "\n")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    personas = load_personas(
        config.personas_csv,
        images_dir=config.images_dir,
        validate_images=not args.dry_run,
        refresh_images=args.refresh_images,
    )

    if args.persona_ids:
        allowed = set(args.persona_ids)
        personas = [p for p in personas if p.persona_id in allowed]
        if not personas:
            raise SystemExit(f"No personas matched: {args.persona_ids}")

    if args.model_keys:
        allowed_models = set(args.model_keys)
        filtered_models = [m for m in config.models if m.key in allowed_models]
        if not filtered_models:
            raise SystemExit(f"No models matched: {args.model_keys}")
        config = replace(config, models=filtered_models)

    if args.test:
        if not personas:
            raise SystemExit("No personas loaded for test run.")
        personas = [personas[0]]
        config = replace(config, runs_per_model=1, request_delay_seconds=0)
        logging.info(
            "Test mode: %s API call(s) — persona=%s, models=%s",
            len(config.models),
            personas[0].persona_id,
            ", ".join(m.display_name for m in config.models),
        )

    if args.runs is not None:
        config = replace(config, runs_per_model=args.runs)

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
            source = persona.image_source or persona.image_path
            if is_http_url(source):
                logging.info(
                    "Would evaluate: %s (%s) — would download image from %s",
                    persona.persona_id,
                    persona.name,
                    source,
                )
            else:
                logging.info("Would evaluate: %s (%s)", persona.persona_id, persona.name)
        return 0

    evaluator = PersonaEvaluator(config)
    all_evaluations: dict[str, list] = {}

    for persona in personas:
        output_dir = persona_output_dir(config.results_dir, persona.persona_id)
        existing = [] if (args.no_resume or args.test) else load_existing_evaluations(output_dir)

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

        if args.test:
            return _print_test_summary(persona, results, output_dir)

    aggregate_path = save_aggregate_json(config.results_dir, personas, all_evaluations)
    logging.info("Saved aggregate results to %s", aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
