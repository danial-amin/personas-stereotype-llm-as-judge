#!/usr/bin/env python3
"""Re-parse saved evaluation JSON from response_raw without calling LLMs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Persona
from src.storage import (
    load_existing_evaluations,
    reparse_evaluation,
    save_aggregate_json,
    save_persona_csv,
    save_persona_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-parse response_raw fields in saved evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Results directory (default: results/)",
    )
    parser.add_argument(
        "--persona-id",
        action="append",
        dest="persona_ids",
        help="Re-parse only specific persona_id(s). Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    persona_dirs = sorted(
        p for p in args.results_dir.iterdir() if p.is_dir() and (p / "evaluations.json").exists()
    )
    if args.persona_ids:
        allowed = set(args.persona_ids)
        persona_dirs = [p for p in persona_dirs if p.name in allowed]

    if not persona_dirs:
        logging.warning("No evaluation results found to re-parse.")
        return 0

    all_evaluations = {}
    personas: list[Persona] = []
    fixed = 0
    still_failing = 0

    for persona_dir in persona_dirs:
        json_path = persona_dir / "evaluations.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        persona = Persona(**data["persona"])
        personas.append(persona)

        evaluations = load_existing_evaluations(persona_dir)
        reparsed = []
        for item in evaluations:
            before = item.parsed is not None
            updated = reparse_evaluation(item)
            after = updated.parsed is not None
            if not before and after:
                fixed += 1
            if updated.error and "parse" in updated.error.lower():
                still_failing += 1
            reparsed.append(updated)

        save_persona_json(persona_dir, persona, reparsed)
        save_persona_csv(persona_dir, reparsed)
        all_evaluations[persona.persona_id] = reparsed
        logging.info("Re-parsed %s (%s evaluations)", persona.persona_id, len(reparsed))

    save_aggregate_json(args.results_dir, personas, all_evaluations)
    logging.info(
        "Done. newly parsed=%s, still failing=%s, personas=%s",
        fixed,
        still_failing,
        len(persona_dirs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
