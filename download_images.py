#!/usr/bin/env python3
"""Download persona images from CSV without running LLM evaluations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.download_images import download_images_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download persona images from HTTP URLs in the CSV (no LLM calls)."
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
        help="Download only specific persona_id(s). Can be repeated.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download images from URLs even if already cached locally.",
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
    persona_ids = set(args.persona_ids) if args.persona_ids else None

    results = download_images_from_csv(
        config.personas_csv,
        images_dir=config.images_dir,
        refresh=args.refresh,
        persona_ids=persona_ids,
    )

    counts: dict[str, int] = {}
    errors = 0
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
        if item.status in {"downloaded", "cached"}:
            logging.info(
                "[%s] %s -> %s",
                item.status,
                item.persona_id,
                item.local_path,
            )
        elif item.status == "local":
            logging.info(
                "[local] %s already on disk at %s",
                item.persona_id,
                item.local_path,
            )
        elif item.status == "missing":
            logging.warning(
                "[missing] %s — %s",
                item.persona_id,
                item.error,
            )
            errors += 1
        else:
            logging.error(
                "[error] %s — %s",
                item.persona_id,
                item.error,
            )
            errors += 1

    logging.info(
        "Done. downloaded=%s cached=%s local=%s missing=%s error=%s",
        counts.get("downloaded", 0),
        counts.get("cached", 0),
        counts.get("local", 0),
        counts.get("missing", 0),
        counts.get("error", 0),
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
