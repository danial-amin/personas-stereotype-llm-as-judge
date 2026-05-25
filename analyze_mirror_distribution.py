#!/usr/bin/env python3
"""Compare human vs human-mirror LLM rating distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.human_data import load_human_study
from src.mirror_storage import load_mirror_results


DEFAULT_MIRROR_DIR = Path("results/human_mirror_experiment")


def _yes_rate(series: pd.Series) -> float:
    values = series.astype(str).str.strip().str.lower()
    return (values == "yes").mean()


def _likert_yes_rate(series: pd.Series) -> float:
    values = series.astype(str).str.strip()
    agree = values.isin(["Somewhat agree", "Strongly agree"])
    return agree.mean()


def _mirror_to_long(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        parsed = record.get("parsed") or {}
        image = parsed.get("image_stereotype") or {}
        description = parsed.get("persona_description_stereotype") or {}
        understand = parsed.get("this_persona_helped_me_understand_this_group_of_people") or {}
        relatable = parsed.get("i_find_this_persona_relatable") or {}

        rows.append(
            {
                "participant_id": record["participant_id"],
                "persona_id": record["persona_id"],
                "condition": record["condition"],
                "image_str": image.get("answer", ""),
                "desc_str": description.get("answer", ""),
                "understand": understand.get("answer", ""),
                "relatable": relatable.get("answer", ""),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human and mirror LLM distributions.")
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=DEFAULT_MIRROR_DIR,
        help=f"Human-mirror experiment results directory (default: {DEFAULT_MIRROR_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    human_long, _ = load_human_study()
    mirror_records = load_mirror_results(args.mirror_dir)
    if not mirror_records:
        raise SystemExit(f"No mirror results found in {args.mirror_dir}")

    mirror_long = _mirror_to_long(mirror_records)

    metrics = [
        ("image stereotype", "image_answer", "image_str"),
        ("description stereotype", "description_answer", "desc_str"),
        ("understand group (agree+)", "usefulness_answer", "understand"),
        ("relatable (agree+)", "relatable_answer", "relatable"),
    ]

    print("=" * 72)
    print("HUMAN vs HUMAN-MIRROR DISTRIBUTION COMPARISON")
    print("=" * 72)
    print(f"Human rows:  {len(human_long)}")
    print(f"Mirror rows: {len(mirror_long)}")
    print()

    for label, human_col, mirror_col in metrics:
        print(f"--- {label} ---")
        for condition in ["stereo", "non_stereo"]:
            human_subset = human_long[human_long["condition"] == condition]
            mirror_subset = mirror_long[mirror_long["condition"] == condition]
            if human_col in {"usefulness_answer", "relatable_answer"}:
                human_rate = _likert_yes_rate(human_subset[human_col])
                mirror_rate = _likert_yes_rate(mirror_subset[mirror_col])
            else:
                human_rate = _yes_rate(human_subset[human_col])
                mirror_rate = _yes_rate(mirror_subset[mirror_col])
            print(
                f"  {condition:12s}  human={human_rate:.3f}  mirror={mirror_rate:.3f}  "
                f"delta={mirror_rate - human_rate:+.3f}"
            )
        print()

    print("--- per persona (image stereotype yes-rate) ---")
    persona_ids = sorted(set(human_long["persona_id"]) | set(mirror_long["persona_id"]))
    comparison = []
    for persona_id in persona_ids:
        human_subset = human_long[human_long["persona_id"] == persona_id]
        mirror_subset = mirror_long[mirror_long["persona_id"] == persona_id]
        comparison.append(
            {
                "persona_id": persona_id,
                "condition": human_subset["condition"].iloc[0] if len(human_subset) else "",
                "human_n": len(human_subset),
                "mirror_n": len(mirror_subset),
                "human_yes_rate": _yes_rate(human_subset["image_answer"]),
                "mirror_yes_rate": _yes_rate(mirror_subset["image_str"]),
            }
        )

    comparison_df = pd.DataFrame(comparison)
    comparison_df["delta"] = comparison_df["mirror_yes_rate"] - comparison_df["human_yes_rate"]
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
