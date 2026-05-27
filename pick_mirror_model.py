#!/usr/bin/env python3
"""Rank LLMs by alignment with human stereotype-rating distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from export_analysis import load_all_runs, load_personas, yes_no_to_binary
from src.human_data import load_human_study

DEFAULT_MODEL = "gpt-5.4"
HUMAN_DESC_STEREO = 0.835
HUMAN_DESC_NON = 0.431
HUMAN_IMG_STEREO = 0.365
HUMAN_IMG_NON = 0.510


def mirror_output_dir(model_key: str) -> Path:
    return Path("results") / f"human_mirror_{model_key}"


def score_model(runs: pd.DataFrame, human_persona: pd.DataFrame) -> dict:
    stereo = runs[runs["is_stereo"]]
    non_stereo = runs[~runs["is_stereo"]]

    desc_stereo = float(stereo["desc_yes"].mean())
    desc_non = float(non_stereo["desc_yes"].mean())
    img_stereo = float(stereo["img_yes"].mean())
    img_non = float(non_stereo["img_yes"].mean())

    desc_gap = desc_stereo - desc_non
    human_gap = HUMAN_DESC_STEREO - HUMAN_DESC_NON

    mae = float(
        np.mean(
            [
                abs(desc_stereo - HUMAN_DESC_STEREO),
                abs(desc_non - HUMAN_DESC_NON),
                abs(img_stereo - HUMAN_IMG_STEREO),
                abs(img_non - HUMAN_IMG_NON),
            ]
        )
    )

    human_rates = human_persona.set_index("persona_id")["human_description_yes_rate"]
    model_rates = runs.groupby("persona_id")["desc_yes"].mean()
    merged = pd.concat([human_rates, model_rates], axis=1, join="inner").dropna()
    persona_r = float(merged.iloc[:, 0].corr(merged.iloc[:, 1])) if merged.iloc[:, 1].nunique() > 1 else float("nan")

    has_desc_variation = (stereo["desc_yes"].nunique() > 1) or (non_stereo["desc_yes"].nunique() > 1)
    has_img_variation = runs["img_yes"].nunique() > 1
    non_stereo_no_rate = float((non_stereo["desc_yes"] == 0).mean())

    # Prefer models with both Yes/No, reasonable human gap direction, low MAE.
    variation_bonus = 0.0
    if has_desc_variation:
        variation_bonus -= 0.05
    if has_img_variation:
        variation_bonus -= 0.03
    if desc_non >= 0.99:
        variation_bonus += 0.25
    if desc_gap < 0.10:
        variation_bonus += 0.20

    return {
        "model": runs["model_key"].iloc[0],
        "desc_stereo": desc_stereo,
        "desc_non": desc_non,
        "desc_gap_pp": desc_gap * 100,
        "human_desc_gap_pp": human_gap * 100,
        "img_stereo": img_stereo,
        "img_non": img_non,
        "mae_vs_human": mae,
        "persona_desc_r": persona_r,
        "non_stereo_desc_no_rate": non_stereo_no_rate,
        "score": mae + variation_bonus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a model for the human-mirror experiment.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--personas", type=Path, default=Path("data/personas.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_keys = set(config["models"])
    runs_per_model = int(config["runs_per_model"])

    personas = load_personas(args.personas)
    runs = load_all_runs(args.results_dir, personas, model_keys)
    runs = runs[runs["run_index"] <= runs_per_model].copy()
    runs["desc_yes"] = runs["description_stereotype_answer"].map(yes_no_to_binary)
    runs["img_yes"] = runs["image_stereotype_answer"].map(yes_no_to_binary)
    runs["is_stereo"] = runs["is_stereo"].astype(str).str.upper().eq("TRUE")

    _, human_persona = load_human_study()
    rows = [score_model(group, human_persona) for _, group in runs.groupby("model_key", sort=True)]
    ranked = pd.DataFrame(rows).sort_values("score")

    print("=" * 78)
    print("MIRROR MODEL RECOMMENDATION (vs human Prolific study)")
    print("=" * 78)
    print(f"Human targets: desc stereo {HUMAN_DESC_STEREO:.1%}, non-stereo {HUMAN_DESC_NON:.1%}; "
          f"gap {HUMAN_DESC_STEREO - HUMAN_DESC_NON:.1%}")
    print(f"               img  stereo {HUMAN_IMG_STEREO:.1%}, non-stereo {HUMAN_IMG_NON:.1%}")
    print()
    print(ranked.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    best = ranked.iloc[0]
    print(f"Recommended: {best['model']}")
    print(f"  Output dir: {mirror_output_dir(best['model'])}")
    print(f"  Desc yes-rates: stereo {best['desc_stereo']:.1%}, non-stereo {best['desc_non']:.1%} "
          f"(human gap {best['human_desc_gap_pp']:.1f} pp vs model {best['desc_gap_pp']:.1f} pp)")
    print(f"  Image yes-rates: stereo {best['img_stereo']:.1%}, non-stereo {best['img_non']:.1%}")
    print(f"  Non-stereo description 'No' rate: {best['non_stereo_desc_no_rate']:.1%}")
    print()
    print("Run mirror experiment:")
    print(f"  python run_human_mirror.py --model {best['model']}")
    print(f"  python run_human_mirror.py --test --model {best['model']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
