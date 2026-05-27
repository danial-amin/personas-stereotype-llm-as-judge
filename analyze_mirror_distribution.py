#!/usr/bin/env python3
"""Compare human vs human-mirror LLM rating distributions (paired design)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from src.human_data import LIKERT_ORDER, likert_to_numeric, load_human_study
from src.mirror_storage import load_mirror_results_successful

from pick_mirror_model import DEFAULT_MODEL, mirror_output_dir

DEFAULT_MIRROR_DIR = mirror_output_dir(DEFAULT_MODEL)
DEFAULT_OUTPUT_DIR = Path("outputs")


def _yes_rate(series: pd.Series) -> float:
    values = series.astype(str).str.strip().str.lower()
    return float((values == "yes").mean())


def _likert_yes_rate(series: pd.Series) -> float:
    values = series.astype(str).str.strip()
    return float(values.isin(["Somewhat agree", "Strongly agree"]).mean())


def _likert_mean(series: pd.Series) -> float:
    values = series.map(likert_to_numeric).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float | None:
    a = y1.astype(int)
    b = y2.astype(int)
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    p_a = float(a.mean())
    p_b = float(b.mean())
    pe = p_a * p_b + (1 - p_a) * (1 - p_b)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return round((po - pe) / (1 - pe), 4)


def _rate_correlation(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return None, None
    r, p = pearsonr(clean["x"], clean["y"])
    return round(float(r), 4), round(float(p), 4)


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
                "participant_index": record["participant_index"],
                "persona_id": record["persona_id"],
                "condition": record["condition"],
                "image_answer": image.get("answer", ""),
                "description_answer": description.get("answer", ""),
                "image_yes": 1.0 if str(image.get("answer", "")).strip().lower() == "yes" else 0.0,
                "description_yes": 1.0
                if str(description.get("answer", "")).strip().lower() == "yes"
                else 0.0,
                "usefulness_answer": understand.get("answer", ""),
                "relatable_answer": relatable.get("answer", ""),
            }
        )
    return pd.DataFrame(rows)


def _condition_summary(long_df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for label, condition in [("Stereotyped (TRUE)", "stereo"), ("Non-stereotyped (FALSE)", "non_stereo")]:
        subset = long_df[long_df["condition"] == condition]
        gt = condition == "stereo"
        rows.append(
            {
                "source": source,
                "group": label,
                "n_rater_persona_pairs": len(subset),
                "image_yes_rate": round(_yes_rate(subset["image_answer"]), 4),
                "description_yes_rate": round(_yes_rate(subset["description_answer"]), 4),
                "image_ground_truth_agreement": round(float(((subset["image_yes"] == 1) == gt).mean()), 4),
                "description_ground_truth_agreement": round(
                    float(((subset["description_yes"] == 1) == gt).mean()), 4
                ),
                "usefulness_agree_rate": round(_likert_yes_rate(subset["usefulness_answer"]), 4),
                "relatable_agree_rate": round(_likert_yes_rate(subset["relatable_answer"]), 4),
                "usefulness_mean_likert": round(_likert_mean(subset["usefulness_answer"]), 4),
                "relatable_mean_likert": round(_likert_mean(subset["relatable_answer"]), 4),
            }
        )
    return pd.DataFrame(rows)


def _per_persona_summary(long_df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for persona_id, group in long_df.groupby("persona_id", sort=True):
        rows.append(
            {
                "source": source,
                "persona_id": persona_id,
                "condition": group["condition"].iloc[0],
                "n_raters": len(group),
                "image_yes_rate": round(_yes_rate(group["image_answer"]), 4),
                "description_yes_rate": round(_yes_rate(group["description_answer"]), 4),
                "usefulness_mean_likert": round(_likert_mean(group["usefulness_answer"]), 4),
                "relatable_mean_likert": round(_likert_mean(group["relatable_answer"]), 4),
            }
        )
    return pd.DataFrame(rows)


def _paired_agreement(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, condition, human_col, mirror_col in [
        ("detection", "stereo", "human_image_yes", "mirror_image_yes"),
        ("detection", "stereo", "human_description_yes", "mirror_description_yes"),
        ("perception", "non_stereo", "human_image_yes", "mirror_image_yes"),
        ("perception", "non_stereo", "human_description_yes", "mirror_description_yes"),
    ]:
        subset = paired[paired["condition"] == condition]
        rows.append(
            {
                "task": task,
                "modality": human_col.replace("human_", "").replace("_yes", ""),
                "n_pairs": len(subset),
                "exact_agreement_rate": round(float((subset[human_col] == subset[mirror_col]).mean()), 4),
                "cohens_kappa": cohens_kappa(subset[human_col], subset[mirror_col]),
                "human_yes_rate": round(float(subset[human_col].mean()), 4),
                "mirror_yes_rate": round(float(subset[mirror_col].mean()), 4),
                "delta_mirror_minus_human": round(float(subset[mirror_col].mean() - subset[human_col].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _paired_persona_agreement(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for persona_id, group in paired.groupby("persona_id", sort=True):
        rows.append(
            {
                "persona_id": persona_id,
                "condition": group["condition"].iloc[0],
                "n_pairs": len(group),
                "human_image_yes_rate": round(float(group["human_image_yes"].mean()), 4),
                "mirror_image_yes_rate": round(float(group["mirror_image_yes"].mean()), 4),
                "image_rate_delta": round(
                    float(group["mirror_image_yes"].mean() - group["human_image_yes"].mean()), 4
                ),
                "human_description_yes_rate": round(float(group["human_description_yes"].mean()), 4),
                "mirror_description_yes_rate": round(float(group["mirror_description_yes"].mean()), 4),
                "description_rate_delta": round(
                    float(group["mirror_description_yes"].mean() - group["human_description_yes"].mean()), 4
                ),
                "image_exact_agreement": round(float((group["human_image_yes"] == group["mirror_image_yes"]).mean()), 4),
                "description_exact_agreement": round(
                    float((group["human_description_yes"] == group["mirror_description_yes"]).mean()), 4
                ),
            }
        )

    detail = pd.DataFrame(rows)
    summary_rows = []
    for modality, hum_col, mir_col in [
        ("image", "human_image_yes_rate", "mirror_image_yes_rate"),
        ("description", "human_description_yes_rate", "mirror_description_yes_rate"),
    ]:
        stereo = detail[detail["condition"] == "stereo"]
        non_stereo = detail[detail["condition"] == "non_stereo"]
        r_all, p_all = _rate_correlation(detail[hum_col], detail[mir_col])
        r_st, p_st = _rate_correlation(stereo[hum_col], stereo[mir_col])
        r_ns, p_ns = _rate_correlation(non_stereo[hum_col], non_stereo[mir_col])
        summary_rows.append(
            {
                "modality": modality,
                "pearson_r_all_personas": r_all,
                "pearson_p_all_personas": p_all,
                "pearson_r_stereo_only": r_st,
                "pearson_p_stereo_only": p_st,
                "pearson_r_non_stereo_only": r_ns,
                "pearson_p_non_stereo_only": p_ns,
                "mean_abs_rate_delta": round(float(detail[f"{modality}_rate_delta"].abs().mean()), 4),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _format_report(
    *,
    model_name: str,
    human_long: pd.DataFrame,
    mirror_long: pd.DataFrame,
    paired: pd.DataFrame,
    group_summary: pd.DataFrame,
    per_persona: pd.DataFrame,
    paired_agreement: pd.DataFrame,
    persona_agreement: pd.DataFrame,
    persona_corr: pd.DataFrame,
) -> str:
    lines = [
        "=" * 78,
        f"HUMAN vs HUMAN-MIRROR LLM COMPARISON ({model_name}, paired design)",
        "=" * 78,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model: {model_name}",
        f"Human rows:  {len(human_long)}",
        f"Mirror rows: {len(mirror_long)} (deduped successful evaluations)",
        f"Paired rows: {len(paired)}",
        "",
        "Each virtual participant saw the same 6 personas as their Prolific counterpart.",
        "",
        "-" * 78,
        "GROUP-LEVEL YES-RATES (mirror minus human in delta column)",
        "-" * 78,
        group_summary.to_string(index=False),
        "",
        "-" * 78,
        "PAIRED AGREEMENT (same participant x persona, 510 pairs total)",
        "-" * 78,
        paired_agreement.to_string(index=False),
        "",
        "-" * 78,
        "PER-PERSONA YES-RATES",
        "-" * 78,
        persona_agreement.to_string(index=False),
        "",
        "-" * 78,
        "PER-PERSONA RATE CORRELATIONS (human vs mirror yes-rates across personas)",
        "-" * 78,
        persona_corr.to_string(index=False),
        "",
        "-" * 78,
        "KEY TAKEAWAYS",
        "-" * 78,
    ]

    stereo_human = group_summary[(group_summary["source"] == "human") & (group_summary["group"].str.contains("TRUE"))]
    stereo_mirror = group_summary[(group_summary["source"] == "mirror") & (group_summary["group"].str.contains("TRUE"))]
    non_human = group_summary[(group_summary["source"] == "human") & (group_summary["group"].str.contains("FALSE"))]
    non_mirror = group_summary[(group_summary["source"] == "mirror") & (group_summary["group"].str.contains("FALSE"))]

    if not stereo_human.empty and not stereo_mirror.empty:
        h_img = float(stereo_human.iloc[0]["image_yes_rate"])
        m_img = float(stereo_mirror.iloc[0]["image_yes_rate"])
        lines.append(
            f"Detection (stereo image): human {h_img:.1%}, mirror {m_img:.1%}, delta {m_img - h_img:+.1%}"
        )
    if not non_human.empty and not non_mirror.empty:
        h_img = float(non_human.iloc[0]["image_yes_rate"])
        m_img = float(non_mirror.iloc[0]["image_yes_rate"])
        lines.append(
            f"Perception (non-stereo image FP): human {h_img:.1%}, mirror {m_img:.1%}, delta {m_img - h_img:+.1%}"
        )

    det_image = paired_agreement[
        (paired_agreement["task"] == "detection") & (paired_agreement["modality"] == "image")
    ]
    if not det_image.empty:
        row = det_image.iloc[0]
        lines.append(
            f"Paired detection image agreement: {row['exact_agreement_rate']:.1%}, "
            f"kappa={row['cohens_kappa']}"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human and mirror LLM distributions.")
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=DEFAULT_MIRROR_DIR,
        help=f"Human-mirror experiment results directory (default: {DEFAULT_MIRROR_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for comparison outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    human_long, human_persona_summary = load_human_study()
    human_long = human_long.rename(columns={"pid": "participant_id"})
    mirror_records = load_mirror_results_successful(args.mirror_dir)
    if not mirror_records:
        raise SystemExit(f"No mirror results found in {args.mirror_dir}")

    if len(mirror_records) != 510:
        raise SystemExit(
            f"Expected 510 successful mirror evaluations, found {len(mirror_records)}. "
            "Finish or repair the experiment before comparing."
        )

    mirror_long = _mirror_to_long(mirror_records)
    model_name = mirror_records[0].get("model_display_name", "unknown")

    human_group = _condition_summary(human_long, "human")
    mirror_group = _condition_summary(mirror_long, "mirror")
    group_summary = pd.concat([human_group, mirror_group], ignore_index=True)
    group_summary["delta_vs_human"] = None
    for group in ["Stereotyped (TRUE)", "Non-stereotyped (FALSE)"]:
        h = human_group[human_group["group"] == group].iloc[0]
        m = mirror_group[mirror_group["group"] == group].iloc[0]
        idx = group_summary[(group_summary["source"] == "mirror") & (group_summary["group"] == group)].index
        for col in ["image_yes_rate", "description_yes_rate", "usefulness_agree_rate", "relatable_agree_rate"]:
            group_summary.loc[idx, f"mirror_minus_human_{col}"] = round(float(m[col] - h[col]), 4)

    human_persona = _per_persona_summary(human_long, "human")
    mirror_persona = _per_persona_summary(mirror_long, "mirror")
    per_persona = human_persona.merge(
        mirror_persona,
        on="persona_id",
        suffixes=("_human", "_mirror"),
    )
    per_persona["image_rate_delta"] = per_persona["image_yes_rate_mirror"] - per_persona["image_yes_rate_human"]
    per_persona["description_rate_delta"] = (
        per_persona["description_yes_rate_mirror"] - per_persona["description_yes_rate_human"]
    )

    paired = human_long.merge(
        mirror_long,
        on=["participant_id", "persona_id", "condition"],
        suffixes=("_human", "_mirror"),
    )
    paired = paired.rename(
        columns={
            "image_yes_human": "human_image_yes",
            "description_yes_human": "human_description_yes",
            "image_yes_mirror": "mirror_image_yes",
            "description_yes_mirror": "mirror_description_yes",
        }
    )

    paired_agreement = _paired_agreement(paired)
    persona_agreement, persona_corr = _paired_persona_agreement(paired)

    report = _format_report(
        model_name=model_name,
        human_long=human_long,
        mirror_long=mirror_long,
        paired=paired,
        group_summary=group_summary,
        per_persona=per_persona,
        paired_agreement=paired_agreement,
        persona_agreement=persona_agreement,
        persona_corr=persona_corr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.output_dir / "mirror_human_comparison.txt"
    xlsx_path = args.output_dir / "mirror_human_comparison.xlsx"
    txt_path.write_text(report, encoding="utf-8")

    paired_export_cols = [
        "participant_id",
        "persona_id",
        "condition",
        "human_image_yes",
        "mirror_image_yes",
        "human_description_yes",
        "mirror_description_yes",
        "image_answer_human",
        "image_answer_mirror",
        "description_answer_human",
        "description_answer_mirror",
        "usefulness_answer_human",
        "usefulness_answer_mirror",
        "relatable_answer_human",
        "relatable_answer_mirror",
    ]
    if "participant_index_mirror" in paired.columns:
        paired_export_cols.insert(1, "participant_index_mirror")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        group_summary.to_excel(writer, sheet_name="Group Summary", index=False)
        paired_agreement.to_excel(writer, sheet_name="Paired Agreement", index=False)
        persona_agreement.to_excel(writer, sheet_name="Per Persona", index=False)
        persona_corr.to_excel(writer, sheet_name="Persona Correlations", index=False)
        per_persona.to_excel(writer, sheet_name="Persona Side by Side", index=False)
        paired[paired_export_cols].to_excel(writer, sheet_name="All Paired Rows", index=False)

    print(report)
    print()
    print(f"Saved: {txt_path}")
    print(f"Saved: {xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
