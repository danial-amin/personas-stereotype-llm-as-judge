#!/usr/bin/env python3
"""Combine LLM evaluation results into a single CSV and multi-sheet Excel analysis."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd
import pingouin as pg
import yaml
from scipy.stats import pearsonr

from src.csv_utils import detect_csv_encoding
from src.human_data import (
    DEFAULT_HUMAN_CSV_URL,
    build_human_stereo_comparison,
    likert_to_numeric,
    load_human_study,
)

PERSONA_META_COLS = ["name", "age", "gender", "workforce", "is_stereo"]
RUN_COLS = [
    "persona_id",
    "name",
    "age",
    "gender",
    "workforce",
    "is_stereo",
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
]


def load_personas(path: Path) -> pd.DataFrame:
    encoding = detect_csv_encoding(path)
    df = pd.read_csv(path, encoding=encoding)
    df["is_stereo"] = df["is_stereo"].astype(str).str.strip().str.upper().map(
        {"TRUE": True, "FALSE": False, "1": True, "0": False}
    )
    return df


def load_model_keys(config_path: Path) -> set[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return set(config.get("models", {}).keys())


def load_all_runs(
    results_dir: Path,
    personas: pd.DataFrame,
    model_keys: set[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    meta = personas.set_index("persona_id")[PERSONA_META_COLS]

    for csv_path in sorted(glob.glob(str(results_dir / "*/evaluations.csv"))):
        df = pd.read_csv(csv_path)
        if model_keys is not None:
            df = df[df["model_key"].isin(model_keys)]
        df = df.join(meta, on="persona_id", how="left")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No evaluations.csv files found under {results_dir}")

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs["is_stereo"] = all_runs["is_stereo"].map({True: "TRUE", False: "FALSE"})
    return all_runs


def yes_no_to_binary(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "yes":
        return 1.0
    if text == "no":
        return 0.0
    return None


def consistency_stats(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.isin(["Yes", "No"])]
    n = len(clean)
    if n == 0:
        return {
            "valid_runs": 0,
            "yes_count": 0,
            "no_count": 0,
            "mode_answer": None,
            "consistency": None,
            "unanimous": None,
        }

    yes_count = int((clean == "Yes").sum())
    no_count = int((clean == "No").sum())
    mode = "Yes" if yes_count >= no_count else "No"
    mode_count = yes_count if mode == "Yes" else no_count

    return {
        "valid_runs": n,
        "yes_count": yes_count,
        "no_count": no_count,
        "mode_answer": mode,
        "consistency": round(mode_count / n, 4),
        "unanimous": mode_count == n,
    }


def build_model_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = all_runs.groupby(["persona_id", "model_key"], sort=True)
    persona_meta = all_runs.drop_duplicates("persona_id").set_index("persona_id")

    for (persona_id, model_key), group in grouped:
        meta = persona_meta.loc[persona_id]
        image = consistency_stats(group["image_stereotype_answer"])
        desc = consistency_stats(group["description_stereotype_answer"])

        rows.append(
            {
                "persona_id": persona_id,
                "name": meta["name"],
                "age": meta["age"],
                "gender": meta["gender"],
                "workforce": meta["workforce"],
                "is_stereo": meta["is_stereo"],
                "model_key": model_key,
                "model_display_name": group["model_display_name"].iloc[0],
                "image_mode_answer": image["mode_answer"],
                "image_yes_count": image["yes_count"],
                "image_no_count": image["no_count"],
                "image_valid_runs": image["valid_runs"],
                "image_internal_consistency": image["consistency"],
                "image_unanimous": image["unanimous"],
                "description_mode_answer": desc["mode_answer"],
                "description_yes_count": desc["yes_count"],
                "description_no_count": desc["no_count"],
                "description_valid_runs": desc["valid_runs"],
                "description_internal_consistency": desc["consistency"],
                "description_unanimous": desc["unanimous"],
            }
        )

    return pd.DataFrame(rows)


def compute_icc(
    long_df: pd.DataFrame,
    rating_col: str,
    target_col: str = "persona_id",
    rater_col: str = "model_key",
    nan_policy: str = "raise",
) -> dict[str, float | None | str]:
    """ICC(2,1) two-way random, single rater, absolute agreement."""
    clean = long_df.dropna(subset=[rating_col]).copy()
    if clean[target_col].nunique() < 2 or clean[rater_col].nunique() < 2:
        return {"icc2_single": None, "icc2_average": None, "icc3_single": None, "note": "insufficient data"}

    # Zero variance across raters for every target → ICC undefined
    per_target_var = clean.groupby(target_col)[rating_col].var(numeric_only=True)
    if (per_target_var.fillna(0) == 0).all():
        return {
            "icc2_single": None,
            "icc2_average": None,
            "icc3_single": None,
            "note": "undefined — zero variance across raters",
        }

    try:
        icc_table = pg.intraclass_corr(
            data=clean,
            targets=target_col,
            raters=rater_col,
            ratings=rating_col,
            nan_policy=nan_policy,
        )
    except ValueError:
        return {"icc2_single": None, "icc2_average": None, "icc3_single": None, "note": "ICC failed"}

    icc2 = icc_table[icc_table["Type"] == "ICC(A,1)"]
    icc2k = icc_table[icc_table["Type"] == "ICC(A,k)"]
    icc3 = icc_table[icc_table["Type"] == "ICC(C,1)"]

    icc2_val = float(icc2["ICC"].iloc[0]) if not icc2.empty else float("nan")
    return {
        "icc2_single": round(icc2_val, 4) if icc2_val == icc2_val else None,
        "icc2_average": round(float(icc2k["ICC"].iloc[0]), 4) if not icc2k.empty else None,
        "icc3_single": round(float(icc3["ICC"].iloc[0]), 4) if not icc3.empty else None,
        "note": "",
    }


def compute_human_agreement(long_df: pd.DataFrame, rating_col: str) -> dict[str, float | None]:
    """
    Mean within-persona majority agreement among human raters.

    Each Prolific participant rated only 6/12 personas, so classic ICC is undefined.
    """
    per_persona: list[float] = []
    for _, group in long_df.groupby("persona_id"):
        values = group[rating_col].dropna()
        if values.empty:
            continue
        rate = float(values.mean())
        per_persona.append(max(rate, 1 - rate))

    if not per_persona:
        return {"mean_majority_agreement": None}

    return {"mean_majority_agreement": round(float(sum(per_persona) / len(per_persona)), 4)}


def _is_stereo_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin({"TRUE", "1", "YES"})


def build_llm_likert_scores(all_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (persona × model) with mean likert scores across 5 runs."""
    rows: list[dict[str, Any]] = []

    for (persona_id, model_key), group in all_runs.groupby(["persona_id", "model_key"], sort=True):
        rel = group["relatable_answer"].map(likert_to_numeric).dropna()
        # Human _use = "This persona helped me understand this group" → same as LLM understand_group
        use = group["understand_group_answer"].map(likert_to_numeric).dropna()
        rows.append(
            {
                "persona_id": persona_id,
                "model_key": model_key,
                "is_stereo": group["is_stereo"].iloc[0],
                "relatable_mean": round(float(rel.mean()), 4) if len(rel) else None,
                "usefulness_mean": round(float(use.mean()), 4) if len(use) else None,
                "relatable_internal_consistency": round(float(1 - rel.std() / 2), 4) if len(rel) > 1 else None,
                "usefulness_internal_consistency": round(float(1 - use.std() / 2), 4) if len(use) > 1 else None,
            }
        )

    return pd.DataFrame(rows)


def build_stereotype_flag_icc_summary(
    model_summary: pd.DataFrame,
    human_long: pd.DataFrame | None,
    *,
    stereo_personas: bool,
) -> pd.DataFrame:
    """
    ICC for stereotype Yes/No flags.

    detection (a_s_*): did raters detect stereotype when present?
    perception (a_us_*): did raters falsely flag stereotype when absent?
    """
    task = "detection" if stereo_personas else "perception"
    group_label = "a_s_* (stereotype present)" if stereo_personas else "a_us_* (stereotype absent)"
    llm_subset = model_summary[_is_stereo_series(model_summary["is_stereo"]) == stereo_personas].copy()
    llm_subset["image_binary"] = llm_subset["image_mode_answer"].map(yes_no_to_binary)
    llm_subset["description_binary"] = llm_subset["description_mode_answer"].map(yes_no_to_binary)

    perception_note = (
        "False stereotype flag rate — stereotype absent; Yes = over-perception."
        if not stereo_personas
        else "True stereotype detection — stereotype present."
    )

    rows = [
        icc_metric_row(
            task,
            "stereotype_flag",
            "image",
            "LLM",
            icc=compute_icc(llm_subset, "image_binary"),
            notes=f"Inter-LLM ICC, {group_label}. {perception_note}",
        ),
        icc_metric_row(
            task,
            "stereotype_flag",
            "description",
            "LLM",
            icc=compute_icc(llm_subset, "description_binary"),
            notes=f"Inter-LLM ICC, {group_label}. {perception_note}",
        ),
    ]
    for row in rows:
        row["persona_group"] = group_label

    if human_long is not None:
        condition = "stereo" if stereo_personas else "non_stereo"
        human_subset = human_long[human_long["condition"] == condition].copy()
        rows.extend(
            [
                icc_metric_row(
                    task,
                    "stereotype_flag",
                    "image",
                    "Human",
                    agreement=compute_human_agreement(human_subset, "image_yes"),
                    notes=f"Mean within-persona majority agreement, {group_label}.",
                ),
                icc_metric_row(
                    task,
                    "stereotype_flag",
                    "description",
                    "Human",
                    agreement=compute_human_agreement(human_subset, "description_yes"),
                    notes=f"Mean within-persona majority agreement, {group_label}.",
                ),
            ]
        )
        for row in rows[-2:]:
            row["persona_group"] = group_label

    return pd.DataFrame(rows)


def build_stereotype_flag_persona_table(
    persona_summary: pd.DataFrame,
    human_persona: pd.DataFrame | None,
    icc_summary: pd.DataFrame,
    *,
    stereo_personas: bool,
) -> pd.DataFrame:
    subset = persona_summary[_is_stereo_series(persona_summary["is_stereo"]) == stereo_personas].copy()
    if human_persona is not None:
        subset = subset.merge(
            human_persona[
                [
                    "persona_id",
                    "human_image_yes_rate",
                    "human_description_yes_rate",
                    "human_image_majority_agreement",
                    "human_description_majority_agreement",
                ]
            ],
            on="persona_id",
            how="left",
        )

    for _, icc_row in icc_summary[icc_summary["source"] == "LLM"].iterrows():
        subset[f"llm_icc2_{icc_row['modality']}"] = icc_row["icc2_single"]

    if not stereo_personas:
        subset = subset.rename(
            columns={
                "image_llm_consensus": "image_false_flag_llm",
                "description_llm_consensus": "description_false_flag_llm",
                "human_image_yes_rate": "human_image_false_flag_rate",
                "human_description_yes_rate": "human_description_false_flag_rate",
            }
        )

    return subset


def build_relatable_usefulness_icc_summary(
    llm_likert: pd.DataFrame,
    human_long: pd.DataFrame | None,
) -> pd.DataFrame:
    """Relatable and usefulness ICC separately for a_s_* and a_us_* groups."""
    rows: list[dict[str, Any]] = []

    for stereo_personas, group_name in [(True, "a_s_*"), (False, "a_us_*")]:
        llm_group = llm_likert[_is_stereo_series(llm_likert["is_stereo"]) == stereo_personas]

        rel_icc = compute_icc(llm_group.dropna(subset=["relatable_mean"]), "relatable_mean")
        use_icc = compute_icc(llm_group.dropna(subset=["usefulness_mean"]), "usefulness_mean")
        for construct, icc, extra_note in [
            (
                "relatable",
                rel_icc,
                f"Inter-LLM ICC on {group_name}. LLM relatable ↔ human _rel.",
            ),
            (
                "usefulness",
                use_icc,
                f"Inter-LLM ICC on {group_name}. LLM understand_group ↔ human _use (same survey item).",
            ),
        ]:
            row = icc_metric_row(
                "relatable_usefulness",
                construct,
                "persona",
                "LLM",
                icc=icc,
                notes=extra_note,
            )
            row["persona_group"] = group_name
            rows.append(row)

        if human_long is not None:
            condition = "stereo" if stereo_personas else "non_stereo"
            human_group = human_long[human_long["condition"] == condition].copy()
            human_group["relatable_numeric"] = human_group["relatable_answer"].map(likert_to_numeric)
            human_group["usefulness_numeric"] = human_group["usefulness_answer"].map(likert_to_numeric)
            for construct, col, human_note in [
                ("relatable", "relatable_numeric", "human _rel"),
                ("usefulness", "usefulness_numeric", "human _use (helped me understand)"),
            ]:
                row = icc_metric_row(
                    "relatable_usefulness",
                    construct,
                    "persona",
                    "Human",
                    agreement=compute_human_likert_agreement(human_group, col),
                    notes=f"Mean within-persona likert agreement on {group_name}; {human_note}.",
                )
                row["persona_group"] = group_name
                rows.append(row)

    return pd.DataFrame(rows)


def build_relatable_usefulness_persona_table(
    llm_likert: pd.DataFrame,
    human_persona: pd.DataFrame | None,
    all_runs: pd.DataFrame,
) -> pd.DataFrame:
    """One row per persona with LLM and human relatable/usefulness means."""
    meta = all_runs.drop_duplicates("persona_id").set_index("persona_id")
    rows: list[dict[str, Any]] = []

    for persona_id in sorted(all_runs["persona_id"].unique()):
        m = meta.loc[persona_id]
        lg = llm_likert[llm_likert["persona_id"] == persona_id]
        rel = lg["relatable_mean"].dropna()
        use = lg["usefulness_mean"].dropna()

        row: dict[str, Any] = {
            "persona_id": persona_id,
            "name": m["name"],
            "is_stereo": m["is_stereo"],
            "persona_group": "a_s_*" if _is_stereo_series(pd.Series([m["is_stereo"]])).iloc[0] else "a_us_*",
            "llm_relatable_mean": round(float(rel.mean()), 4) if len(rel) else None,
            "llm_usefulness_mean": round(float(use.mean()), 4) if len(use) else None,
            "llm_relatable_inter_model_sd": round(float(rel.std()), 4) if len(rel) > 1 else None,
            "llm_usefulness_inter_model_sd": round(float(use.std()), 4) if len(use) > 1 else None,
        }

        if human_persona is not None:
            hp = human_persona[human_persona["persona_id"] == persona_id]
            if not hp.empty:
                row["human_relatable_mean"] = hp.iloc[0].get("human_relatable_mean")
                row["human_usefulness_mean"] = hp.iloc[0].get("human_usefulness_mean")

        rows.append(row)

    return pd.DataFrame(rows)


def compute_human_likert_agreement(long_df: pd.DataFrame, numeric_col: str) -> dict[str, float | None]:
    """Mean within-persona agreement for likert scores (1 - std/2 on 1–5 scale)."""
    agreements: list[float] = []
    for _, group in long_df.groupby("persona_id"):
        values = group[numeric_col].dropna()
        if len(values) < 2:
            continue
        agreements.append(max(0.0, 1.0 - float(values.std()) / 2.0))

    if not agreements:
        return {"mean_within_persona_agreement": None}

    return {"mean_within_persona_agreement": round(float(sum(agreements) / len(agreements)), 4)}


def icc_metric_row(
    task: str,
    construct: str,
    modality: str,
    source: str,
    icc: dict[str, float | None] | None = None,
    agreement: dict[str, float | None] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "construct": construct,
        "modality": modality,
        "source": source,
        "icc2_single": icc.get("icc2_single") if icc else None,
        "icc2_average": icc.get("icc2_average") if icc else None,
        "mean_agreement": agreement["mean_majority_agreement"]
        if agreement and "mean_majority_agreement" in agreement
        else (agreement or {}).get("mean_within_persona_agreement"),
        "notes": notes or (icc.get("note", "") if icc else ""),
    }
    return row


def build_detection_icc_summary(
    model_summary: pd.DataFrame,
    human_long: pd.DataFrame | None,
) -> pd.DataFrame:
    return build_stereotype_flag_icc_summary(
        model_summary, human_long, stereo_personas=True
    )


def build_perception_icc_summary(
    model_summary: pd.DataFrame,
    human_long: pd.DataFrame | None,
) -> pd.DataFrame:
    return build_stereotype_flag_icc_summary(
        model_summary, human_long, stereo_personas=False
    )


def build_persona_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for persona_id, group in model_summary.groupby("persona_id", sort=True):
        meta = group.iloc[0]
        image = consistency_stats(group["image_mode_answer"])
        desc = consistency_stats(group["description_mode_answer"])
        gt = _is_stereo_series(pd.Series([meta["is_stereo"]])).iloc[0]
        image_mode = image["mode_answer"]
        desc_mode = desc["mode_answer"]

        rows.append(
            {
                "persona_id": persona_id,
                "name": meta["name"],
                "age": meta["age"],
                "gender": meta["gender"],
                "workforce": meta["workforce"],
                "is_stereo": meta["is_stereo"],
                "ground_truth_stereotypical": gt,
                "image_llm_consensus": image_mode,
                "image_llm_yes_count": image["yes_count"],
                "image_llm_no_count": image["no_count"],
                "image_inter_llm_agreement": image["consistency"],
                "image_llm_unanimous": image["unanimous"],
                "image_matches_ground_truth": (image_mode == "Yes") == gt if image_mode else None,
                "description_llm_consensus": desc_mode,
                "description_llm_yes_count": desc["yes_count"],
                "description_llm_no_count": desc["no_count"],
                "description_inter_llm_agreement": desc["consistency"],
                "description_llm_unanimous": desc["unanimous"],
                "description_matches_ground_truth": (desc_mode == "Yes") == gt if desc_mode else None,
            }
        )

    return pd.DataFrame(rows)


def build_stereo_comparison(model_summary: pd.DataFrame) -> pd.DataFrame:
    merged = model_summary.copy()
    merged["is_stereo_bool"] = merged["is_stereo"].astype(str).str.upper() == "TRUE"
    merged["image_binary"] = merged["image_mode_answer"].map(yes_no_to_binary)
    merged["description_binary"] = merged["description_mode_answer"].map(yes_no_to_binary)
    merged["image_matches_gt"] = (merged["image_mode_answer"] == "Yes") == merged["is_stereo_bool"]
    merged["description_matches_gt"] = (merged["description_mode_answer"] == "Yes") == merged["is_stereo_bool"]

    group_rows: list[dict[str, Any]] = []
    for label, is_stereo in [("Stereotypical (TRUE)", True), ("Non-stereotypical (FALSE)", False)]:
        subset = merged[merged["is_stereo_bool"] == is_stereo]
        group_rows.append(
            {
                "source": "llm",
                "comparison_level": "group",
                "group": label,
                "model_key": "ALL",
                "n_personas": subset["persona_id"].nunique(),
                "n_observations": len(subset),
                "image_yes_rate": round(subset["image_binary"].mean(), 4),
                "description_yes_rate": round(subset["description_binary"].mean(), 4),
                "mean_image_internal_consistency": round(subset["image_internal_consistency"].mean(), 4),
                "mean_description_internal_consistency": round(
                    subset["description_internal_consistency"].mean(), 4
                ),
                "image_ground_truth_agreement_rate": round(subset["image_matches_gt"].mean(), 4),
                "description_ground_truth_agreement_rate": round(
                    subset["description_matches_gt"].mean(), 4
                ),
            }
        )

    model_rows: list[dict[str, Any]] = []
    for model_key, model_subset in merged.groupby("model_key", sort=True):
        for label, is_stereo in [("Stereotypical (TRUE)", True), ("Non-stereotypical (FALSE)", False)]:
            subset = model_subset[model_subset["is_stereo_bool"] == is_stereo]
            if subset.empty:
                continue
            model_rows.append(
                {
                    "source": "llm",
                    "comparison_level": "model_x_group",
                    "group": label,
                    "model_key": model_key,
                    "n_personas": subset["persona_id"].nunique(),
                    "n_observations": len(subset),
                    "image_yes_rate": round(subset["image_binary"].mean(), 4),
                    "description_yes_rate": round(subset["description_binary"].mean(), 4),
                    "mean_image_internal_consistency": round(
                        subset["image_internal_consistency"].mean(), 4
                    ),
                    "mean_description_internal_consistency": round(
                        subset["description_internal_consistency"].mean(), 4
                    ),
                    "image_ground_truth_agreement_rate": round(subset["image_matches_gt"].mean(), 4),
                    "description_ground_truth_agreement_rate": round(
                        subset["description_matches_gt"].mean(), 4
                    ),
                }
            )

    return pd.DataFrame(group_rows + model_rows)


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float | None:
    """Cohen's kappa for two binary codings (0/1)."""
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
    if len(clean) < 2:
        return None, None
    r, p = pearsonr(clean["x"], clean["y"])
    return round(float(r), 4), round(float(p), 4)


def build_human_llm_stereotype_agreement(
    persona_summary: pd.DataFrame,
    human_persona: pd.DataFrame | None,
    *,
    stereo_personas: bool,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Human vs LLM agreement on stereotype Yes/No flags.

    detection: a_s_* (did both detect stereotype when present?)
    perception: a_us_* (did both falsely flag when absent?)
    """
    if human_persona is None:
        return pd.DataFrame(), pd.DataFrame()

    merged = persona_summary.merge(
        human_persona,
        on="persona_id",
        how="inner",
        suffixes=("", "_human"),
    )
    stereo_col = "is_stereo" if "is_stereo" in merged.columns else "ground_truth_stereotypical"
    merged = merged[_is_stereo_series(merged[stereo_col]) == stereo_personas].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["llm_image_yes_rate"] = merged["image_llm_yes_count"] / 5
    merged["llm_description_yes_rate"] = merged["description_llm_yes_count"] / 5
    merged["image_consensus_match"] = merged["image_llm_consensus"] == merged["human_image_consensus"]
    merged["description_consensus_match"] = (
        merged["description_llm_consensus"] == merged["human_description_consensus"]
    )
    merged["image_yes_rate_delta"] = merged["llm_image_yes_rate"] - merged["human_image_yes_rate"]
    merged["description_yes_rate_delta"] = (
        merged["llm_description_yes_rate"] - merged["human_description_yes_rate"]
    )

    detail = merged[
        [
            "persona_id",
            "name",
            "image_llm_consensus",
            "human_image_consensus",
            "image_consensus_match",
            "llm_image_yes_rate",
            "human_image_yes_rate",
            "image_yes_rate_delta",
            "description_llm_consensus",
            "human_description_consensus",
            "description_consensus_match",
            "llm_description_yes_rate",
            "human_description_yes_rate",
            "description_yes_rate_delta",
        ]
    ].copy()
    detail.insert(0, "task", task)

    summary_rows: list[dict[str, Any]] = []
    for modality, llm_cons, hum_cons, llm_rate, hum_rate, match_col in [
        (
            "image",
            "image_llm_consensus",
            "human_image_consensus",
            "llm_image_yes_rate",
            "human_image_yes_rate",
            "image_consensus_match",
        ),
        (
            "description",
            "description_llm_consensus",
            "human_description_consensus",
            "llm_description_yes_rate",
            "human_description_yes_rate",
            "description_consensus_match",
        ),
    ]:
        llm_bin = (merged[llm_cons] == "Yes").astype(int)
        hum_bin = (merged[hum_cons] == "Yes").astype(int)
        r, p = _rate_correlation(merged[llm_rate], merged[hum_rate])
        summary_rows.append(
            {
                "task": task,
                "modality": modality,
                "n_personas": len(merged),
                "consensus_agreement_rate": round(float(merged[match_col].mean()), 4),
                "cohens_kappa": cohens_kappa(llm_bin, hum_bin),
                "yes_rate_pearson_r": r,
                "yes_rate_pearson_p": p,
                "mean_abs_yes_rate_delta": round(
                    float(merged[f"{modality}_yes_rate_delta"].abs().mean()), 4
                ),
                "mean_yes_rate_delta_llm_minus_human": round(
                    float(merged[f"{modality}_yes_rate_delta"].mean()), 4
                ),
            }
        )

    return detail, pd.DataFrame(summary_rows)


def build_human_llm_likert_agreement(rel_use_persona: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Human vs LLM agreement on relatable and usefulness (understand) likert means."""
    rows: list[dict[str, Any]] = []

    for group in ["a_s_*", "a_us_*"]:
        sub = rel_use_persona[rel_use_persona["persona_group"] == group]
        for construct, llm_col, hum_col in [
            ("relatable", "llm_relatable_mean", "human_relatable_mean"),
            ("usefulness", "llm_usefulness_mean", "human_usefulness_mean"),
        ]:
            clean = sub[[llm_col, hum_col, "persona_id"]].dropna()
            r, p = _rate_correlation(clean[llm_col], clean[hum_col]) if len(clean) >= 2 else (None, None)
            delta = clean[llm_col] - clean[hum_col] if len(clean) else pd.Series(dtype=float)
            rows.append(
                {
                    "persona_group": group,
                    "construct": construct,
                    "n_personas": len(clean),
                    "pearson_r": r,
                    "pearson_p": p,
                    "mean_abs_delta": round(float(delta.abs().mean()), 4) if len(delta) else None,
                    "mean_delta_llm_minus_human": round(float(delta.mean()), 4) if len(delta) else None,
                    "llm_column": llm_col,
                    "human_column": hum_col + (" (_use)" if construct == "usefulness" else " (_rel)"),
                }
            )

    summary = pd.DataFrame(rows)

    detail_rows: list[dict[str, Any]] = []
    for _, row in rel_use_persona.iterrows():
        for construct, llm_col, hum_col in [
            ("relatable", "llm_relatable_mean", "human_relatable_mean"),
            ("usefulness", "llm_usefulness_mean", "human_usefulness_mean"),
        ]:
            lv, hv = row.get(llm_col), row.get(hum_col)
            detail_rows.append(
                {
                    "persona_id": row["persona_id"],
                    "name": row["name"],
                    "persona_group": row["persona_group"],
                    "construct": construct,
                    "llm_mean": lv,
                    "human_mean": hv,
                    "delta_llm_minus_human": round(float(lv - hv), 4)
                    if pd.notna(lv) and pd.notna(hv)
                    else None,
                }
            )

    return pd.DataFrame(detail_rows), summary


def write_sheet_with_blocks(
    writer: pd.ExcelWriter,
    sheet_name: str,
    blocks: list[tuple[str, pd.DataFrame]],
) -> None:
    """Write labelled blocks to one sheet with blank rows between."""
    row = 0
    for label, df in blocks:
        if df.empty:
            continue
        header = pd.DataFrame([{"section": label}])
        header.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=row)
        row += 1
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
        row += len(df) + 2


def build_human_llm_comparison(
    persona_summary: pd.DataFrame,
    human_persona_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = persona_summary.merge(human_persona_summary, on="persona_id", how="left", suffixes=("", "_human"))

    merged["llm_human_image_match"] = merged["image_llm_consensus"] == merged["human_image_consensus"]
    merged["llm_human_description_match"] = (
        merged["description_llm_consensus"] == merged["human_description_consensus"]
    )
    merged["human_image_matches_ground_truth"] = (
        (merged["human_image_consensus"] == "Yes") == merged["ground_truth_stereotypical"]
    )
    merged["human_description_matches_ground_truth"] = (
        (merged["human_description_consensus"] == "Yes") == merged["ground_truth_stereotypical"]
    )
    merged["image_yes_rate_delta_llm_minus_human"] = (
        merged["image_llm_yes_count"] / 5 - merged["human_image_yes_rate"]
    )
    merged["description_yes_rate_delta_llm_minus_human"] = (
        merged["description_llm_yes_count"] / 5 - merged["human_description_yes_rate"]
    )

    summary = pd.DataFrame(
        [
            {
                "persona_id": "_SUMMARY_",
                "name": "Mean agreement rates",
                "llm_human_image_match": round(merged["llm_human_image_match"].mean(), 4),
                "llm_human_description_match": round(merged["llm_human_description_match"].mean(), 4),
                "human_image_matches_ground_truth": round(
                    merged["human_image_matches_ground_truth"].mean(), 4
                ),
                "human_description_matches_ground_truth": round(
                    merged["human_description_matches_ground_truth"].mean(), 4
                ),
                "image_matches_ground_truth": round(merged["image_matches_ground_truth"].mean(), 4),
                "description_matches_ground_truth": round(
                    merged["description_matches_ground_truth"].mean(), 4
                ),
            }
        ]
    )
    return pd.concat([merged, summary], ignore_index=True)


def write_sheet4(
    writer: pd.ExcelWriter,
    llm_stereo: pd.DataFrame,
    human_stereo: pd.DataFrame | None,
    human_llm: pd.DataFrame | None,
) -> None:
    sheet_name = "Stereo vs Human"
    row = 0
    llm_stereo.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    row += len(llm_stereo) + 2

    if human_stereo is not None:
        human_stereo.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
        row += len(human_stereo) + 2

    if human_llm is not None:
        human_llm.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)


def export(
    results_dir: Path,
    personas_path: Path,
    config_path: Path,
    output_csv: Path,
    output_xlsx: Path,
    human_data_path: Path | None,
    human_url: str = DEFAULT_HUMAN_CSV_URL,
    export_human_long: Path | None = Path("results/human_responses_long.csv"),
) -> None:
    personas = load_personas(personas_path)
    model_keys = load_model_keys(config_path)
    all_runs = load_all_runs(results_dir, personas, model_keys=model_keys)

    export_cols = [c for c in RUN_COLS if c in all_runs.columns]
    all_runs[export_cols].to_csv(output_csv, index=False, encoding="utf-8")

    model_summary = build_model_summary(all_runs)
    persona_summary = build_persona_summary(model_summary)
    llm_likert = build_llm_likert_scores(all_runs)
    llm_stereo = build_stereo_comparison(model_summary)

    human_long: pd.DataFrame | None = None
    human_persona: pd.DataFrame | None = None
    human_stereo: pd.DataFrame | None = None
    human_llm: pd.DataFrame | None = None

    try:
        human_long, human_persona = load_human_study(path=human_data_path, url=human_url)
        human_stereo = build_human_stereo_comparison(human_long)
        human_llm = build_human_llm_comparison(persona_summary, human_persona)
        if export_human_long is not None:
            export_human_long.parent.mkdir(parents=True, exist_ok=True)
            human_long.to_csv(export_human_long, index=False, encoding="utf-8")
    except FileNotFoundError:
        pass

    detection_icc_summary = build_detection_icc_summary(model_summary, human_long)
    perception_icc_summary = build_perception_icc_summary(model_summary, human_long)
    rel_use_icc_summary = build_relatable_usefulness_icc_summary(llm_likert, human_long)

    detection_persona = build_stereotype_flag_persona_table(
        persona_summary, human_persona, detection_icc_summary, stereo_personas=True
    )
    perception_persona = build_stereotype_flag_persona_table(
        persona_summary, human_persona, perception_icc_summary, stereo_personas=False
    )
    rel_use_persona = build_relatable_usefulness_persona_table(llm_likert, human_persona, all_runs)

    detection_hl_detail, detection_hl_summary = build_human_llm_stereotype_agreement(
        persona_summary, human_persona, stereo_personas=True, task="detection"
    )
    perception_hl_detail, perception_hl_summary = build_human_llm_stereotype_agreement(
        persona_summary, human_persona, stereo_personas=False, task="perception"
    )
    rel_use_hl_detail, rel_use_hl_summary = build_human_llm_likert_agreement(rel_use_persona)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        all_runs[export_cols].to_excel(writer, sheet_name="All Runs", index=False)
        model_summary.to_excel(writer, sheet_name="Per Persona x LLM", index=False)
        write_sheet_with_blocks(
            writer,
            "Detection ICC",
            [
                ("Within-rater ICC / agreement (LLM & Human)", detection_icc_summary),
                ("Per persona — detection (a_s_*)", detection_persona),
                ("Human vs LLM agreement — summary", detection_hl_summary),
                ("Human vs LLM agreement — per persona", detection_hl_detail),
            ],
        )
        write_sheet_with_blocks(
            writer,
            "Perception ICC",
            [
                ("Within-rater ICC / agreement (LLM & Human)", perception_icc_summary),
                ("Per persona — perception / false flags (a_us_*)", perception_persona),
                ("Human vs LLM agreement — summary", perception_hl_summary),
                ("Human vs LLM agreement — per persona", perception_hl_detail),
            ],
        )
        write_sheet_with_blocks(
            writer,
            "Relatable Usefulness ICC",
            [
                ("Within-rater ICC / agreement (LLM & Human)", rel_use_icc_summary),
                ("Per persona — likert means", rel_use_persona),
                ("Human vs LLM agreement — summary", rel_use_hl_summary),
                ("Human vs LLM agreement — per persona", rel_use_hl_detail),
            ],
        )
        write_sheet4(writer, llm_stereo, human_stereo, human_llm)

    print(f"Wrote combined CSV: {output_csv} ({len(all_runs)} rows)")
    print(f"Wrote Excel workbook: {output_xlsx}")
    print(f"  Sheet 1 — All Runs: {len(all_runs)} rows")
    print(f"  Sheet 2 — Per Persona x LLM: {len(model_summary)} rows")
    print(f"  Sheet 3 — Detection ICC: internal ICC + human↔LLM agreement")
    print(f"  Sheet 4 — Perception ICC: internal ICC + human↔LLM agreement")
    print(f"  Sheet 5 — Relatable Usefulness ICC: internal ICC + human↔LLM agreement")
    print(f"  Sheet 6 — Stereo vs Human")
    if human_long is not None:
        print(f"  Human data: {human_long['pid'].nunique()} raters, {len(human_long)} rater×persona rows")
        if export_human_long is not None:
            print(f"  Human long CSV: {export_human_long}")
    else:
        print("  Human data: not loaded (provide --human-data or allow download from GitHub)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export combined CSV and analysis Excel workbook.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--personas-csv", type=Path, default=Path("data/personas.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/all_evaluations.csv"))
    parser.add_argument("--output-xlsx", type=Path, default=Path("results/analysis.xlsx"))
    parser.add_argument(
        "--human-data",
        type=Path,
        default=Path("data/human_study_cleaned.csv"),
        help="Prolific/Qualtrics cleaned CSV (wide format). Downloaded from GitHub if missing.",
    )
    parser.add_argument(
        "--no-download-human",
        action="store_true",
        help="Do not download human CSV from GitHub when local file is missing.",
    )
    args = parser.parse_args()

    human_path = args.human_data if args.human_data.exists() else args.human_data
    if args.no_download_human and not args.human_data.exists():
        human_path = None

    export(
        results_dir=args.results_dir,
        personas_path=args.personas_csv,
        config_path=args.config,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
        human_data_path=human_path,
        human_url="" if args.no_download_human else DEFAULT_HUMAN_CSV_URL,
    )


if __name__ == "__main__":
    main()
