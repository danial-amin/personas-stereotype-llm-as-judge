"""Load and aggregate Prolific human stereotype ratings (Qualtrics export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.csv_utils import detect_csv_encoding

# Column suffixes in the cleaned Qualtrics CSV (see persona_stereotypes_analysis_v4_DA.ipynb)
US_PERSONAS = ["marcus", "eleanor", "mei", "marlinq", "richard", "margart"]
S_PERSONAS = ["bernie", "sofia", "caroline", "james", "ray", "aisha"]

PERSONA_SUFFIX_TO_ID: dict[str, str] = {
    "marcus": "a_us_marcus",
    "eleanor": "a_us_eleanor",
    "mei": "a_us_mei",
    "marlinq": "a_us_marcus_linq",
    "richard": "a_us_richard",
    "margart": "a_us_margaret",
    "bernie": "a_s_bernie",
    "sofia": "a_s_sofia",
    "caroline": "a_s_caroline",
    "james": "a_s_james",
    "ray": "a_s_ray",
    "aisha": "a_s_aisha",
}

DEFAULT_HUMAN_CSV_URL = (
    "https://raw.githubusercontent.com/danial-amin/persona-papers/refs/heads/main/"
    "Persona%20Stereotypes%20Evaluation-2026%20-%20Cleaned-v1.csv"
)


def _yn_to_int(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "yes":
        return 1.0
    if text == "no":
        return 0.0
    return None


def load_human_raw(path: Path) -> pd.DataFrame:
    """Load the wide Qualtrics CSV, dropping the question-text row."""
    encoding = detect_csv_encoding(path)
    raw = pd.read_csv(path, encoding=encoding)
    return raw.iloc[1:].reset_index(drop=True).copy()


def build_human_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (Prolific participant × persona seen)."""
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        pid = row["PROLIFIC_PID"]
        for suffix in US_PERSONAS:
            desc = row.get(f"a_us_{suffix}_str")
            if pd.notna(desc):
                records.append(
                    {
                        "pid": pid,
                        "persona_suffix": suffix,
                        "persona_id": PERSONA_SUFFIX_TO_ID[suffix],
                        "condition": "non_stereo",
                        "description_answer": str(desc).strip(),
                        "description_yes": _yn_to_int(desc),
                        "image_answer": _img_val(row.get(f"a_us_{suffix}_img")),
                        "image_yes": _yn_to_int(row.get(f"a_us_{suffix}_img")),
                        "relatable_answer": _likert_val(row.get(f"a_us_{suffix}_rel")),
                        "usefulness_answer": _likert_val(row.get(f"a_us_{suffix}_use")),
                    }
                )
        for suffix in S_PERSONAS:
            desc = row.get(f"a_s_{suffix}_str")
            if pd.notna(desc):
                records.append(
                    {
                        "pid": pid,
                        "persona_suffix": suffix,
                        "persona_id": PERSONA_SUFFIX_TO_ID[suffix],
                        "condition": "stereo",
                        "description_answer": str(desc).strip(),
                        "description_yes": _yn_to_int(desc),
                        "image_answer": _img_val(row.get(f"a_s_{suffix}_img")),
                        "image_yes": _yn_to_int(row.get(f"a_s_{suffix}_img")),
                        "relatable_answer": _likert_val(row.get(f"a_s_{suffix}_rel")),
                        "usefulness_answer": _likert_val(row.get(f"a_s_{suffix}_use")),
                    }
                )

    long_df = pd.DataFrame(records)
    long_df["is_stereo"] = long_df["condition"].eq("stereo")
    return long_df


def _img_val(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"yes", "no"}:
        return text.capitalize()
    return text


LIKERT_ORDER = [
    "Strongly disagree",
    "Somewhat disagree",
    "Neither agree nor disagree",
    "Somewhat agree",
    "Strongly agree",
]


def _likert_val(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text in LIKERT_ORDER else None


def likert_to_numeric(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return float(LIKERT_ORDER.index(text) + 1)
    except ValueError:
        return None


def build_human_persona_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate human ratings to one row per persona."""
    rows: list[dict[str, Any]] = []

    for persona_id, group in long_df.groupby("persona_id", sort=True):
        desc = group["description_yes"].dropna()
        img = group["image_yes"].dropna()
        desc_rate = float(desc.mean()) if len(desc) else None
        img_rate = float(img.mean()) if len(img) else None

        def _consensus(rate: float | None) -> str | None:
            if rate is None:
                return None
            return "Yes" if rate >= 0.5 else "No"

        rows.append(
            {
                "persona_id": persona_id,
                "human_n_raters": int(len(group)),
                "human_description_yes_count": int(desc.sum()) if len(desc) else 0,
                "human_description_no_count": int((1 - desc).sum()) if len(desc) else 0,
                "human_description_yes_rate": round(desc_rate, 4) if desc_rate is not None else None,
                "human_description_consensus": _consensus(desc_rate),
                "human_description_majority_agreement": round(max(desc_rate, 1 - desc_rate), 4)
                if desc_rate is not None
                else None,
                "human_image_yes_count": int(img.sum()) if len(img) else 0,
                "human_image_no_count": int((1 - img).sum()) if len(img) else 0,
                "human_image_yes_rate": round(img_rate, 4) if img_rate is not None else None,
                "human_image_consensus": _consensus(img_rate),
                "human_image_majority_agreement": round(max(img_rate, 1 - img_rate), 4)
                if img_rate is not None
                else None,
                "condition": group["condition"].iloc[0],
                "is_stereo": group["is_stereo"].iloc[0],
                "human_relatable_mean": _likert_mean(group["relatable_answer"]),
                "human_usefulness_mean": _likert_mean(group["usefulness_answer"]),
            }
        )

    return pd.DataFrame(rows)


def _likert_mean(series: pd.Series) -> float | None:
    values = series.map(likert_to_numeric).dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def build_human_stereo_comparison(long_df: pd.DataFrame) -> pd.DataFrame:
    """Human yes-rates by stereotype condition (mirrors LLM stereo comparison)."""
    rows: list[dict[str, Any]] = []
    for label, condition in [("Stereotypical (TRUE)", "stereo"), ("Non-stereotypical (FALSE)", "non_stereo")]:
        subset = long_df[long_df["condition"] == condition]
        gt = condition == "stereo"
        rows.append(
            {
                "source": "human",
                "comparison_level": "group",
                "group": label,
                "n_personas": subset["persona_id"].nunique(),
                "n_rater_persona_pairs": len(subset),
                "image_yes_rate": round(subset["image_yes"].mean(), 4),
                "description_yes_rate": round(subset["description_yes"].mean(), 4),
                "image_ground_truth_agreement_rate": round(
                    ((subset["image_yes"] == 1) == gt).mean(), 4
                ),
                "description_ground_truth_agreement_rate": round(
                    ((subset["description_yes"] == 1) == gt).mean(), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def load_human_study(path: Path | None = None, url: str = DEFAULT_HUMAN_CSV_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load human study data and return (long_df, persona_summary).

    If `path` is missing, downloads from the default GitHub URL into data/human_study_cleaned.csv.
    """
    if path is None:
        path = Path("data/human_study_cleaned.csv")

    if not path.exists() and url:
        path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(url, path)

    if not path.exists():
        raise FileNotFoundError(f"Human study CSV not found: {path}")

    raw = load_human_raw(path)
    long_df = build_human_long(raw)
    persona_summary = build_human_persona_summary(long_df)
    return long_df, persona_summary
