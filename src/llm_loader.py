"""Load and validate the corrected five-LLM evaluation dataset."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from export_analysis import RUN_COLS, load_personas, yes_no_to_binary
from src.persona_labels import MIRROR_EXCLUDED_DIRS

PERSONA_META_COLS = ["name", "age", "gender", "workforce", "is_stereo"]


def load_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "runs_per_model": int(config["runs_per_model"]),
        "model_keys": set(config.get("models", {}).keys()),
        "config": config,
    }


def _is_standard_persona_dir(dirname: str) -> bool:
    return dirname not in MIRROR_EXCLUDED_DIRS and not dirname.startswith("human_mirror")


def _invalid_yes_no(value: Any) -> bool:
    if pd.isna(value) or str(value).strip() == "":
        return True
    return str(value).strip().lower() not in {"yes", "no"}


def load_clean_llm_runs(
    results_dir: Path,
    personas: pd.DataFrame,
    *,
    runs_per_model: int,
    model_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load standard LLM evaluations with filtering audit.

    Returns (clean_df, audit_df).
    """
    meta = personas.set_index("persona_id")[PERSONA_META_COLS]
    kept_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for csv_path in sorted(glob.glob(str(results_dir / "*/evaluations.csv"))):
        persona_id = Path(csv_path).parent.name
        if not _is_standard_persona_dir(persona_id):
            continue

        raw = pd.read_csv(csv_path)
        raw = raw.join(meta, on="persona_id", how="left")
        raw["run_index"] = pd.to_numeric(raw["run_index"], errors="coerce")

        for idx, row in raw.iterrows():
            reasons: list[str] = []
            if row["model_key"] not in model_keys:
                reasons.append("model_not_configured")
            if pd.isna(row["run_index"]) or int(row["run_index"]) < 1 or int(row["run_index"]) > runs_per_model:
                reasons.append("run_index_out_of_range")
            if pd.notna(row.get("error")) and str(row.get("error")).strip():
                reasons.append("evaluation_error")
            if _invalid_yes_no(row.get("description_stereotype_answer")) or _invalid_yes_no(
                row.get("image_stereotype_answer")
            ):
                reasons.append("invalid_yes_no")

            if reasons:
                audit_rows.append(
                    {
                        "persona_id": persona_id,
                        "model_key": row.get("model_key"),
                        "run_index": row.get("run_index"),
                        "exclusion_reason": ";".join(reasons),
                        "source_file": csv_path,
                    }
                )

        candidate = raw[
            raw["model_key"].isin(model_keys)
            & raw["run_index"].between(1, runs_per_model)
            & (raw["error"].isna() | (raw["error"].astype(str).str.strip() == ""))
        ].copy()

        candidate = candidate[
            ~candidate["description_stereotype_answer"].apply(_invalid_yes_no)
            & ~candidate["image_stereotype_answer"].apply(_invalid_yes_no)
        ]

        dedupe_cols = [c for c in candidate.columns if c not in {"timestamp", "latency_ms", "response_raw"}]
        duplicated_mask = candidate.duplicated(subset=dedupe_cols, keep="first")
        for _, row in candidate[duplicated_mask].iterrows():
            audit_rows.append(
                {
                    "persona_id": row["persona_id"],
                    "model_key": row["model_key"],
                    "run_index": row["run_index"],
                    "exclusion_reason": "exact_duplicate",
                    "source_file": csv_path,
                }
            )
        deduped = candidate.drop_duplicates(subset=dedupe_cols, keep="first")
        kept_frames.append(deduped)

    if not kept_frames:
        raise FileNotFoundError(f"No standard evaluations.csv files found under {results_dir}")

    clean = pd.concat(kept_frames, ignore_index=True)
    clean["run_index"] = clean["run_index"].astype(int)
    clean = clean.sort_values(["persona_id", "model_key", "run_index"]).reset_index(drop=True)
    clean["is_stereo"] = clean["is_stereo"].map({True: "TRUE", False: "FALSE"})

    validate_persona_model_counts(clean, runs_per_model=runs_per_model)
    return clean, pd.DataFrame(audit_rows)


def validate_persona_model_counts(df: pd.DataFrame, *, runs_per_model: int) -> None:
    counts = df.groupby(["persona_id", "model_key"]).agg(
        n=("run_index", "size"),
        run_indices=("run_index", lambda s: sorted(s.astype(int).tolist())),
    )
    bad = counts[counts["n"] != runs_per_model]
    if bad.empty:
        return
    lines = ["Invalid persona-model evaluation counts:"]
    for (persona_id, model_key), row in bad.iterrows():
        lines.append(
            f"  persona_id={persona_id}, model_key={model_key}, "
            f"observed={int(row['n'])}, run_indices={row['run_indices']}"
        )
    raise ValueError("\n".join(lines))


def prepare_llm_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    from src.human_data import likert_to_numeric

    out = df.copy()
    out["image_yes"] = out["image_stereotype_answer"].map(yes_no_to_binary)
    out["description_yes"] = out["description_stereotype_answer"].map(yes_no_to_binary)
    out["is_stereo_bool"] = out["is_stereo"].astype(str).str.upper().eq("TRUE")
    out["condition"] = out["is_stereo_bool"].map({True: "stereo", False: "non_stereo"})
    out["usefulness_num"] = out["understand_group_answer"].map(likert_to_numeric)
    out["relatable_num"] = out["relatable_answer"].map(likert_to_numeric)
    return out


def build_validation_summary(df: pd.DataFrame, *, runs_per_model: int, model_keys: set[str]) -> dict[str, Any]:
    prep = prepare_llm_analysis_frame(df)
    by_condition = prep.groupby("condition").size().to_dict()
    by_model = prep.groupby("model_key").size().to_dict()
    by_persona = prep.groupby("persona_id").size().to_dict()
    by_persona_model = {
        f"{persona}|{model}": int(n)
        for (persona, model), n in prep.groupby(["persona_id", "model_key"]).size().items()
    }

    return {
        "total_evaluations": len(prep),
        "sp_evaluations": int(by_condition.get("stereo", 0)),
        "nsp_evaluations": int(by_condition.get("non_stereo", 0)),
        "evaluations_by_model": {k: int(v) for k, v in by_model.items()},
        "evaluations_by_persona": {k: int(v) for k, v in by_persona.items()},
        "evaluations_by_persona_model": by_persona_model,
        "min_run_index": int(prep["run_index"].min()),
        "max_run_index": int(prep["run_index"].max()),
        "n_configured_models": len(model_keys),
        "n_personas": prep["persona_id"].nunique(),
        "runs_per_model": runs_per_model,
    }


def write_validation_summary(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "LLM CLEAN DATA VALIDATION",
        "=" * 60,
        f"Total evaluations: {summary['total_evaluations']}",
        f"SP evaluations: {summary['sp_evaluations']}",
        f"NSP evaluations: {summary['nsp_evaluations']}",
        f"Configured models: {summary['n_configured_models']}",
        f"Personas: {summary['n_personas']}",
        f"Run indices: {summary['min_run_index']}--{summary['max_run_index']}",
        "",
        "Evaluations by model:",
    ]
    for model, n in sorted(summary["evaluations_by_model"].items()):
        lines.append(f"  {model}: {n}")
    lines.extend(["", "Evaluations by persona:"])
    for persona, n in sorted(summary["evaluations_by_persona"].items()):
        lines.append(f"  {persona}: {n}")
    lines.extend(["", "Evaluations by persona x model (expect 5 each):"])
    for key, n in sorted(summary["evaluations_by_persona_model"].items()):
        persona, model = key.split("|", 1)
        lines.append(f"  {persona} x {model}: {n}")

    runs_per_model = summary["runs_per_model"]
    checks = [
        summary["total_evaluations"] == 300,
        summary["sp_evaluations"] == 150,
        summary["nsp_evaluations"] == 150,
        summary["min_run_index"] == 1,
        summary["max_run_index"] == runs_per_model,
        all(n == 60 for n in summary["evaluations_by_model"].values()),
        all(n == 25 for n in summary["evaluations_by_persona"].values()),
        all(n == 5 for n in summary["evaluations_by_persona_model"].values()),
    ]
    lines.extend(["", "Validation checks:"])
    labels = [
        "Total evaluations == 300",
        "SP evaluations == 150",
        "NSP evaluations == 150",
        "Min run_index == 1",
        f"Max run_index == {summary['runs_per_model']}",
        "Each model == 60",
        "Each persona == 25",
        "Each persona-model == 5",
    ]
    for label, ok in zip(labels, checks):
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_aggregates(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    export_cols = [c for c in RUN_COLS if c in df.columns]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df[export_cols].to_csv(csv_path, index=False, encoding="utf-8")

    records = df[export_cols].to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
