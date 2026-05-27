#!/usr/bin/env python3
"""Generate text summaries for standard LLM evaluations and the human-mirror study."""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pingouin as pg
import yaml
from scipy.stats import chi2_contingency, wilcoxon

from export_analysis import (
    consistency_stats,
    load_all_runs,
    load_personas,
    yes_no_to_binary,
)
from src.human_data import LIKERT_ORDER, likert_to_numeric
from src.mirror_storage import load_mirror_results_successful

DEFAULT_OUTPUT_DIR = Path("outputs")
from pick_mirror_model import DEFAULT_MODEL, mirror_output_dir

DEFAULT_MIRROR_DIR = mirror_output_dir(DEFAULT_MODEL)
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_CONFIG = Path("config.yaml")

PERSONA_SHORT: dict[str, str] = {
    "a_us_marcus": "marcus",
    "a_us_eleanor": "eleanor",
    "a_us_mei": "mei",
    "a_us_marcus_linq": "marlinq",
    "a_us_richard": "richard",
    "a_us_margaret": "margart",
    "a_s_bernie": "bernie",
    "a_s_sofia": "sofia",
    "a_s_caroline": "caroline",
    "a_s_james": "james",
    "a_s_ray": "ray",
    "a_s_aisha": "aisha",
}


def apa_p(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def two_proportion_z(count1: int, n1: int, count2: int, n2: int) -> tuple[float, float]:
    import numpy as np
    from statsmodels.stats.proportion import proportions_ztest

    z, p = proportions_ztest([count1, count2], [n1, n2])
    return float(z), float(p)


def majority_agreement(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    rate = float(values.mean())
    return max(rate, 1 - rate)


def icc21(data: pd.DataFrame, target_col: str, rater_col: str, rating_col: str) -> dict[str, Any]:
    clean = data.dropna(subset=[target_col, rater_col, rating_col]).copy()
    if clean[target_col].nunique() < 2 or clean[rater_col].nunique() < 2:
        return {"icc21": None, "icc2k": None, "note": "insufficient data"}

    per_target_var = clean.groupby(target_col)[rating_col].var(numeric_only=True)
    if (per_target_var.fillna(0) == 0).all():
        return {"icc21": None, "icc2k": None, "note": "undefined — zero variance across raters"}

    try:
        icc_table = pg.intraclass_corr(
            data=clean,
            targets=target_col,
            raters=rater_col,
            ratings=rating_col,
            nan_policy="raise",
        )
        row21 = icc_table[icc_table["Type"] == "ICC2"]
        row2k = icc_table[icc_table["Type"] == "ICC2k"]
        return {
            "icc21": round(float(row21["ICC"].iloc[0]), 3) if not row21.empty else None,
            "icc2k": round(float(row2k["ICC"].iloc[0]), 3) if not row2k.empty else None,
            "note": "",
            "n_targets": clean[target_col].nunique(),
            "n_raters": clean[rater_col].nunique(),
            "n_obs": len(clean),
        }
    except Exception as exc:
        return {"icc21": None, "icc2k": None, "note": str(exc)}


class SummaryWriter:
    def __init__(self) -> None:
        self.buf = io.StringIO()

    def w(self, *args: Any) -> None:
        print(*args, file=self.buf)

    def h(self, title: str, char: str = "=") -> None:
        self.w()
        self.w(char * 78)
        self.w(title)
        self.w(char * 78)

    def text(self) -> str:
        return self.buf.getvalue()


def _prepare_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["image_yes"] = out["image_stereotype_answer"].map(yes_no_to_binary)
    out["description_yes"] = out["description_stereotype_answer"].map(yes_no_to_binary)
    if "is_stereo_bool" not in out.columns:
        if "is_stereo" in out.columns:
            out["is_stereo_bool"] = out["is_stereo"].astype(str).str.upper().eq("TRUE")
        elif "condition" in out.columns:
            out["is_stereo_bool"] = out["condition"].eq("stereo")
        else:
            raise KeyError("Expected is_stereo, is_stereo_bool, or condition column")
    out["condition"] = out["is_stereo_bool"].map({True: "stereo", False: "non_stereo"})
    out["usefulness_num"] = out["understand_group_answer"].map(likert_to_numeric)
    out["relatable_num"] = out["relatable_answer"].map(likert_to_numeric)
    return out


def _mirror_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        parsed = record.get("parsed") or {}
        image = parsed.get("image_stereotype") or {}
        description = parsed.get("persona_description_stereotype") or {}
        understand = parsed.get("this_persona_helped_me_understand_this_group_of_people") or {}
        relatable = parsed.get("i_find_this_persona_relatable") or {}
        stereotype_type = parsed.get("stereotype_type") or {}

        rows.append(
            {
                "participant_id": record["participant_id"],
                "participant_index": record["participant_index"],
                "persona_id": record["persona_id"],
                "condition": record["condition"],
                "is_stereo_bool": record["condition"] == "stereo",
                "image_stereotype_answer": image.get("answer", ""),
                "description_stereotype_answer": description.get("answer", ""),
                "understand_group_answer": understand.get("answer", ""),
                "relatable_answer": relatable.get("answer", ""),
                "stereotype_age": stereotype_type.get("age", ""),
                "stereotype_gender": stereotype_type.get("gender", ""),
                "stereotype_occupation": stereotype_type.get("occupation", ""),
                "stereotype_other": stereotype_type.get("other", ""),
            }
        )
    df = pd.DataFrame(rows)
    return _prepare_runs(df)


def _write_manipulation_check(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("RQ1 — MANIPULATION CHECK")
    stereo = df[df["condition"] == "stereo"]
    non_stereo = df[df["condition"] == "non_stereo"]

    s_yes = int(stereo["description_yes"].sum())
    s_n = int(stereo["description_yes"].notna().sum())
    ns_yes = int(non_stereo["description_yes"].sum())
    ns_n = int(non_stereo["description_yes"].notna().sum())

    s_rate = s_yes / s_n if s_n else float("nan")
    ns_rate = ns_yes / ns_n if ns_n else float("nan")
    diff_pp = (s_rate - ns_rate) * 100

    sw.w(f"  Yes-rate stereo:     {s_rate:.1%} ({s_yes}/{s_n})")
    sw.w(f"  Yes-rate non-stereo: {ns_rate:.1%} ({ns_yes}/{ns_n})")
    sw.w(f"  Difference: {diff_pp:+.1f} pp")

    if s_n and ns_n and s_rate != ns_rate:
        z, p = two_proportion_z(s_yes, s_n, ns_yes, ns_n)
        sw.w(f"  Two-proportion z: z = {z:.2f}, {apa_p(p)} (one-sided)")
    elif s_n and ns_n:
        sw.w("  Two-proportion z: n/a (identical yes-rates)")


def _write_per_persona_detectability(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("PER-PERSONA DETECTABILITY (description Yes-rate)")
    sw.w("  Stereotyped personas (true-positive rate):")
    stereo = df[df["condition"] == "stereo"]
    for persona_id, group in stereo.groupby("persona_id", sort=True):
        yes = int(group["description_yes"].sum())
        n = int(group["description_yes"].notna().sum())
        short = PERSONA_SHORT.get(persona_id, persona_id)
        sw.w(f"    {short:9s}: {yes / n:6.1%} ({yes}/{n})" if n else f"    {short:9s}: n/a")

    sw.w("  Non-stereotyped personas (false-positive rate):")
    non_stereo = df[df["condition"] == "non_stereo"]
    for persona_id, group in non_stereo.groupby("persona_id", sort=True):
        yes = int(group["description_yes"].sum())
        n = int(group["description_yes"].notna().sum())
        short = PERSONA_SHORT.get(persona_id, persona_id)
        sw.w(f"    {short:9s}: {yes / n:6.1%} ({yes}/{n})" if n else f"    {short:9s}: n/a")


def _write_image_vs_description(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("RQ7 — IMAGE vs DESCRIPTION CONGRUENCE")
    stereo = df[df["condition"] == "stereo"]
    non_stereo = df[df["condition"] == "non_stereo"]

    for label, subset in [("stereo", stereo), ("non-stereo", non_stereo)]:
        yes = int(subset["image_yes"].sum())
        n = int(subset["image_yes"].notna().sum())
        sw.w(f"  Image Yes-rate {label:10s}: {yes / n:.1%} ({yes}/{n})" if n else f"  Image Yes-rate {label}: n/a")

    s_img = int(stereo["image_yes"].sum())
    s_n = int(stereo["image_yes"].notna().sum())
    ns_img = int(non_stereo["image_yes"].sum())
    ns_n = int(non_stereo["image_yes"].notna().sum())
    if s_n and ns_n:
        z, p = two_proportion_z(s_img, s_n, ns_img, ns_n)
        sw.w(f"  Two-proportion z (img_yes, stereo vs non-stereo): z = {z:.2f}, {apa_p(p)}")

    valid = df.dropna(subset=["image_yes", "description_yes"])
    if len(valid) >= 5:
        table = pd.crosstab(valid["description_yes"], valid["image_yes"])
        if table.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(table)
            n = len(valid)
            phi = (chi2 / n) ** 0.5
            sw.w(f"  img_yes x str_yes association: chi2(1) = {chi2:.2f}, {apa_p(p)}, phi = {phi:.2f}")


def _citation_rate(flaggers: pd.DataFrame, col: str) -> float:
    if flaggers.empty:
        return float("nan")
    values = flaggers[col].astype(str).str.strip().str.lower()
    return float(values.isin({"yes", "true", "1"}).mean())


def _write_stereotype_attribution(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("RQ3 — STEREOTYPE-TYPE ATTRIBUTION")
    sw.w("  Note: citation rates below are conditional on description_stereotype == Yes.")
    flaggers = df[df["description_stereotype_answer"].astype(str).str.strip().str.lower() == "yes"].copy()

    for condition in ["non_stereo", "stereo"]:
        subset = flaggers[flaggers["condition"] == condition]
        if subset.empty:
            continue
        sw.w(
            f"    {condition} (N flaggers = {len(subset)}): "
            f"age {_citation_rate(subset, 'stereotype_age'):.2f}, "
            f"gender {_citation_rate(subset, 'stereotype_gender'):.2f}, "
            f"occupation {_citation_rate(subset, 'stereotype_occupation'):.2f}, "
            f"other {_citation_rate(subset, 'stereotype_other'):.2f}"
        )

    stereo_flaggers = flaggers[flaggers["condition"] == "stereo"]
    if not stereo_flaggers.empty:
        sw.w(f"  Stereo-only summary (N = {len(stereo_flaggers)} flaggers): proportion citing each dimension")
        for label, col in [
            ("Age", "stereotype_age"),
            ("Gender", "stereotype_gender"),
            ("Occupation", "stereotype_occupation"),
            ("Other", "stereotype_other"),
        ]:
            sw.w(f"    {label:12s}: {_citation_rate(stereo_flaggers, col):.1%}")


def _write_use_rel(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("RQ4 — USE / RELATABILITY TRADE-OFFS")
    sw.w("  Usefulness maps to understand_group; relatability maps to relatable Likert items.")
    stereo = df[df["condition"] == "stereo"]
    non_stereo = df[df["condition"] == "non_stereo"]
    sw.w(
        f"  Usefulness: stereo M = {stereo['usefulness_num'].mean():.2f}, "
        f"non-stereo M = {non_stereo['usefulness_num'].mean():.2f}"
    )
    sw.w(
        f"  Relatability: stereo M = {stereo['relatable_num'].mean():.2f}, "
        f"non-stereo M = {non_stereo['relatable_num'].mean():.2f}"
    )

    for condition in ["stereo", "non_stereo"]:
        subset = df[df["condition"] == condition]
        flagged = subset[subset["description_yes"] == 1]
        unflagged = subset[subset["description_yes"] == 0]
        if len(flagged) and len(unflagged):
            sw.w(
                f"    [{condition}] flagged usefulness M = {flagged['usefulness_num'].mean():.2f} "
                f"(n={len(flagged)}), unflagged M = {unflagged['usefulness_num'].mean():.2f} (n={len(unflagged)})"
            )
            sw.w(
                f"    [{condition}] flagged relatability M = {flagged['relatable_num'].mean():.2f} "
                f"(n={len(flagged)}), unflagged M = {unflagged['relatable_num'].mean():.2f} (n={len(unflagged)})"
            )


def _write_majority_agreement(sw: SummaryWriter, df: pd.DataFrame, rater_col: str) -> None:
    sw.h("RQ9 — INTER-RATER AGREEMENT")
    sw.w(f"  Primary metric: majority agreement on description Yes/No (rater column = {rater_col}).")

    for condition in ["stereo", "non_stereo"]:
        subset = df[df["condition"] == condition]
        persona_stats = []
        for persona_id, group in subset.groupby("persona_id", sort=True):
            agree = majority_agreement(group["description_yes"])
            yes_rate = float(group["description_yes"].mean())
            persona_stats.append((persona_id, agree, yes_rate, len(group)))
        if persona_stats:
            mean_agree = sum(item[1] for item in persona_stats) / len(persona_stats)
            sw.w(f"  [{condition}] description mean majority agreement: {mean_agree:.3f}")

    sw.w("  Per-persona description majority agreement (sorted):")
    rows = []
    for persona_id, group in df.groupby("persona_id", sort=True):
        rows.append(
            {
                "persona_id": persona_id,
                "condition": group["condition"].iloc[0],
                "agree": majority_agreement(group["description_yes"]),
                "yes_rate": float(group["description_yes"].mean()),
                "n": len(group),
            }
        )
    rows.sort(key=lambda item: item["agree"])
    for row in rows:
        short = PERSONA_SHORT.get(row["persona_id"], row["persona_id"])
        sw.w(
            f"    {row['condition']:10s} {short:9s}: agree={row['agree']:.2f}, "
            f"yes={row['yes_rate']:.2f} (n={row['n']})"
        )

    icc_rows = []
    for label, condition in [("All personas", None), ("Stereotyped", "stereo"), ("Non-stereotyped", "non_stereo")]:
        subset = df if condition is None else df[df["condition"] == condition]
        icc = icc21(subset, "persona_id", rater_col, "description_yes")
        icc_rows.append((label, icc))
    sw.w("  ICC(2,1) for description_yes:")
    for label, icc in icc_rows:
        if icc["icc21"] is None:
            sw.w(f"    [{label}] ICC(2,1) = n/a ({icc['note']})")
        else:
            sw.w(
                f"    [{label}] ICC(2,1) = {icc['icc21']}, ICC(2,k) = {icc['icc2k']} "
                f"(targets={icc['n_targets']}, raters={icc['n_raters']}, n={icc['n_obs']})"
            )

    sw.w("  Image (img_yes) mean majority agreement:")
    for condition in ["stereo", "non_stereo"]:
        subset = df[df["condition"] == condition]
        persona_stats = [
            majority_agreement(group["image_yes"])
            for _, group in subset.groupby("persona_id", sort=True)
        ]
        if persona_stats:
            sw.w(f"    [{condition}]: {sum(persona_stats) / len(persona_stats):.3f}")


def _write_model_internal_consistency(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("INTERNAL CONSISTENCY BY MODEL (5 runs per persona)")
    rows = []
    for (persona_id, model_key), group in df.groupby(["persona_id", "model_key"], sort=True):
        image = consistency_stats(group["image_stereotype_answer"])
        desc = consistency_stats(group["description_stereotype_answer"])
        rows.append(
            {
                "model_key": model_key,
                "image_consistency": image["consistency"],
                "description_consistency": desc["consistency"],
            }
        )
    summary = pd.DataFrame(rows).groupby("model_key", sort=True).mean(numeric_only=True)
    for model_key, row in summary.iterrows():
        sw.w(
            f"  {model_key:12s}: image {row['image_consistency']:.3f}, "
            f"description {row['description_consistency']:.3f}"
        )
    sw.w(
        f"  Overall mean: image {summary['image_consistency'].mean():.3f}, "
        f"description {summary['description_consistency'].mean():.3f}"
    )


def _write_model_breakdown(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("MODEL BREAKDOWN (description Yes-rate by condition)")
    for model_key, model_df in df.groupby("model_key", sort=True):
        stereo = model_df[model_df["condition"] == "stereo"]
        non_stereo = model_df[model_df["condition"] == "non_stereo"]
        s_rate = stereo["description_yes"].mean()
        ns_rate = non_stereo["description_yes"].mean()
        sw.w(
            f"  {model_key:12s}: stereo {s_rate:.1%}, non-stereo {ns_rate:.1%}, "
            f"delta {(s_rate - ns_rate) * 100:+.1f} pp"
        )


def _write_ground_truth(sw: SummaryWriter, df: pd.DataFrame) -> None:
    sw.h("GROUND-TRUTH AGREEMENT")
    for modality, col in [("Image", "image_yes"), ("Description", "description_yes")]:
        match = ((df[col] == 1) == df["is_stereo_bool"]).mean()
        stereo = df[df["is_stereo_bool"]]
        non_stereo = df[~df["is_stereo_bool"]]
        sw.w(
            f"  {modality:12s} overall: {match:.1%} | "
            f"stereo TP rate {stereo[col].mean():.1%} | "
            f"non-stereo FP rate {non_stereo[col].mean():.1%}"
        )


def generate_standard_llm_summary(
    results_dir: Path,
    config_path: Path,
    personas_path: Path,
) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_keys = set(config["models"])
    runs_per_model = int(config["runs_per_model"])

    personas = load_personas(personas_path)
    all_runs = load_all_runs(results_dir, personas, model_keys)
    all_runs = all_runs[all_runs["run_index"] <= runs_per_model].copy()
    df = _prepare_runs(all_runs)

    sw = SummaryWriter()
    sw.h("PERSONA STEREOTYPES EVALUATION — LLM RESULTS SUMMARY")
    sw.w(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    sw.w(f"Source: {results_dir.resolve()}")
    sw.w(f"Design: {personas['persona_id'].nunique()} personas × {len(model_keys)} models × {runs_per_model} runs")
    sw.w(f"N evaluations: {len(df)}")
    sw.w(f"Models: {', '.join(sorted(model_keys))}")

    _write_ground_truth(sw, df)
    _write_manipulation_check(sw, df)
    _write_per_persona_detectability(sw, df)
    _write_model_breakdown(sw, df)
    _write_model_internal_consistency(sw, df)
    _write_image_vs_description(sw, df)
    _write_stereotype_attribution(sw, df)
    _write_use_rel(sw, df)

    model_means = (
        df.groupby(["persona_id", "model_key"], sort=True)["description_yes"]
        .mean()
        .reset_index()
    )
    sw.h("RQ9 — INTER-MODEL AGREEMENT")
    sw.w("  ICC(2,1) with personas as targets and models as raters (mean yes across 5 runs).")
    for label, condition in [("All personas", None), ("Stereotyped", "stereo"), ("Non-stereotyped", "non_stereo")]:
        subset = model_means if condition is None else model_means.merge(
            df[["persona_id", "condition"]].drop_duplicates(), on="persona_id", how="left"
        )
        if condition is not None:
            subset = subset[subset["condition"] == condition]
        icc = icc21(subset, "persona_id", "model_key", "description_yes")
        if icc["icc21"] is None:
            sw.w(f"    [{label}] ICC(2,1) = n/a ({icc['note']})")
        else:
            sw.w(
                f"    [{label}] ICC(2,1) = {icc['icc21']}, ICC(2,k) = {icc['icc2k']} "
                f"(targets={icc['n_targets']}, raters={icc['n_raters']})"
            )

    sw.h("END OF SUMMARY")
    return sw.text()


def generate_mirror_summary(mirror_dir: Path) -> str:
    records = load_mirror_results_successful(mirror_dir)
    if len(records) != 510:
        raise SystemExit(f"Expected 510 mirror evaluations, found {len(records)}")

    df = _mirror_records_to_df(records)
    model_name = records[0].get("model_display_name", "Claude Sonnet 4.6")

    sw = SummaryWriter()
    sw.h("PERSONA STEREOTYPES EVALUATION — HUMAN-MIRROR LLM RESULTS SUMMARY")
    sw.w(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    sw.w(f"Source: {mirror_dir.resolve()}")
    sw.w(f"Model: {model_name}")
    sw.w("Design: 85 virtual participants × 6 personas (3 stereotyped + 3 non-stereotyped)")
    sw.w("Assignment schedule: matched to Prolific human study (partial overlap, 6/12 personas per participant)")
    sw.w(f"N participants: {df['participant_id'].nunique()}")
    sw.w(f"N persona ratings: {len(df)}")

    _write_ground_truth(sw, df)
    _write_manipulation_check(sw, df)

    stereo = df[df["condition"] == "stereo"]
    non_stereo = df[df["condition"] == "non_stereo"]
    per_participant = (
        df.groupby("participant_id")
        .apply(
            lambda group: pd.Series(
                {
                    "stereo": group[group["condition"] == "stereo"]["description_yes"].mean(),
                    "non_stereo": group[group["condition"] == "non_stereo"]["description_yes"].mean(),
                }
            ),
            include_groups=False,
        )
        .dropna()
    )
    if len(per_participant) >= 5 and (per_participant["stereo"] != per_participant["non_stereo"]).any():
        try:
            w = wilcoxon(per_participant["stereo"], per_participant["non_stereo"], alternative="greater")
            sw.w(
                f"  Wilcoxon paired (participant stereo > non-stereo): "
                f"W = {w.statistic:.1f}, {apa_p(w.pvalue)}, N = {len(per_participant)}"
            )
        except ValueError:
            pass
    elif len(per_participant) >= 5:
        sw.w("  Wilcoxon paired: n/a (identical stereo and non-stereo rates within participants)")

    _write_per_persona_detectability(sw, df)
    _write_image_vs_description(sw, df)
    _write_stereotype_attribution(sw, df)
    _write_use_rel(sw, df)
    _write_majority_agreement(sw, df, rater_col="participant_id")

    sw.h("END OF SUMMARY")
    return sw.text()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM and mirror study text summaries.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--personas", type=Path, default=Path("data/personas.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_summary = generate_standard_llm_summary(args.results_dir, args.config, args.personas)
    mirror_summary = generate_mirror_summary(args.mirror_dir)

    llm_path = args.output_dir / "llm_results_summary.txt"
    mirror_path = args.output_dir / "mirror_results_summary.txt"
    llm_path.write_text(llm_summary, encoding="utf-8")
    mirror_path.write_text(mirror_summary, encoding="utf-8")

    print(f"Saved: {llm_path}")
    print(f"Saved: {mirror_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
