#!/usr/bin/env python3
"""Recalculate corrected RQ2 and RQ3 statistics for the five-LLM and mirror studies."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon

from export_analysis import load_personas
from generate_llm_summaries import _mirror_records_to_df
from pick_mirror_model import DEFAULT_MODEL, mirror_output_dir
from src.human_data import likert_to_numeric, load_human_study
from src.llm_loader import (
    build_validation_summary,
    load_clean_llm_runs,
    load_config,
    prepare_llm_analysis_frame,
    write_clean_aggregates,
    write_validation_summary,
)
from src.mirror_storage import load_mirror_results_successful
from src.persona_labels import INTENDED_DIMS, PERSONA_P_IDS, persona_label, persona_suffix
from src.stats_reporting import (
    LLM_LMM_SPEC,
    apa_p,
    apa_pct,
    cited_dims_from_row,
    cohens_kappa,
    cramers_v,
    fit_crossed_lmm,
    fleiss_kappa_modal,
    jaccard,
    latex_pct,
    likert_descriptives,
    lmm_result_line,
    mcnemar_attribution_batch,
    mcnemar_binary,
    paired_likert_test,
    parse_dim_flag,
    within_llm_consistency,
)


def run_reproducibility_assertions(df: pd.DataFrame) -> None:
    prep = prepare_llm_analysis_frame(df)
    assert len(prep) == 300
    assert prep["persona_id"].nunique() == 12
    assert prep["model_key"].nunique() == 5
    assert prep["run_index"].between(1, 5).all()
    counts = prep.groupby(["persona_id", "model_key"]).size()
    assert (counts == 5).all()
    cond = prep.groupby("condition").size()
    assert int(cond["stereo"]) == 150
    assert int(cond["non_stereo"]) == 150
    for model in prep["model_key"].unique():
        m = prep[prep["model_key"] == model]
        assert int((m["condition"] == "stereo").sum()) == 30
        assert int((m["condition"] == "non_stereo").sum()) == 30
    overall_nsp = int(prep.loc[prep["condition"] == "non_stereo", "description_yes"].sum())
    model_nsp = prep.loc[prep["condition"] == "non_stereo"].groupby("model_key")["description_yes"].sum()
    assert int(model_nsp.sum()) == overall_nsp
    persona_nsp = prep.loc[prep["condition"] == "non_stereo"].groupby("persona_id")["description_yes"].sum()
    assert int(persona_nsp.sum()) == overall_nsp


def analyze_rq2(df: pd.DataFrame) -> dict[str, Any]:
    prep = prepare_llm_analysis_frame(df)
    results: dict[str, Any] = {"model_specification": LLM_LMM_SPEC}

    stereo = prep[prep["condition"] == "stereo"]
    non_stereo = prep[prep["condition"] == "non_stereo"]
    sp_yes = int(stereo["description_yes"].sum())
    sp_n = int(stereo["description_yes"].notna().sum())
    nsp_yes = int(non_stereo["description_yes"].sum())
    nsp_n = int(non_stereo["description_yes"].notna().sum())
    results["description_detection"] = {
        "sp_yes": sp_yes,
        "sp_n": sp_n,
        "sp_rate": sp_yes / sp_n,
        "nsp_yes": nsp_yes,
        "nsp_n": nsp_n,
        "nsp_rate": nsp_yes / nsp_n,
        "diff_pp": 100 * (sp_yes / sp_n - nsp_yes / nsp_n),
    }

    prep["condition_bin"] = (prep["condition"] == "stereo").astype(int)
    lmm, lmm_note = fit_crossed_lmm(prep, "description_yes ~ condition_bin")
    results["description_lmm"] = {
        "note": lmm_note,
        "condition": lmm_result_line(lmm, "condition_bin", "Condition (SP vs NSP)"),
    }

    # By model and condition
    model_cond_rows = []
    for model, g in prep.groupby("model_key"):
        for cond in ["stereo", "non_stereo"]:
            sub = g[g["condition"] == cond]
            yes = int(sub["description_yes"].sum())
            n = len(sub)
            model_cond_rows.append(
                {"model_key": model, "condition": cond, "yes": yes, "n": n, "rate": yes / n if n else None}
            )
    results["description_by_model_condition"] = model_cond_rows

    persona_rows = []
    for persona_id, g in prep.groupby("persona_id"):
        name = g["name"].iloc[0]
        for cond in ["stereo", "non_stereo"]:
            sub = g[g["condition"] == cond]
            yes = int(sub["description_yes"].sum())
            n = len(sub)
            persona_rows.append(
                {
                    "persona_id": persona_id,
                    "persona_label": persona_label(persona_id, name),
                    "condition": cond,
                    "yes": yes,
                    "n": n,
                    "rate": yes / n if n else None,
                }
            )
    results["description_by_persona_condition"] = persona_rows

    # NSP model comparison chi-square
    nsp = prep[prep["condition"] == "non_stereo"]
    ct = pd.crosstab(nsp["model_key"], nsp["description_yes"].astype(int))
    chi2, p, dof, _ = chi2_contingency(ct.values)
    n = int(ct.values.sum())
    v = cramers_v(chi2, n, ct.shape[1], ct.shape[0])
    results["nsp_model_chi_square"] = {
        "chi2": round(float(chi2), 2),
        "df": int(dof),
        "n": n,
        "p": float(p),
        "cramers_v": round(v, 3),
    }

    # Within-LLM response consistency
    cons_rows = []
    for (persona_id, model_key), g in prep.groupby(["persona_id", "model_key"]):
        cons_rows.append(
            {
                "persona_id": persona_id,
                "model_key": model_key,
                "condition": g["condition"].iloc[0],
                "description_consistency": within_llm_consistency(g["description_stereotype_answer"]),
                "image_consistency": within_llm_consistency(g["image_stereotype_answer"]),
            }
        )
    cons_df = pd.DataFrame(cons_rows)
    results["within_llm_consistency"] = {
        "overall_description": round(float(cons_df["description_consistency"].mean()), 3),
        "overall_image": round(float(cons_df["image_consistency"].mean()), 3),
        "by_model": cons_df.groupby("model_key")[["description_consistency", "image_consistency"]]
        .mean()
        .round(3)
        .reset_index()
        .to_dict(orient="records"),
        "by_condition": cons_df.groupby("condition")[["description_consistency", "image_consistency"]]
        .mean()
        .round(3)
        .reset_index()
        .to_dict(orient="records"),
    }

    # Inter-LLM agreement
    sp_all_yes = bool((stereo["description_yes"] == 1).all())
    results["inter_llm_agreement"] = {
        "sp_all_positive": sp_all_yes,
        "sp_note": (
            "All SP description ratings were positive across all independent evaluations; "
            "inter-LLM disagreement was therefore undefined (zero variance)."
            if sp_all_yes
            else "SP description ratings showed variation across LLMs."
        ),
        "nsp_fleiss": fleiss_kappa_modal(prep, "description_yes", "non_stereo"),
    }

    # Image ratings
    sp_img_yes = int(stereo["image_yes"].sum())
    nsp_img_yes = int(non_stereo["image_yes"].sum())
    results["image"] = {
        "sp_yes": sp_img_yes,
        "sp_n": sp_n,
        "nsp_yes": nsp_img_yes,
        "nsp_n": nsp_n,
    }
    img_lmm, img_note = fit_crossed_lmm(prep, "image_yes ~ condition_bin")
    results["image_lmm"] = {"note": img_note, "condition": lmm_result_line(img_lmm, "condition_bin")}
    ct_img = pd.crosstab(prep["image_yes"].astype(int), prep["description_yes"].astype(int))
    chi2_i, p_i, dof_i, _ = chi2_contingency(ct_img.values)
    n_i = int(ct_img.values.sum())
    phi = math.sqrt(chi2_i / n_i) if n_i else float("nan")
    results["image_description_chi_square"] = {
        "chi2": round(float(chi2_i), 2),
        "df": int(dof_i),
        "n": n_i,
        "p": float(p_i),
        "phi": round(float(phi), 3),
    }
    img_adj_lmm, img_adj_note = fit_crossed_lmm(prep, "image_yes ~ condition_bin + description_yes")
    results["image_adjusted_lmm"] = {
        "note": img_adj_note,
        "description": lmm_result_line(img_adj_lmm, "description_yes"),
        "condition": lmm_result_line(img_adj_lmm, "condition_bin"),
    }

    # Usefulness and relatability
    for dv, col in [("usefulness", "usefulness_num"), ("relatability", "relatable_num")]:
        desc = {}
        for cond in ["stereo", "non_stereo"]:
            sub = prep[prep["condition"] == cond]
            desc[cond] = likert_descriptives(sub[col])
        lmm_dv, note_dv = fit_crossed_lmm(prep, f"{col} ~ condition_bin")
        results[dv] = {
            "descriptives": desc,
            "lmm": {"note": note_dv, "condition": lmm_result_line(lmm_dv, "condition_bin")},
        }

    nsp_flag = non_stereo.groupby(non_stereo["description_yes"] == 1)
    flagged = non_stereo[non_stereo["description_yes"] == 1]
    unflagged = non_stereo[non_stereo["description_yes"] == 0]
    results["nsp_flagged_quality"] = {
        "usefulness_flagged": likert_descriptives(flagged["usefulness_num"]),
        "usefulness_unflagged": likert_descriptives(unflagged["usefulness_num"]),
        "relatability_flagged": likert_descriptives(flagged["relatable_num"]),
        "relatability_unflagged": likert_descriptives(unflagged["relatable_num"]),
    }
    for dv, col in [("usefulness", "usefulness_num"), ("relatability", "relatable_num")]:
        flagged_bin = (non_stereo["description_yes"] == 1).astype(int)
        tmp = non_stereo.assign(flagged_bin=flagged_bin)
        lmm_f, note_f = fit_crossed_lmm(tmp, f"{col} ~ flagged_bin")
        results["nsp_flagged_quality"][f"{dv}_lmm"] = {
            "note": note_f,
            "flagged": lmm_result_line(lmm_f, "flagged_bin"),
        }

    # Dimensional attribution
    flagged_desc = prep[prep["description_yes"] == 1].copy()
    dim_rows = []
    for cond in ["stereo", "non_stereo"]:
        sub = flagged_desc[flagged_desc["condition"] == cond]
        n = len(sub)
        for dim, col in [
            ("age", "stereotype_age"),
            ("gender", "stereotype_gender"),
            ("occupation", "stereotype_occupation"),
            ("other", "stereotype_other"),
        ]:
            count = int(sub[col].apply(parse_dim_flag).sum())
            dim_rows.append({"condition": cond, "dimension": dim, "count": count, "n": n, "rate": count / n if n else None})
    results["attribution"] = dim_rows

    jaccard_rows = []
    for persona_id, g in flagged_desc[flagged_desc["condition"] == "stereo"].groupby("persona_id"):
        suffix = persona_suffix(persona_id)
        intended = INTENDED_DIMS.get(suffix, set())
        scores = []
        for _, row in g.iterrows():
            cited = cited_dims_from_row(row) - {"Other"}
            score = jaccard(cited, intended)
            if score is not None:
                scores.append(score)
        if scores:
            jaccard_rows.append(
                {
                    "persona_id": persona_id,
                    "persona_label": persona_label(persona_id, g["name"].iloc[0]),
                    "mean": round(float(np.mean(scores)), 3),
                    "sd": round(float(np.std(scores, ddof=1)), 3) if len(scores) > 1 else 0.0,
                    "n": len(scores),
                }
            )
    results["jaccard"] = {
        "persona_level": jaccard_rows,
        "overall_mean": round(float(np.mean([r["mean"] for r in jaccard_rows])), 2) if jaccard_rows else None,
        "overall_sd": round(float(np.std([r["mean"] for r in jaccard_rows], ddof=1)), 2)
        if len(jaccard_rows) > 1
        else None,
        "min": min(r["mean"] for r in jaccard_rows) if jaccard_rows else None,
        "max": max(r["mean"] for r in jaccard_rows) if jaccard_rows else None,
        "primary_dimension_note": (
            "The LLM JSON output stores stereotype dimensions as unordered boolean fields; "
            "selection order is not preserved. A mutually exclusive primary-dimension "
            "inferential comparison therefore cannot be reproduced for the LLM data."
        ),
    }

    return results


def analyze_rq3(
    llm_df: pd.DataFrame,
    human_long: pd.DataFrame,
    mirror_prep: pd.DataFrame,
) -> dict[str, Any]:
    llm = prepare_llm_analysis_frame(llm_df)
    human = human_long.rename(columns={"pid": "participant_id"}).copy()
    human["usefulness_num"] = human["usefulness_answer"].map(likert_to_numeric)
    human["relatable_num"] = human["relatable_answer"].map(likert_to_numeric)
    mirror = mirror_prep.copy()

    results: dict[str, Any] = {}

    human_persona = (
        human.groupby(["persona_id", "condition"])
        .agg(description_yes=("description_yes", "mean"), image_yes=("image_yes", "mean"))
        .reset_index()
    )
    llm_persona = (
        llm.groupby(["persona_id", "condition"])
        .agg(description_yes=("description_yes", "mean"), image_yes=("image_yes", "mean"), name=("name", "first"))
        .reset_index()
    )
    llm_model_persona = (
        llm.groupby(["persona_id", "model_key", "condition"])
        .agg(description_yes=("description_yes", "mean"), image_yes=("image_yes", "mean"), name=("name", "first"))
        .reset_index()
    )
    results["human_vs_llm_persona"] = {
        "human": human_persona.to_dict(orient="records"),
        "llm_aggregate": llm_persona.to_dict(orient="records"),
        "llm_by_model": llm_model_persona.to_dict(orient="records"),
    }

    paired = human.merge(
        mirror,
        on=["participant_id", "persona_id", "condition"],
        suffixes=("_human", "_mirror"),
    )
    for dim, mirror_col in [
        ("age", "stereotype_age"),
        ("gender", "stereotype_gender"),
        ("occupation", "stereotype_occupation"),
        ("other", "stereotype_other"),
    ]:
        source_col = mirror_col if mirror_col in paired.columns else f"{mirror_col}_mirror"
        paired[f"mirror_cited_{dim}"] = paired[source_col].apply(parse_dim_flag).astype(int)

    paired_binary: dict[str, Any] = {}
    for cond, label in [("stereo", "sp"), ("non_stereo", "nsp")]:
        sub = paired[paired["condition"] == cond]
        paired_binary[f"description_{label}"] = mcnemar_binary(
            sub["description_yes_human"], sub["description_yes_mirror"]
        )
        paired_binary[f"image_{label}"] = mcnemar_binary(sub["image_yes_human"], sub["image_yes_mirror"])
    results["paired_binary"] = paired_binary

    human_dim_cols = {
        "age": "cited_age_human" if "cited_age_human" in paired.columns else "cited_age",
        "gender": "cited_gender_human" if "cited_gender_human" in paired.columns else "cited_gender",
        "occupation": "cited_occupation_human"
        if "cited_occupation_human" in paired.columns
        else "cited_occupation",
        "other": "cited_other_human" if "cited_other_human" in paired.columns else "cited_other",
    }
    mirror_dim_cols = {
        "age": "mirror_cited_age",
        "gender": "mirror_cited_gender",
        "occupation": "mirror_cited_occupation",
        "other": "mirror_cited_other",
    }
    results["paired_attribution"] = {
        "stereo": mcnemar_attribution_batch(
            paired, condition="stereo", human_cols=human_dim_cols, mirror_cols=mirror_dim_cols
        ),
        "non_stereo": mcnemar_attribution_batch(
            paired, condition="non_stereo", human_cols=human_dim_cols, mirror_cols=mirror_dim_cols
        ),
    }

    quality: dict[str, Any] = {}
    for cond in ["stereo", "non_stereo"]:
        sub = paired[paired["condition"] == cond]
        quality[cond] = {
            "usefulness": paired_likert_test(sub["usefulness_num_human"], sub["usefulness_num_mirror"]),
            "relatability": paired_likert_test(sub["relatable_num_human"], sub["relatable_num_mirror"]),
        }
    results["paired_quality"] = quality

    results["attribution_comparison_note"] = (
        "Paired McNemar tests compared binary dimension selections between human raters and "
        "GPT-5.4 mirror evaluations at the participant--persona level, with Holm correction "
        "across the four dimensions within each condition. Mutually exclusive primary-dimension "
        "comparisons cannot be reproduced because selection order was not preserved in the "
        "GPT-5.4 mirror output."
    )
    return results


def write_rq2_apa(results: dict[str, Any], path: Path) -> None:
    d = results["description_detection"]
    lines = [
        "% RQ2 — Five-LLM study (corrected 300 independent evaluations)",
        "",
        "\\subsection{Model specification}",
        results["model_specification"],
        "",
        "\\subsection{Description detection and perception}",
        f"SP positive description ratings: {latex_pct(d['sp_yes'], d['sp_n'])}.",
        f"NSP positive description ratings: {latex_pct(d['nsp_yes'], d['nsp_n'])}.",
        f"Difference: {d['diff_pp']:.1f} percentage points.",
        f"Mixed model: {results['description_lmm']['condition']['apa']}.",
        "",
        "\\subsection{Within-LLM response consistency}",
        f"Overall description within-LLM response consistency: $M = {results['within_llm_consistency']['overall_description']:.2f}$.",
        f"Overall image within-LLM response consistency: $M = {results['within_llm_consistency']['overall_image']:.2f}$.",
        "",
        "\\subsection{Inter-LLM agreement}",
        results["inter_llm_agreement"]["sp_note"],
    ]
    if results["inter_llm_agreement"]["nsp_fleiss"]["kappa"] is not None:
        fk = results["inter_llm_agreement"]["nsp_fleiss"]
        lines.append(
            f"NSP modal description agreement across LLMs: Fleiss' $\\kappa = {fk['kappa']:.3f}$, "
            f"observed agreement = {fk.get('observed_agreement', 'n/a')}."
        )
    img = results["image"]
    lines.extend(
        [
            "",
            "\\subsection{Image ratings}",
            f"SP image Yes: {latex_pct(img['sp_yes'], img['sp_n'])}.",
            f"NSP image Yes: {latex_pct(img['nsp_yes'], img['nsp_n'])}.",
            f"Condition mixed model: {results['image_lmm']['condition']['apa']}.",
            f"Image--description association: $\\chi^2({results['image_description_chi_square']['df']}) = "
            f"{results['image_description_chi_square']['chi2']:.2f}$, "
            f"$N = {results['image_description_chi_square']['n']}$, "
            f"{apa_p(results['image_description_chi_square']['p'])}, "
            f"$\\phi = {results['image_description_chi_square']['phi']:.3f}$.",
            "",
            "\\subsection{Usefulness and relatability}",
        ]
    )
    for dv in ["usefulness", "relatability"]:
        desc = results[dv]["descriptives"]
        lines.append(
            f"{dv.capitalize()} SP: $M = {desc['stereo']['mean']}$, $SD = {desc['stereo']['sd']}$, "
            f"$n = {desc['stereo']['n']}$, 95\\% CI $[{desc['stereo']['ci_low']}, {desc['stereo']['ci_high']}]$."
        )
        lines.append(
            f"{dv.capitalize()} NSP: $M = {desc['non_stereo']['mean']}$, $SD = {desc['non_stereo']['sd']}$, "
            f"$n = {desc['non_stereo']['n']}$, 95\\% CI $[{desc['non_stereo']['ci_low']}, {desc['non_stereo']['ci_high']}]$."
        )
        lines.append(f"Condition effect: {results[dv]['lmm']['condition']['apa']}.")
    lines.append(results["jaccard"]["primary_dimension_note"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rq3_apa(results: dict[str, Any], path: Path) -> None:
    lines = ["% RQ3 — Human vs GPT-5.4 mirror comparison", "", results["attribution_comparison_note"], ""]
    for key, val in results["paired_binary"].items():
        if isinstance(val, dict) and "apa" in val:
            lines.append(f"\\subsection{{{key.replace('_', ' ')}}}")
            lines.append(val["apa"])
    for cond in ["stereo", "non_stereo"]:
        block = results["paired_attribution"][cond]
        lines.append(f"\\subsection{{Attribution — {cond}}}")
        lines.append(f"$N_{{matched}} = {block['n_pairs']}$.")
        for dim_row in block["dimensions"]:
            lines.append(dim_row["apa"])
        lines.append(block["primary_dimension_note"])
    for cond, metrics in results["paired_quality"].items():
        lines.append(f"\\subsection{{Paired quality — {cond}}}")
        for name, stat in metrics.items():
            lines.append(f"{name.capitalize()}: {stat.get('apa', 'n/a')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalculate corrected RQ2/RQ3 statistics.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    personas = load_personas(Path("data/personas.csv"))
    clean_df, audit_df = load_clean_llm_runs(
        args.results_dir,
        personas,
        runs_per_model=cfg["runs_per_model"],
        model_keys=cfg["model_keys"],
    )
    run_reproducibility_assertions(clean_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.output_dir / "llm_filter_audit.csv", index=False)
    write_clean_aggregates(
        clean_df,
        args.results_dir / "all_evaluations_clean.csv",
        args.results_dir / "all_evaluations_clean.json",
    )
    summary = build_validation_summary(clean_df, runs_per_model=cfg["runs_per_model"], model_keys=cfg["model_keys"])
    write_validation_summary(summary, args.output_dir / "llm_clean_validation.txt")

    rq2 = analyze_rq2(clean_df)
    human_long, _ = load_human_study(path=Path("data/human_study_cleaned.csv"), with_attribution=True)
    mirror_records = load_mirror_results_successful(mirror_output_dir(DEFAULT_MODEL))
    mirror_prep = _mirror_records_to_df(mirror_records)
    rq3 = analyze_rq3(clean_df, human_long, mirror_prep)

    write_rq2_apa(rq2, args.output_dir / "rq2_apa_results.txt")
    write_rq3_apa(rq3, args.output_dir / "rq3_apa_results.txt")

    stats_bundle = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validation": summary,
        "rq2": rq2,
        "rq3": rq3,
    }
    (args.output_dir / "rq2_rq3_statistics.json").write_text(
        json.dumps(stats_bundle, indent=2, default=str),
        encoding="utf-8",
    )

    pd.DataFrame(rq2["description_by_model_condition"]).to_csv(args.output_dir / "rq2_tables.csv", index=False)
    pd.DataFrame(rq3["human_vs_llm_persona"]["llm_by_model"]).to_csv(args.output_dir / "rq3_tables.csv", index=False)

    print((args.output_dir / "llm_clean_validation.txt").read_text())
    d = rq2["description_detection"]
    print(f"\nCorrected SP description positives: {d['sp_yes']}/{d['sp_n']} ({d['sp_rate']:.1%})")
    print(f"Corrected NSP description positives: {d['nsp_yes']}/{d['nsp_n']} ({d['nsp_rate']:.1%})")
    print("\nNSP positives by LLM:")
    for row in rq2["description_by_model_condition"]:
        if row["condition"] == "non_stereo":
            print(f"  {row['model_key']}: {row['yes']}/{row['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
