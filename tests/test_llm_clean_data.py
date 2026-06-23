"""Tests for corrected five-LLM data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from export_analysis import load_personas
from src.llm_loader import (
    build_validation_summary,
    load_clean_llm_runs,
    load_config,
    prepare_llm_analysis_frame,
)


@pytest.fixture(scope="module")
def clean_df():
    cfg = load_config(Path("config.yaml"))
    personas = load_personas(Path("data/personas.csv"))
    df, _audit = load_clean_llm_runs(
        Path("results"),
        personas,
        runs_per_model=cfg["runs_per_model"],
        model_keys=cfg["model_keys"],
    )
    return df


def test_clean_row_count(clean_df):
    assert len(clean_df) == 300


def test_clean_structure(clean_df):
    prep = prepare_llm_analysis_frame(clean_df)
    assert prep["persona_id"].nunique() == 12
    assert prep["model_key"].nunique() == 5
    assert prep["run_index"].between(1, 5).all()
    counts = prep.groupby(["persona_id", "model_key"]).size()
    assert (counts == 5).all()


def test_condition_counts(clean_df):
    prep = prepare_llm_analysis_frame(clean_df)
    cond = prep.groupby("condition").size()
    assert int(cond["stereo"]) == 150
    assert int(cond["non_stereo"]) == 150


def test_model_condition_balance(clean_df):
    prep = prepare_llm_analysis_frame(clean_df)
    for model in prep["model_key"].unique():
        m = prep[prep["model_key"] == model]
        assert int((m["condition"] == "stereo").sum()) == 30
        assert int((m["condition"] == "non_stereo").sum()) == 30


def test_nsp_positive_count_consistency(clean_df):
    prep = prepare_llm_analysis_frame(clean_df)
    overall = int(prep.loc[prep["condition"] == "non_stereo", "description_yes"].sum())
    by_model = int(
        prep.loc[prep["condition"] == "non_stereo"].groupby("model_key")["description_yes"].sum().sum()
    )
    by_persona = int(
        prep.loc[prep["condition"] == "non_stereo"].groupby("persona_id")["description_yes"].sum().sum()
    )
    assert by_model == overall
    assert by_persona == overall


def test_validation_summary_passes(clean_df):
    cfg = load_config(Path("config.yaml"))
    summary = build_validation_summary(
        clean_df,
        runs_per_model=cfg["runs_per_model"],
        model_keys=cfg["model_keys"],
    )
    assert summary["total_evaluations"] == 300
    assert summary["sp_evaluations"] == 150
    assert summary["nsp_evaluations"] == 150
    assert all(n == 60 for n in summary["evaluations_by_model"].values())
    assert all(n == 25 for n in summary["evaluations_by_persona"].values())
    assert all(n == 5 for n in summary["evaluations_by_persona_model"].values())
