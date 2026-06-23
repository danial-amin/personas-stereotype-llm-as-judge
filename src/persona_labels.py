"""Persona identifiers and stereotype-dimension mappings for manuscript outputs."""

from __future__ import annotations

PERSONA_P_IDS: dict[str, str] = {
    "a_us_marcus_linq": "P1",
    "a_s_aisha": "P2",
    "a_s_ray": "P3",
    "a_us_margaret": "P4",
    "a_us_marcus": "P5",
    "a_s_sofia": "P6",
    "a_s_bernie": "P7",
    "a_us_eleanor": "P8",
    "a_s_james": "P9",
    "a_us_mei": "P10",
    "a_us_richard": "P11",
    "a_s_caroline": "P12",
}

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

PERSONA_ID_TO_SUFFIX: dict[str, str] = {v: k for k, v in PERSONA_SUFFIX_TO_ID.items()}

# Intended inserted dimensions (from persona_stereotypes_analysis_v4_DA.ipynb).
INTENDED_DIMS: dict[str, set[str]] = {
    "bernie": {"Age", "Occupation"},
    "sofia": {"Gender", "Age", "Occupation"},
    "caroline": {"Gender", "Age"},
    "james": {"Occupation", "Gender"},
    "ray": {"Age", "Gender"},
    "aisha": {"Gender", "Age"},
}

MIRROR_EXCLUDED_DIRS = {"human_mirror_experiment", "human_mirror_gpt-5.4"}


def persona_label(persona_id: str, name: str | None = None) -> str:
    pid = PERSONA_P_IDS.get(persona_id, persona_id)
    if name:
        return f"{pid} ({name})"
    return pid


def persona_suffix(persona_id: str) -> str:
    return PERSONA_ID_TO_SUFFIX.get(persona_id, persona_id)
