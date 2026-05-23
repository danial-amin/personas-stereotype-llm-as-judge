from __future__ import annotations

from pathlib import Path

from src.models import Persona


def build_prompt(template_path: Path, persona: Persona) -> str:
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        name=persona.name,
        age=persona.age,
        gender=persona.gender,
        workforce=persona.workforce,
        description=persona.description,
    )
