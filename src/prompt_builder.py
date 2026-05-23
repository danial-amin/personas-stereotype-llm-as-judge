from __future__ import annotations

from pathlib import Path

from src.models import Persona


def build_prompt(template_path: Path, persona: Persona) -> str:
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "{persona_id}": persona.persona_id,
        "{name}": persona.name,
        "{gender}": persona.gender,
        "{age_group}": persona.age,
        "{occupation}": persona.workforce,
        "{description}": persona.description,
        "{image}": "Provided as an attached image in this request.",
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt
