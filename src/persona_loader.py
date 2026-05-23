from __future__ import annotations

import csv
from pathlib import Path

from src.models import Persona


REQUIRED_COLUMNS = {
    "persona_id",
    "name",
    "age",
    "gender",
    "workforce",
    "description",
    "image_path",
}


def load_personas(csv_path: Path, *, validate_images: bool = True) -> list[Persona]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Personas CSV not found: {csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Personas CSV has no header row: {csv_path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Personas CSV missing required columns {sorted(missing)}: {csv_path}"
            )

        personas: list[Persona] = []
        for row in reader:
            persona_id = (row.get("persona_id") or "").strip()
            if not persona_id:
                continue

            image_path = (row.get("image_path") or "").strip()
            if validate_images and image_path and not Path(image_path).exists():
                raise FileNotFoundError(
                    f"Image not found for persona '{persona_id}': {image_path}"
                )

            personas.append(
                Persona(
                    persona_id=persona_id,
                    name=(row.get("name") or "").strip(),
                    age=(row.get("age") or "").strip(),
                    gender=(row.get("gender") or "").strip(),
                    workforce=(row.get("workforce") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    image_path=image_path,
                )
            )

    if not personas:
        raise ValueError(f"No personas found in CSV: {csv_path}")

    return personas
