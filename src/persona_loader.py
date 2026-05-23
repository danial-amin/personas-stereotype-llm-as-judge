from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.csv_utils import open_csv
from src.image_preprocessor import is_http_url, resolve_image_reference
from src.models import Persona

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "persona_id",
    "name",
    "age",
    "gender",
    "workforce",
    "description",
    "image_path",
}


def load_personas(
    csv_path: Path,
    *,
    images_dir: Path | None = None,
    validate_images: bool = True,
    refresh_images: bool = False,
) -> list[Persona]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Personas CSV not found: {csv_path}")

    if images_dir is None:
        images_dir = csv_path.parent / "images"

    with open_csv(csv_path) as f:
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

            image_source = (row.get("image_path") or "").strip()
            if validate_images:
                local_image_path = resolve_image_reference(
                    persona_id,
                    image_source,
                    images_dir,
                    download=True,
                    refresh=refresh_images,
                )
            elif is_http_url(image_source):
                local_image_path = image_source
                logger.debug(
                    "Dry run: would download image for %s from %s",
                    persona_id,
                    image_source,
                )
            else:
                local_image_path = image_source

            personas.append(
                Persona(
                    persona_id=persona_id,
                    name=(row.get("name") or "").strip(),
                    age=(row.get("age") or "").strip(),
                    gender=(row.get("gender") or "").strip(),
                    workforce=(row.get("workforce") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    image_path=local_image_path,
                    image_source=image_source,
                )
            )

    if not personas:
        raise ValueError(f"No personas found in CSV: {csv_path}")

    return personas
