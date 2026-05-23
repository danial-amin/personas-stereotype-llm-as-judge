from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from src.csv_utils import open_csv
from src.image_preprocessor import is_http_url, resolve_image_reference_with_status

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


@dataclass
class ImageDownloadResult:
    persona_id: str
    source: str
    local_path: str | None
    status: str  # downloaded | cached | local | missing | error
    error: str | None = None


def download_images_from_csv(
    csv_path: Path,
    *,
    images_dir: Path | None = None,
    refresh: bool = False,
    persona_ids: set[str] | None = None,
) -> list[ImageDownloadResult]:
    """Download HTTP(S) images from CSV. Does not call any LLMs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Personas CSV not found: {csv_path}")

    if images_dir is None:
        images_dir = csv_path.parent / "images"

    results: list[ImageDownloadResult] = []

    with open_csv(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Personas CSV has no header row: {csv_path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Personas CSV missing required columns {sorted(missing)}: {csv_path}"
            )

        for row in reader:
            persona_id = (row.get("persona_id") or "").strip()
            if not persona_id:
                continue
            if persona_ids and persona_id not in persona_ids:
                continue

            source = (row.get("image_path") or "").strip()
            if not source:
                results.append(
                    ImageDownloadResult(
                        persona_id=persona_id,
                        source="",
                        local_path=None,
                        status="error",
                        error="empty image_path",
                    )
                )
                continue

            try:
                if is_http_url(source):
                    local_path, status = resolve_image_reference_with_status(
                        persona_id,
                        source,
                        images_dir,
                        download=True,
                        refresh=refresh,
                    )
                    results.append(
                        ImageDownloadResult(
                            persona_id=persona_id,
                            source=source,
                            local_path=local_path,
                            status=status,
                        )
                    )
                else:
                    local = Path(source)
                    if local.exists():
                        results.append(
                            ImageDownloadResult(
                                persona_id=persona_id,
                                source=source,
                                local_path=str(local),
                                status="local",
                            )
                        )
                    else:
                        results.append(
                            ImageDownloadResult(
                                persona_id=persona_id,
                                source=source,
                                local_path=None,
                                status="missing",
                                error=f"local file not found: {source}",
                            )
                        )
            except Exception as exc:
                results.append(
                    ImageDownloadResult(
                        persona_id=persona_id,
                        source=source,
                        local_path=None,
                        status="error",
                        error=str(exc),
                    )
                )

    if not results:
        raise ValueError(f"No personas found in CSV: {csv_path}")

    return results
