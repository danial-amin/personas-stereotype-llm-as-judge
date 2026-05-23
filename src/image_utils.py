from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("image/"):
        media_type = "image/jpeg"

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type
