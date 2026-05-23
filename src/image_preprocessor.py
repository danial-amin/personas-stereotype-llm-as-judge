from __future__ import annotations

import json
import logging
import mimetypes
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "_download_manifest.json"
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = "personas-stereotype-llm-as-judge/1.0"

_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_image_reference(
    persona_id: str,
    image_ref: str,
    images_dir: Path,
    *,
    download: bool = True,
    refresh: bool = False,
) -> str:
    path, _status = resolve_image_reference_with_status(
        persona_id,
        image_ref,
        images_dir,
        download=download,
        refresh=refresh,
    )
    return path


def resolve_image_reference_with_status(
    persona_id: str,
    image_ref: str,
    images_dir: Path,
    *,
    download: bool = True,
    refresh: bool = False,
) -> tuple[str, str]:
    """
    Resolve image_path from CSV to a local file path.

    Returns (local_path, status) where status is:
    - downloaded: freshly fetched from URL
    - cached: reused existing local file for URL
    - local: already a local path
    """
    image_ref = image_ref.strip()
    if not image_ref:
        raise ValueError(f"Persona '{persona_id}' has an empty image_path")

    if is_http_url(image_ref):
        if not download:
            return image_ref, "url"
        local_path, fresh = _download_image(persona_id, image_ref, images_dir, refresh=refresh)
        return local_path, "downloaded" if fresh else "cached"

    local_path = Path(image_ref)
    if download and not local_path.exists():
        raise FileNotFoundError(
            f"Local image not found for persona '{persona_id}': {image_ref}"
        )
    return str(local_path), "local"


def _download_image(
    persona_id: str,
    url: str,
    images_dir: Path,
    *,
    refresh: bool,
) -> tuple[str, bool]:
    """Return (local_path, was_freshly_downloaded)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = images_dir / MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    cached = manifest.get(persona_id)
    if cached and not refresh:
        cached_path = Path(cached["local_path"])
        if cached_path.exists() and cached.get("source_url") == url:
            logger.info("Using cached image for %s: %s", persona_id, cached_path)
            return str(cached_path), False

    logger.info("Downloading image for %s from %s", persona_id, url)
    content, content_type = _fetch_url(url)
    extension = _extension_from_url_or_type(url, content_type)
    local_path = images_dir / f"{persona_id}{extension}"

    local_path.write_bytes(content)
    manifest[persona_id] = {
        "source_url": url,
        "local_path": str(local_path),
    }
    _save_manifest(manifest_path, manifest)

    logger.info("Saved image for %s to %s", persona_id, local_path)
    return str(local_path), True


def _fetch_url(url: str) -> tuple[bytes, str | None]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get_content_type()
            return response.read(), content_type
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc


def _extension_from_url_or_type(url: str, content_type: str | None) -> str:
    if content_type:
        normalized = content_type.split(";")[0].strip().lower()
        if normalized in _EXT_BY_CONTENT_TYPE:
            return _EXT_BY_CONTENT_TYPE[normalized]

    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    guessed, _ = mimetypes.guess_type(path)
    if guessed and guessed in _EXT_BY_CONTENT_TYPE:
        return _EXT_BY_CONTENT_TYPE[guessed]

    return ".jpg"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sanitize_persona_id(persona_id: str) -> str:
    """Ensure downloaded filenames are safe."""
    cleaned = re.sub(r"[^\w\-]+", "_", persona_id.strip())
    if not cleaned:
        raise ValueError("persona_id must contain at least one alphanumeric character")
    return cleaned
