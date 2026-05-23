from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def detect_csv_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in _CSV_ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "csv",
        raw,
        0,
        1,
        f"Could not decode {path} as any of {_CSV_ENCODINGS}",
    )


def open_csv(path: Path):
    """Open a CSV file, auto-detecting common text encodings."""
    encoding = detect_csv_encoding(path)
    if encoding not in {"utf-8", "utf-8-sig"}:
        logger.info("Reading %s with encoding %s", path, encoding)
    return path.open(encoding=encoding, newline="")
