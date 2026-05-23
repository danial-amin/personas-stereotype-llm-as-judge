from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Persona:
    persona_id: str
    name: str
    age: str
    gender: str
    workforce: str
    description: str
    image_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class EvaluationResult:
    persona_id: str
    model_key: str
    model_display_name: str
    model_id: str
    run_index: int
    timestamp: str
    prompt_text: str
    response_raw: str
    parsed: dict[str, Any] | None
    latency_ms: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "model_key": self.model_key,
            "model_display_name": self.model_display_name,
            "model_id": self.model_id,
            "run_index": self.run_index,
            "timestamp": self.timestamp,
            "prompt_text": self.prompt_text,
            "response_raw": self.response_raw,
            "parsed": self.parsed,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }
