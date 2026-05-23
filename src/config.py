from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    key: str
    provider: str
    model_id: str
    display_name: str


@dataclass(frozen=True)
class AppConfig:
    runs_per_model: int
    temperature: float
    max_tokens: int
    request_delay_seconds: float
    max_retries: int
    retry_wait_seconds: int
    personas_csv: Path
    images_dir: Path
    prompt_template: Path
    results_dir: Path
    models: list[ModelConfig]


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv()

    root = Path(__file__).resolve().parent.parent
    config_path = config_path or root / "config.yaml"

    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths = raw["paths"]
    models = [
        ModelConfig(
            key=key,
            provider=value["provider"],
            model_id=value["model_id"],
            display_name=value["display_name"],
        )
        for key, value in raw["models"].items()
    ]

    return AppConfig(
        runs_per_model=int(raw["runs_per_model"]),
        temperature=float(raw["temperature"]),
        max_tokens=int(raw["max_tokens"]),
        request_delay_seconds=float(raw["request_delay_seconds"]),
        max_retries=int(raw["max_retries"]),
        retry_wait_seconds=int(raw["retry_wait_seconds"]),
        personas_csv=root / paths["personas_csv"],
        images_dir=root / paths["images_dir"],
        prompt_template=root / paths["prompt_template"],
        results_dir=root / paths["results_dir"],
        models=models,
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
