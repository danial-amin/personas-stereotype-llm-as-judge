from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone

from tenacity import retry, stop_after_attempt, wait_fixed

from src.config import AppConfig, ModelConfig
from src.models import EvaluationResult, Persona
from src.prompt_builder import build_prompt
from src.providers.base import LLMProvider
from src.providers.factory import create_provider
from src.response_parser import parse_json_response

logger = logging.getLogger(__name__)


class PersonaEvaluator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._providers: dict[str, LLMProvider] = {}

    def _get_provider(self, model: ModelConfig) -> LLMProvider:
        if model.key not in self._providers:
            self._providers[model.key] = create_provider(model)
        return self._providers[model.key]

    def evaluate_persona(
        self,
        persona: Persona,
        existing: list[EvaluationResult] | None = None,
        on_result: Callable[[list[EvaluationResult]], None] | None = None,
    ) -> list[EvaluationResult]:
        prompt_text = build_prompt(self.config.prompt_template, persona)
        completed = {(e.model_key, e.run_index) for e in (existing or [])}
        results = list(existing or [])

        for model in self.config.models:
            provider = self._get_provider(model)
            for run_index in range(1, self.config.runs_per_model + 1):
                key = (model.key, run_index)
                if key in completed:
                    logger.info(
                        "Skipping %s run %s for %s (already completed)",
                        model.key,
                        run_index,
                        persona.persona_id,
                    )
                    continue

                logger.info(
                    "Evaluating %s with %s (run %s/%s)",
                    persona.persona_id,
                    model.display_name,
                    run_index,
                    self.config.runs_per_model,
                )

                result = self._run_single(
                    persona=persona,
                    model=model,
                    provider=provider,
                    prompt_text=prompt_text,
                    run_index=run_index,
                )
                results.append(result)
                if on_result is not None:
                    on_result(results)

                if self.config.request_delay_seconds > 0:
                    time.sleep(self.config.request_delay_seconds)

        return results

    def evaluate_once(
        self,
        persona: Persona,
        model_key: str,
        run_index: int,
    ) -> EvaluationResult:
        model = next((m for m in self.config.models if m.key == model_key), None)
        if model is None:
            raise ValueError(f"Unknown model key: {model_key}")

        prompt_text = build_prompt(self.config.prompt_template, persona)
        provider = self._get_provider(model)
        result = self._run_single(
            persona=persona,
            model=model,
            provider=provider,
            prompt_text=prompt_text,
            run_index=run_index,
        )

        if self.config.request_delay_seconds > 0:
            time.sleep(self.config.request_delay_seconds)

        return result

    def _run_single(
        self,
        persona: Persona,
        model: ModelConfig,
        provider: LLMProvider,
        prompt_text: str,
        run_index: int,
    ) -> EvaluationResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()

        try:
            response_raw = self._call_with_retry(
                provider=provider,
                prompt_text=prompt_text,
                image_path=persona.image_path,
            )
            parsed = parse_json_response(response_raw)
            error = None if parsed else "Failed to parse JSON response"
            latency_ms = int((time.perf_counter() - start) * 1000)

            return EvaluationResult(
                persona_id=persona.persona_id,
                model_key=model.key,
                model_display_name=model.display_name,
                model_id=model.model_id,
                run_index=run_index,
                timestamp=timestamp,
                prompt_text=prompt_text,
                response_raw=response_raw,
                parsed=parsed,
                latency_ms=latency_ms,
                error=error,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "Evaluation failed for %s / %s run %s",
                persona.persona_id,
                model.key,
                run_index,
            )
            return EvaluationResult(
                persona_id=persona.persona_id,
                model_key=model.key,
                model_display_name=model.display_name,
                model_id=model.model_id,
                run_index=run_index,
                timestamp=timestamp,
                prompt_text=prompt_text,
                response_raw="",
                parsed=None,
                latency_ms=latency_ms,
                error=str(exc),
            )

    def _call_with_retry(
        self,
        provider: LLMProvider,
        prompt_text: str,
        image_path: str,
    ) -> str:
        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_fixed(self.config.retry_wait_seconds),
            reraise=True,
        )
        def _call() -> str:
            return provider.evaluate(
                prompt_text,
                image_path,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        return _call()
