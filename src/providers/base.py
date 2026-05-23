from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt_text: str,
        image_path: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send a fresh, stateless evaluation request and return raw text."""
