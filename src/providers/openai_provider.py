from __future__ import annotations

from openai import OpenAI

from src.config import ModelConfig, require_env
from src.image_utils import encode_image
from src.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: ModelConfig) -> None:
        self.model = model
        self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    def evaluate(
        self,
        prompt_text: str,
        image_path: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        image_b64, media_type = encode_image(image_path)

        kwargs: dict = {
            "model": self.model.model_id,
            "max_completion_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        }

        if self.model.reasoning_effort:
            kwargs["reasoning_effort"] = self.model.reasoning_effort
        if self.model.use_temperature:
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
