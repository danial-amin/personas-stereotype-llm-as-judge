from __future__ import annotations

import anthropic

from src.config import require_env
from src.image_utils import encode_image
from src.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    def evaluate(
        self,
        prompt_text: str,
        image_path: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        image_b64, media_type = encode_image(image_path)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks)
