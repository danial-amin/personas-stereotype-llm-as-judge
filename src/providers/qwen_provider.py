from __future__ import annotations

from openai import OpenAI

from src.config import ModelConfig, require_env
from src.image_utils import encode_image
from src.providers.base import LLMProvider

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenProvider(LLMProvider):
    def __init__(self, model: ModelConfig) -> None:
        self.model_id = model.model_id
        self.client = OpenAI(
            api_key=require_env("DASHSCOPE_API_KEY"),
            base_url=model.base_url or DEFAULT_DASHSCOPE_BASE_URL,
        )

    def evaluate(
        self,
        prompt_text: str,
        image_path: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        image_b64, media_type = encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
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
        )
        return response.choices[0].message.content or ""
