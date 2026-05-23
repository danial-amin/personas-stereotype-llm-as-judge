from __future__ import annotations

from google import genai
from google.genai import types

from src.config import require_env
from src.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))

    def evaluate(
        self,
        prompt_text: str,
        image_path: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime(image_path)),
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""


def _guess_mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"
