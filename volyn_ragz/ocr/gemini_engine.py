"""Gemini inference for OCR-like text extraction from image strips."""

from __future__ import annotations

import os

from google import genai
from PIL import Image

DEFAULT_MODEL = "gemini-2.5-flash"

PROMPT = (
    "You are doing OCR on a historical registry card line image. "
    "Return only the handwritten text visible in this crop. "
    "Do not explain anything. "
    "If the crop is blank or unreadable, return an empty string."
)


class GeminiEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini потребує GOOGLE_API_KEY у змінних середовища.")
        self._client = genai.Client(api_key=self.api_key)

    def recognize_pil(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[PROMPT, image],
        )
        return (response.text or "").strip()
