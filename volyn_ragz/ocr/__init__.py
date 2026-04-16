from __future__ import annotations

from typing import Protocol


class OCREngine(Protocol):
    def recognize_pil(self, image: object) -> str:
        """Return OCR text for a PIL image."""


def get_ocr_engine(
    backend: str,
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> OCREngine:
    if backend == "trocr":
        from volyn_ragz.ocr.trocr_engine import DEFAULT_MODEL, TrOCREngine

        return TrOCREngine(model_name=model_name or DEFAULT_MODEL, device=device)
    if backend == "gemini":
        from volyn_ragz.ocr.gemini_engine import DEFAULT_MODEL, GeminiEngine

        return GeminiEngine(model_name=model_name or DEFAULT_MODEL)
    raise ValueError(f"Невідомий OCR backend: {backend}")


__all__ = ["OCREngine", "get_ocr_engine"]
