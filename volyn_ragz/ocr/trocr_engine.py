"""TrOCR inference for handwritten Cyrillic lines."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


class TrOCREngine:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def recognize_pil(self, image: Image.Image) -> str:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        out = self._model.generate(pixel_values)
        return self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
