"""TrOCR inference for handwritten Cyrillic lines."""

from __future__ import annotations

import sys
import types

import torch
from PIL import Image

try:
    import torchvision.transforms as tv_transforms

    if not hasattr(tv_transforms.InterpolationMode, "NEAREST_EXACT"):
        tv_transforms.InterpolationMode.NEAREST_EXACT = tv_transforms.InterpolationMode.NEAREST

    # transformers>=4.57 expects torchvision.transforms.v2, absent in older torchvision builds.
    if "torchvision.transforms.v2" not in sys.modules:
        tv_v2 = types.ModuleType("torchvision.transforms.v2")
        tv_v2.functional = tv_transforms.functional
        sys.modules["torchvision.transforms.v2"] = tv_v2
except Exception:  # noqa: BLE001
    pass

try:
    from transformers import VisionEncoderDecoderModel
except ImportError:  # transformers>=4.57 may stop re-exporting at package root
    from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
        VisionEncoderDecoderModel,
    )

try:
    from transformers import TrOCRProcessor
except ImportError:  # transformers>=4.57 may stop re-exporting at package root
    from transformers.models.trocr.processing_trocr import TrOCRProcessor

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
