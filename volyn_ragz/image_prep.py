"""Prepare scan crops for OCR."""

from __future__ import annotations

from PIL import Image


def right_half(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((w // 2, 0, w, h))


def horizontal_strips(
    image: Image.Image,
    *,
    strip_frac: float = 0.11,
    step_frac: float = 0.055,
) -> list[tuple[int, Image.Image]]:
    """Overlapping horizontal windows (fractions of height), top to bottom."""
    w, h = image.size
    strip_h = max(24, int(h * strip_frac))
    step = max(12, int(h * step_frac))
    out: list[tuple[int, Image.Image]] = []
    y = 0
    idx = 0
    while y < h:
        y1 = min(y + strip_h, h)
        crop = image.crop((0, y, w, y1))
        out.append((idx, crop))
        idx += 1
        y += step
        if y1 >= h:
            break
    return out


def is_mostly_blank(image: Image.Image, threshold: float = 0.92) -> bool:
    """True if mean grayscale luminance is very high (empty paper)."""
    gray = image.convert("L")
    pixels = list(gray.getdata())
    if not pixels:
        return True
    mean = sum(pixels) / len(pixels) / 255.0
    return mean >= threshold
