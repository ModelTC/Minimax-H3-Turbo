"""Resolution helpers for MiniMax-H3 video generation.

JSON jobs describe ``megapixels`` and ``aspect_ratio``.  This module converts
those user-facing values into dimensions accepted by the video VAE.
"""

from __future__ import annotations

import math
from typing import Final

ROUND_TO: Final[int] = 32
SUPPORTED_ASPECT_RATIOS: Final[tuple[str, ...]] = (
    "21:9",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
    "9:21",
)
ASPECT_VALUES: Final[dict[str, float]] = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
    "9:21": 9 / 21,
}

# The 16:9 ladder is kept explicit to preserve the established H3 canvases.
PIXEL_LADDER_16_9: Final[dict[float, tuple[int, int]]] = {
    0.2: (608, 352),
    0.3: (736, 416),
    0.4: (864, 480),
    0.5: (960, 544),
    0.6: (1056, 608),
    0.7: (1152, 640),
    0.8: (1216, 672),
    0.9: (1280, 736),
    0.98: (1344, 768),
    1.0: (1376, 768),
    1.2: (1504, 832),
    1.5: (1664, 928),
    1.8: (1824, 1024),
    2.0: (1920, 1088),
}


def validate_aspect_ratio(aspect_ratio: str) -> str:
    if not isinstance(aspect_ratio, str) or aspect_ratio not in ASPECT_VALUES:
        allowed = ", ".join(SUPPORTED_ASPECT_RATIOS)
        raise ValueError(f"aspect_ratio must be one of: {allowed}")
    return aspect_ratio


def validate_megapixels(megapixels: float) -> float:
    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)):
        raise TypeError("megapixels must be a number")
    value = float(megapixels)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("megapixels must be finite and positive")
    return value


def _round_dimension(value: float) -> int:
    return max(ROUND_TO, int(round(value / ROUND_TO)) * ROUND_TO)


def _computed_size(megapixels: float, ratio: float) -> tuple[int, int]:
    nearest = min(PIXEL_LADDER_16_9, key=lambda value: abs(value - megapixels))
    reference_width, reference_height = PIXEL_LADDER_16_9[nearest]
    target_pixels = reference_width * reference_height
    width = _round_dimension(math.sqrt(target_pixels * ratio))
    height = _round_dimension(math.sqrt(target_pixels / ratio))
    return width, height


def resolve_output_size(megapixels: float, aspect_ratio: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a megapixel target and supported ratio."""
    megapixels = validate_megapixels(megapixels)
    aspect_ratio = validate_aspect_ratio(aspect_ratio)

    if aspect_ratio == "16:9":
        nearest = min(PIXEL_LADDER_16_9, key=lambda value: abs(value - megapixels))
        if math.isclose(nearest, megapixels, rel_tol=0.0, abs_tol=1e-9):
            return PIXEL_LADDER_16_9[nearest]
    if aspect_ratio == "9:16":
        nearest = min(PIXEL_LADDER_16_9, key=lambda value: abs(value - megapixels))
        if math.isclose(nearest, megapixels, rel_tol=0.0, abs_tol=1e-9):
            width, height = PIXEL_LADDER_16_9[nearest]
            return height, width
    if aspect_ratio == "21:9" and math.isclose(megapixels, 0.5, abs_tol=1e-9):
        return 1088, 480
    if aspect_ratio == "9:21" and math.isclose(megapixels, 0.5, abs_tol=1e-9):
        return 480, 1088

    return _computed_size(megapixels, ASPECT_VALUES[aspect_ratio])
