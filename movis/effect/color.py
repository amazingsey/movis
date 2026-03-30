from __future__ import annotations

import cv2
import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..util import to_rgb


class FillColor(AttributesMixin):
    """Fill the image with the specified color while preserving the alpha channel.

    Args:
        color:
            Color to fill. It can be specified as a tuple of RGB values or a string of color name.

    Animatable Attributes:
        ``color``
    """

    def __init__(self, color: tuple[int, int, int] | str = (255, 255, 255)):
        c = to_rgb(color)
        self.color = Attribute(c, AttributeType.COLOR, range=(0., 255.))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        rgb_image = np.full_like(
            prev_image[:, :, :3], np.round(self.color(time)).astype(np.uint8))
        alpha_image = prev_image[:, :, 3:]
        return np.concatenate([rgb_image, alpha_image], axis=2)


class Grayscale(AttributesMixin):
    """Convert the image to grayscale using luminance-weighted conversion (BT.601).

    Uses the standard weights (0.299R + 0.587G + 0.114B) — the same conversion
    used by Photoshop's grayscale mode. Preserves contrast between colors of
    different brightness (e.g. yellow on red).

    Args:
        strength:
            Blend factor in the range ``[0, 1]``. 0.0 = original color,
            1.0 = fully grayscale. Defaults to 1.0.

    Animatable Attributes:
        ``strength``
    """

    def __init__(self, strength: float = 1.0):
        self.strength = Attribute(strength, AttributeType.SCALAR, range=(0., 1.))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        rgb = prev_image[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        s = np.float32(self.strength(time))
        if s >= 1.0:
            blended = gray_rgb
        elif s <= 0.0:
            blended = rgb
        else:
            blended = np.clip(
                rgb.astype(np.float32) * (1 - s) + gray_rgb.astype(np.float32) * s,
                0, 255
            ).astype(np.uint8)
        return np.concatenate([blended, prev_image[:, :, 3:]], axis=2)


class HSLShift(AttributesMixin):
    """Shift hue, saturation, and luminance of the image.

    Args:
        hue:
            Hue shift in degrees.
        saturation:
            Saturation shift in the range ``[-1, 1]``.
        luminance:
            Luminance shift in the range ``[-1, 1]``.

    Animatable Attributes:
        ``hue``
        ``saturation``
        ``luminance``
    """

    def __init__(self, hue: float = 0.0, saturation: float = 0.0, luminance: float = 0.0):
        self.hue = Attribute(hue, AttributeType.SCALAR)
        self.saturation = Attribute(saturation, AttributeType.SCALAR, range=(-1., 1.))
        self.luminance = Attribute(luminance, AttributeType.SCALAR, range=(-1., 1.))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        hsv_image = cv2.cvtColor(prev_image[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
        dh = np.float32(self.hue(time))
        ds = np.float32(self.saturation(time))
        dl = np.float32(self.luminance(time))
        h = np.round((h + dh / 2) % 180).astype(np.uint8)
        s = np.clip(np.round(s + ds * 255), 0, 255).astype(np.uint8)
        v = np.clip(np.round(v + dl * 255), 0, 255).astype(np.uint8)
        new_hsv_image = cv2.cvtColor(np.stack([h, s, v], axis=2), cv2.COLOR_HSV2RGB)
        alpha_image = prev_image[:, :, 3:]
        return np.concatenate([new_hsv_image, alpha_image], axis=2)
