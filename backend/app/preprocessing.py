"""Image preprocessing without TensorFlow.

Matches the training pipeline of the notebook:
  PIL decode -> grayscale (PIL "L", ITU-R 601-2) -> 3 identical channels -> 224x224 antialiased bilinear -> /255.
The resize re-implements tf.image.resize(method="bilinear", antialias=True). It differs from TensorFlow by at most
1 gray level on ~0.01% of pixels and changed 0 of 158 test predictions for all three models.
"""
import io
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 224


@lru_cache(maxsize=32)
def _resize_weights(n_in: int, n_out: int) -> np.ndarray:
    """Row-normalised triangle-kernel weights, widened by 1/scale when downscaling (TF's antialias behaviour)."""
    inv = n_in / n_out
    support = max(inv, 1.0)
    w = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        centre = (i + 0.5) * inv
        j = np.arange(max(int(np.floor(centre - support)), 0), min(int(np.ceil(centre + support)), n_in))
        k = np.maximum(0.0, 1.0 - np.abs((j + 0.5 - centre) / support))
        if k.sum() > 0:
            w[i, j] = k / k.sum()
    return w.astype(np.float32)


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("The file is not a readable image.") from e
    return img


def to_model_input(img: Image.Image) -> np.ndarray:
    """PIL image -> uint8 array (224, 224) of the grayscale content the model sees."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    out = _resize_weights(h, IMG_SIZE) @ gray @ _resize_weights(w, IMG_SIZE).T
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def to_tensor(gray224: np.ndarray) -> np.ndarray:
    """uint8 (224, 224) -> float32 (1, 224, 224, 3) in [0, 1]."""
    x = gray224.astype(np.float32) / 255.0
    return np.repeat(x[None, :, :, None], 3, axis=-1)


def input_warnings(img: Image.Image, checks: dict, fg_threshold: int = 15) -> list[str]:
    """Same checks as the notebook: they only add warnings and never change the prediction."""
    warns = []
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    ch_diff = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    if ch_diff > checks["max_channel_diff"]:
        warns.append("Colour image — converted to grayscale. Please confirm it is a brain MRI.")
    fg = float((gray > fg_threshold).mean())
    lo, hi = checks["foreground_ratio_range"]
    if not lo <= fg <= hi:
        warns.append(f"Unusual framing: the head fills {fg:.0%} of the image (training range {lo:.0%}–{hi:.0%}).")
    lo, hi = checks["mean_intensity_range"]
    if not lo <= gray.mean() <= hi:
        warns.append(f"Unusual brightness ({gray.mean():.0f}; training range {lo:.0f}–{hi:.0f}).")
    b = max(1, int(0.05 * min(gray.shape)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    if (border <= fg_threshold).mean() < checks["border_dark_min"]:
        warns.append("The border is not dark — MRI slices normally sit on a black background.")
    return warns
