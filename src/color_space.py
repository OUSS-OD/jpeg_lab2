"""
Task 1.2 — Color space conversions.

RGB <-> YCbCr using the ITU-T.871 / JFIF formulas (same as JPEG).

    Y  =       0.299*R + 0.587*G + 0.114*B
    Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B

Inverse:
    R = Y +                    1.402*(Cr-128)
    G = Y - 0.344136*(Cb-128) - 0.714136*(Cr-128)
    B = Y + 1.772*(Cb-128)

Input / output: numpy uint8 arrays of shape (H, W, 3).
"""

from __future__ import annotations

import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H,W,3) uint8 RGB image to a uint8 YCbCr image."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {rgb.dtype} {rgb.shape}")

    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    out = np.stack([y, cb, cr], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert an (H,W,3) uint8 YCbCr image back to a uint8 RGB image."""
    if ycbcr.dtype != np.uint8 or ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {ycbcr.dtype} {ycbcr.shape}")

    arr = ycbcr.astype(np.float32)
    y, cb, cr = arr[..., 0], arr[..., 1], arr[..., 2]
    cb_s = cb - 128.0
    cr_s = cr - 128.0

    r = y + 1.402 * cr_s
    g = y - 0.344136 * cb_s - 0.714136 * cr_s
    b = y + 1.772 * cb_s

    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)
