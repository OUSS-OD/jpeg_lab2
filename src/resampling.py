"""
Task 1.3 + Task 2.1 — Downsampling, upsampling, interpolation, resizing.

Implements:
* downsample_2x        : decimation (keep every 2nd pixel)
* upsample_2x          : nearest-neighbour expansion (shows pixelisation)
* linear_interp        : 1D linear interpolation between two points
* linear_spline        : evaluate a linear spline at a given x
* bilinear_interp      : bilinear interpolation on 4 corner points
* resize_bilinear      : resize an image to arbitrary (new_h, new_w)
                         using bilinear interpolation (Task 2.1)
"""

from __future__ import annotations

import numpy as np


# ---------- Task 1.3: simple 2x decimation / expansion ----------------------

def downsample_2x(image: np.ndarray) -> np.ndarray:
    """Decimation with factor 2 — keep every other pixel along each axis.

    Requires even H and W.
    """
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError("Height and width must be even")
    return image[::2, ::2].copy()


def upsample_2x(image: np.ndarray) -> np.ndarray:
    """Nearest-neighbour 2x upsampling. Produces visible pixelisation."""
    if image.ndim == 2:
        return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)


# ---------- Task 1.3: linear and bilinear interpolation ---------------------

def linear_interp(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    """Linear interpolation between two points."""
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 * (1.0 - t) + y2 * t


def linear_spline(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Evaluate a piecewise-linear spline at point x.

    xs must be strictly increasing.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    # find the segment via binary search (searchsorted)
    i = int(np.searchsorted(xs, x)) - 1
    i = max(0, min(i, len(xs) - 2))
    return linear_interp(xs[i], xs[i + 1], ys[i], ys[i + 1], x)


def bilinear_interp(
    x1: float, x2: float, y1: float, y2: float,
    z11: float, z12: float, z21: float, z22: float,
    x: float, y: float,
) -> float:
    """Bilinear interpolation on a rectangle.

    z11 = value at (x1, y1), z12 = value at (x1, y2),
    z21 = value at (x2, y1), z22 = value at (x2, y2).
    """
    # first interpolate along x at y1 and y2
    fxy1 = linear_interp(x1, x2, z11, z21, x)
    fxy2 = linear_interp(x1, x2, z12, z22, x)
    # then interpolate the two results along y
    return linear_interp(y1, y2, fxy1, fxy2, y)


# ---------- Task 2.1: bilinear resize to arbitrary size ---------------------

def resize_bilinear(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Resize an image to (new_h, new_w) using bilinear interpolation.

    Works on 2D (grayscale) and 3D (color) uint8 images.
    Sampling coordinates are picked so that the output corner pixels align
    with the input corner pixels.
    """
    new_h, new_w = new_size
    if image.ndim == 2:
        h, w = image.shape
        channels = 1
        src = image[..., None]
    else:
        h, w, channels = image.shape
        src = image

    src = src.astype(np.float32)

    # coordinates in source space
    if new_h == 1:
        ys = np.zeros(new_h, dtype=np.float32)
    else:
        ys = np.linspace(0, h - 1, new_h, dtype=np.float32)
    if new_w == 1:
        xs = np.zeros(new_w, dtype=np.float32)
    else:
        xs = np.linspace(0, w - 1, new_w, dtype=np.float32)

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (xs - x0).astype(np.float32)  # (new_w,)
    wy = (ys - y0).astype(np.float32)  # (new_h,)

    # gather corner pixels using advanced indexing
    # shape: (new_h, new_w, channels)
    Ia = src[np.ix_(y0, x0)]
    Ib = src[np.ix_(y0, x1)]
    Ic = src[np.ix_(y1, x0)]
    Id = src[np.ix_(y1, x1)]

    wx_b = wx[None, :, None]  # broadcast over (new_h, new_w, ch)
    wy_b = wy[:, None, None]

    top = Ia * (1 - wx_b) + Ib * wx_b
    bot = Ic * (1 - wx_b) + Id * wx_b
    out = top * (1 - wy_b) + bot * wy_b

    out = np.clip(out, 0, 255).astype(np.uint8)
    if channels == 1:
        return out[..., 0]
    return out
