"""
Task 1.1 — Generate the test image set.

Produces five images in the ./images directory:
    1. lena_synth.png        — synthetic "Lena-like" 512x512 color image
    2. color_pattern.png     — 512x512 color pattern
    3. color_gray.png        — grayscale from color_pattern
    4. color_bw_round.png    — black-and-white via rounding c' = round(c/255)*255
    5. color_bw_dither.png   — black-and-white via PIL dithering
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_lena(size: int = 512) -> np.ndarray:
    """A synthetic stand-in for Lena: smooth gradients + a couple of features.

    Not the real Lena (copyrighted); just a colored image with smooth areas
    and sharp edges so compression artefacts are visible.
    """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cy, cx = size / 2, size / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)

    # base skin-tone-ish gradient
    R = 180 + 40 * np.sin(xx / 80) * np.cos(yy / 120)
    G = 140 + 30 * np.sin(yy / 70 + 1.0)
    B = 120 + 30 * np.cos(xx / 90 + 0.5)

    # a disk "face"
    face = r < size * 0.28
    R = np.where(face, 220 + 20 * np.sin(angle * 6), R)
    G = np.where(face, 170 + 15 * np.cos(angle * 4), G)
    B = np.where(face, 150, B)

    # "eyes"
    for ex in (-size * 0.08, size * 0.08):
        d = np.sqrt((xx - (cx + ex)) ** 2 + (yy - (cy - size * 0.03)) ** 2)
        eye = d < size * 0.025
        R = np.where(eye, 30, R)
        G = np.where(eye, 30, G)
        B = np.where(eye, 40, B)

    # a ring
    ring = (r > size * 0.33) & (r < size * 0.34)
    R = np.where(ring, 20, R)
    G = np.where(ring, 20, G)
    B = np.where(ring, 20, B)

    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_color_pattern(size: int = 512) -> np.ndarray:
    """A colorful test pattern with smooth and high-frequency regions."""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    R = (127 + 127 * np.sin(xx / 20)).astype(np.float32)
    G = (127 + 127 * np.sin(yy / 30 + 1)).astype(np.float32)
    B = (127 + 127 * np.sin((xx + yy) / 50 + 2)).astype(np.float32)

    # a few colored rectangles to make it non-periodic
    R[100:180, 100:300] = 255
    G[100:180, 100:300] = 50
    B[100:180, 100:300] = 50

    R[300:420, 250:460] = 30
    G[300:420, 250:460] = 200
    B[300:420, 250:460] = 90

    # fine stripes (high frequency)
    R[40:90, 350:500] = ((np.arange(500 - 350) // 2) % 2 * 255).astype(np.float32)
    G[40:90, 350:500] = ((np.arange(500 - 350) // 2) % 2 * 255).astype(np.float32)
    B[40:90, 350:500] = 0

    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma — same formula used as Y in YCbCr."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def gray_to_bw_round(gray: np.ndarray) -> np.ndarray:
    """c' = round(c/255)*255."""
    return (np.round(gray.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)


def gray_to_bw_dither(gray: np.ndarray) -> np.ndarray:
    """Use PIL's Floyd-Steinberg dithering, then back to numpy as 0/255."""
    img = Image.fromarray(gray, mode="L").convert(mode="1")  # default dithering
    return (np.asarray(img, dtype=np.uint8) * 255)


def generate_all(out_dir: str | Path = "images") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lena = make_synthetic_lena(512)
    Image.fromarray(lena).save(out / "lena_synth.png")

    color = make_color_pattern(512)
    Image.fromarray(color).save(out / "color_pattern.png")

    gray = rgb_to_gray(color)
    Image.fromarray(gray, mode="L").save(out / "color_gray.png")

    bw_round = gray_to_bw_round(gray)
    Image.fromarray(bw_round, mode="L").save(out / "color_bw_round.png")

    bw_dither = gray_to_bw_dither(gray)
    Image.fromarray(bw_dither, mode="L").save(out / "color_bw_dither.png")

    return {
        "lena": lena,
        "color": color,
        "gray": gray,
        "bw_round": bw_round,
        "bw_dither": bw_dither,
    }


if __name__ == "__main__":
    imgs = generate_all()
    for name, arr in imgs.items():
        print(f"{name:12s}  shape={arr.shape}  dtype={arr.dtype}")
