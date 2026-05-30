"""
Task 1.1 — Custom RAW image format.

File layout (little-endian):
    magic        : 4  bytes  = b'MYRW'
    version      : 1  byte   = 1
    mode         : 1  byte   (0 = B&W 1-bit, 1 = Grayscale 8-bit, 2 = RGB, 3 = YCbCr)
    width        : 4  bytes  (uint32)
    height       : 4  bytes  (uint32)
    reserved     : 2  bytes  (padding / future flags)
    pixel_data   : width*height*channels bytes
                   (for mode=0 (B&W) we still store 1 byte per pixel: 0 or 255 — simplest)

This keeps the format trivial to parse while covering all required metadata:
type (B&W / grayscale / color) and size. The color-space tag is also stored
(RGB vs YCbCr), as required in Task 1.2.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"MYRW"
VERSION = 1

MODE_BW = 0
MODE_GRAY = 1
MODE_RGB = 2
MODE_YCBCR = 3

MODE_NAMES = {
    MODE_BW: "B&W",
    MODE_GRAY: "Grayscale",
    MODE_RGB: "RGB",
    MODE_YCBCR: "YCbCr",
}

CHANNELS = {MODE_BW: 1, MODE_GRAY: 1, MODE_RGB: 3, MODE_YCBCR: 3}


def save_raw(path: str | Path, image: np.ndarray, mode: int) -> None:
    """Serialize an image array to the custom RAW format.

    Parameters
    ----------
    path  : output file path
    image : numpy array, shape (H, W) for 1-channel or (H, W, 3) for 3-channel,
            dtype uint8
    mode  : one of MODE_BW / MODE_GRAY / MODE_RGB / MODE_YCBCR
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    expected_channels = CHANNELS[mode]
    if expected_channels == 1:
        if image.ndim != 2:
            raise ValueError(f"Mode {MODE_NAMES[mode]} expects 2D array, got shape {image.shape}")
        h, w = image.shape
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Mode {MODE_NAMES[mode]} expects (H, W, 3), got shape {image.shape}")
        h, w, _ = image.shape

    header = MAGIC + struct.pack("<BBII2x", VERSION, mode, w, h)

    with open(path, "wb") as f:
        f.write(header)
        f.write(image.tobytes())


def load_raw(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an image from the custom RAW format.

    Returns
    -------
    (image, mode) : numpy array + mode constant
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != MAGIC:
        raise ValueError("Not a MYRW file (magic mismatch)")

    version, mode, w, h = struct.unpack("<BBII", data[4:14])
    if version != VERSION:
        raise ValueError(f"Unsupported version {version}")

    # 2 bytes reserved, so pixel data starts at offset 16
    pixel_data = data[16:]
    ch = CHANNELS[mode]
    expected = w * h * ch
    if len(pixel_data) != expected:
        raise ValueError(f"Expected {expected} pixel bytes, got {len(pixel_data)}")

    arr = np.frombuffer(pixel_data, dtype=np.uint8)
    if ch == 1:
        arr = arr.reshape(h, w)
    else:
        arr = arr.reshape(h, w, ch)

    return arr.copy(), mode


def raw_file_size(image: np.ndarray, mode: int) -> int:
    """Return the total size (bytes) of the raw file for the given image."""
    ch = CHANNELS[mode]
    if ch == 1:
        h, w = image.shape
    else:
        h, w, _ = image.shape
    return 16 + h * w * ch  # 16-byte header
