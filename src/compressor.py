"""
Task 2.6 + 2.7 — Full JPEG-inspired compressor / decompressor.

Pipeline:
    RGB -> YCbCr -> (split into 8x8 blocks per channel, pad if needed)
         -> level shift (subtract 128)
         -> DCT
         -> quantise using scaled Q tables
         -> zigzag per block
         -> DC differential coding + AC RLE + Huffman encoding
         -> pack into a file with metadata

File layout (.myjpg — my own container, not interchange with real JPEG):
    magic         : 4 bytes  b'MJPG'
    version       : 1 byte   = 1
    color_mode    : 1 byte   (0 = grayscale, 1 = YCbCr color)
    quality       : 1 byte   (1..99)
    reserved      : 1 byte   (padding)
    width         : 4 bytes  (uint32, LE)
    height        : 4 bytes  (uint32, LE)
    payload_len   : 4 bytes  (uint32, LE — number of bytes in the Huffman
                              bitstream that follows)
    payload       : payload_len bytes (Huffman-coded bitstream)

Quantisation tables and Huffman tables are implicit — the decoder uses the
same fixed standard tables that the encoder used, scaled from `quality`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np

from .color_space import rgb_to_ycbcr, ycbcr_to_rgb
from .dct import (
    BLOCK_SIZE, STD_LUMA_Q, STD_CHROMA_Q,
    dct2_blocks, idct2_blocks, quality_scale_q, quantise, dequantise,
    split_blocks, merge_blocks,
)
from .entropy import (
    differential_encode_dc, differential_decode_dc,
    run_length_encode_ac, run_length_decode_ac,
)
from .huffman import (
    BitReader, BitWriter,
    DC_LUMA_TABLE, DC_CHROMA_TABLE, AC_LUMA_TABLE, AC_CHROMA_TABLE,
    DC_LUMA_DECODE, DC_CHROMA_DECODE, AC_LUMA_DECODE, AC_CHROMA_DECODE,
    encode_block, decode_block,
)
from .zigzag import zigzag, inverse_zigzag

MJPG_MAGIC = b"MJPG"
MJPG_VERSION = 1
MODE_GRAY = 0
MODE_COLOR = 1


# ---------- Per-channel encode / decode -------------------------------------

def _encode_channel(
    channel: np.ndarray,
    q_table: np.ndarray,
    dc_table, ac_table,
    bw: BitWriter,
) -> Tuple[int, int]:
    """Encode a single channel (2D uint8) into `bw`. Returns (h, w) of channel."""
    h, w = channel.shape
    # Level shift to roughly [-128, 127]
    shifted = channel.astype(np.float32) - 128.0
    blocks, _ = split_blocks(shifted)
    coeffs = dct2_blocks(blocks)
    q = quantise(coeffs, q_table)  # (Nr, Nc, 8, 8) int32

    # Zigzag each block, then DC diff + AC RLE + Huffman
    n_rows, n_cols = q.shape[:2]
    dc_values = []
    ac_per_block = []
    for by in range(n_rows):
        for bx in range(n_cols):
            zz = zigzag(q[by, bx])
            dc_values.append(int(zz[0]))
            ac_per_block.append(run_length_encode_ac(zz[1:].tolist()))

    dc_diffs = differential_encode_dc(dc_values)
    for diff_dc, ac in zip(dc_diffs, ac_per_block):
        encode_block(diff_dc, ac, bw, dc_table, ac_table)
    return h, w


def _decode_channel(
    h: int, w: int,
    q_table: np.ndarray,
    dc_decode, ac_decode,
    br: BitReader,
) -> np.ndarray:
    """Decode a channel from `br` and return uint8 image of shape (h, w)."""
    # How many blocks?
    padded_h = h + ((-h) % BLOCK_SIZE)
    padded_w = w + ((-w) % BLOCK_SIZE)
    n_rows, n_cols = padded_h // BLOCK_SIZE, padded_w // BLOCK_SIZE
    total = n_rows * n_cols

    dc_diffs = []
    ac_per_block = []
    for _ in range(total):
        diff_dc, ac = decode_block(br, dc_decode, ac_decode)
        dc_diffs.append(diff_dc)
        ac_per_block.append(ac)
    dc_values = differential_decode_dc(dc_diffs)

    blocks_q = np.zeros((n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE), dtype=np.int32)
    idx = 0
    for by in range(n_rows):
        for bx in range(n_cols):
            ac_vals = run_length_decode_ac(ac_per_block[idx], length=63)
            zz = np.concatenate([[dc_values[idx]], np.asarray(ac_vals, dtype=np.int32)])
            blocks_q[by, bx] = inverse_zigzag(zz, (BLOCK_SIZE, BLOCK_SIZE))
            idx += 1

    coeffs = dequantise(blocks_q, q_table)
    recovered = idct2_blocks(coeffs)
    merged = merge_blocks(recovered, (h, w))
    merged += 128.0  # undo level shift
    return np.clip(merged, 0, 255).astype(np.uint8)


# ---------- Public API: file compress / decompress --------------------------

def compress(image: np.ndarray, quality: int = 50) -> bytes:
    """Compress a uint8 image. Returns the full file bytes (header + payload).

    `image` : (H, W) grayscale or (H, W, 3) RGB uint8.
    """
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")

    quality = max(1, min(99, int(quality)))

    if image.ndim == 2:
        # grayscale
        color_mode = MODE_GRAY
        h, w = image.shape
        y_q = quality_scale_q(STD_LUMA_Q, quality)
        bw = BitWriter()
        _encode_channel(image, y_q, DC_LUMA_TABLE, AC_LUMA_TABLE, bw)
        payload = bw.to_bytes()
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected (H,W,3) color image")
        color_mode = MODE_COLOR
        h, w = image.shape[:2]
        ycbcr = rgb_to_ycbcr(image)
        y_q = quality_scale_q(STD_LUMA_Q, quality)
        c_q = quality_scale_q(STD_CHROMA_Q, quality)
        bw = BitWriter()
        _encode_channel(ycbcr[..., 0], y_q, DC_LUMA_TABLE,   AC_LUMA_TABLE,   bw)
        _encode_channel(ycbcr[..., 1], c_q, DC_CHROMA_TABLE, AC_CHROMA_TABLE, bw)
        _encode_channel(ycbcr[..., 2], c_q, DC_CHROMA_TABLE, AC_CHROMA_TABLE, bw)
        payload = bw.to_bytes()

    header = (
        MJPG_MAGIC
        + struct.pack("<BBBBII I",
                      MJPG_VERSION,
                      color_mode,
                      quality,
                      0,           # reserved
                      w, h,
                      len(payload))
    )
    return header + payload


def decompress(data: bytes) -> np.ndarray:
    """Inverse of compress()."""
    if data[:4] != MJPG_MAGIC:
        raise ValueError("not a MJPG stream (bad magic)")
    version, color_mode, quality, _res = struct.unpack("<BBBB", data[4:8])
    if version != MJPG_VERSION:
        raise ValueError(f"unsupported version {version}")
    w, h, payload_len = struct.unpack("<III", data[8:20])
    payload = data[20:20 + payload_len]

    br = BitReader(payload)
    if color_mode == MODE_GRAY:
        y_q = quality_scale_q(STD_LUMA_Q, quality)
        return _decode_channel(h, w, y_q, DC_LUMA_DECODE, AC_LUMA_DECODE, br)
    elif color_mode == MODE_COLOR:
        y_q = quality_scale_q(STD_LUMA_Q, quality)
        c_q = quality_scale_q(STD_CHROMA_Q, quality)
        y  = _decode_channel(h, w, y_q, DC_LUMA_DECODE,   AC_LUMA_DECODE,   br)
        cb = _decode_channel(h, w, c_q, DC_CHROMA_DECODE, AC_CHROMA_DECODE, br)
        cr = _decode_channel(h, w, c_q, DC_CHROMA_DECODE, AC_CHROMA_DECODE, br)
        ycbcr = np.stack([y, cb, cr], axis=-1)
        return ycbcr_to_rgb(ycbcr)
    else:
        raise ValueError(f"unknown color mode {color_mode}")


def compress_to_file(image: np.ndarray, path: str | Path, quality: int = 50) -> int:
    """Compress and write to disk. Returns number of bytes written."""
    data = compress(image, quality=quality)
    Path(path).write_bytes(data)
    return len(data)


def decompress_from_file(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    return decompress(data)
