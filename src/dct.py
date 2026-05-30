"""
Task 1.4 — Discrete Cosine Transform, blocking, quantisation.

Implements:
* split_blocks / merge_blocks  : split (H,W) image into 8x8 blocks (pads as needed)
* dct2_naive / idct2_naive     : direct double-sum formula, NxM generic
* dct2_matrix / idct2_matrix   : C^T * s * C matrix formulation
* quantise / dequantise        : JPEG-style quantisation
* STD_LUMA_Q / STD_CHROMA_Q    : standard ITU-T.81 Annex K quantisation tables
* quality_scale_q              : Task 2.6 — rebuild Q table for a given quality
"""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 8


# ---------- Blocking ---------------------------------------------------------

def split_blocks(channel: np.ndarray, block: int = BLOCK_SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad an (H,W) image so dims are multiples of `block`, then split.

    Padding is done by filling incomplete right/bottom blocks with the mean
    of the existing pixels in that block (per the "Additional Materials"
    section of the assignment).

    Returns
    -------
    blocks : array of shape (n_rows, n_cols, block, block), dtype float32
    original_shape : (H, W) of the input, so we can crop after merging
    """
    h, w = channel.shape
    pad_h = (-h) % block
    pad_w = (-w) % block

    if pad_h == 0 and pad_w == 0:
        padded = channel.astype(np.float32)
    else:
        padded = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)
        padded[:h, :w] = channel
        # Fill right strip
        if pad_w:
            for by in range(0, h, block):
                y1 = min(by + block, h)
                strip = channel[by:y1, :]
                # for each incomplete block along the right edge
                bx_start = (w // block) * block
                if bx_start < w:
                    existing = channel[by:y1, bx_start:w].astype(np.float32)
                    mean = float(existing.mean()) if existing.size else 0.0
                    padded[by:y1, w:w + pad_w] = mean
                else:
                    padded[by:y1, w:w + pad_w] = float(strip.mean()) if strip.size else 0.0
        # Fill bottom strip
        if pad_h:
            for bx in range(0, w + pad_w, block):
                x1 = min(bx + block, w + pad_w)
                existing = padded[:h, bx:x1]
                mean = float(existing.mean()) if existing.size else 0.0
                padded[h:h + pad_h, bx:x1] = mean

    H, W = padded.shape
    n_rows, n_cols = H // block, W // block
    blocks = padded.reshape(n_rows, block, n_cols, block).swapaxes(1, 2).copy()
    return blocks, (h, w)


def merge_blocks(blocks: np.ndarray, original_shape: tuple[int, int],
                 block: int = BLOCK_SIZE) -> np.ndarray:
    """Inverse of split_blocks — crops back to the original (H, W)."""
    n_rows, n_cols, _, _ = blocks.shape
    H = n_rows * block
    W = n_cols * block
    merged = blocks.swapaxes(1, 2).reshape(H, W)
    h, w = original_shape
    return merged[:h, :w]


# ---------- DCT: direct, generic NxM ----------------------------------------

def dct2_naive(block: np.ndarray) -> np.ndarray:
    """2D DCT-II computed directly from the sum definition.

    Works for any block size NxM. Complexity: O((N*M)^2) — quadratic in block
    area. For an 8x8 block this is 64*64 = 4096 multiplications per block.

    Coefficients are stored as float32 (or float64); they are real-valued and
    typically span roughly [-1024, 1024] for 8x8 blocks of [0,255] inputs,
    so float32 is plenty.
    """
    N, M = block.shape
    result = np.zeros((N, M), dtype=np.float64)
    # precompute cosines
    cos_x = np.zeros((M, M))
    for u in range(M):
        for x in range(M):
            cos_x[u, x] = np.cos((2 * x + 1) * u * np.pi / (2 * M))
    cos_y = np.zeros((N, N))
    for v in range(N):
        for y in range(N):
            cos_y[v, y] = np.cos((2 * y + 1) * v * np.pi / (2 * N))

    for v in range(N):
        av = 1.0 / np.sqrt(2) if v == 0 else 1.0
        for u in range(M):
            au = 1.0 / np.sqrt(2) if u == 0 else 1.0
            s = 0.0
            for y in range(N):
                for x in range(M):
                    s += block[y, x] * cos_x[u, x] * cos_y[v, y]
            result[v, u] = (2.0 / np.sqrt(N * M)) * au * av * s
    return result.astype(np.float32)


def idct2_naive(coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT-II, direct sum, NxM generic."""
    N, M = coeffs.shape
    result = np.zeros((N, M), dtype=np.float64)
    for y in range(N):
        for x in range(M):
            s = 0.0
            for v in range(N):
                av = 1.0 / np.sqrt(2) if v == 0 else 1.0
                for u in range(M):
                    au = 1.0 / np.sqrt(2) if u == 0 else 1.0
                    s += (av * au * coeffs[v, u]
                          * np.cos((2 * x + 1) * u * np.pi / (2 * M))
                          * np.cos((2 * y + 1) * v * np.pi / (2 * N)))
            result[y, x] = (2.0 / np.sqrt(N * M)) * s
    return result.astype(np.float32)


# ---------- DCT: matrix-multiplication form, 8x8 only -----------------------

def _dct_matrix(N: int = BLOCK_SIZE) -> np.ndarray:
    """Build the DCT matrix C such that S = C^T @ s @ C for an NxN block.

    C[i, j] = c_j * cos((2i+1) j pi / (2N))
    c_0 = 1/sqrt(2), c_j = 1 for j>0, then overall scale 2/sqrt(N*N) = 2/N.
    Combined: C[i, j] = sqrt(2/N) * k_j * cos(...)  with k_0 = 1/sqrt(2).
    """
    C = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            kj = 1.0 / np.sqrt(2) if j == 0 else 1.0
            C[i, j] = np.sqrt(2.0 / N) * kj * np.cos((2 * i + 1) * j * np.pi / (2 * N))
    return C


_C8 = _dct_matrix(BLOCK_SIZE)


def dct2_matrix(block: np.ndarray) -> np.ndarray:
    """Forward 2D DCT via matrix multiplication: S = C^T @ s @ C."""
    s = block.astype(np.float64)
    return (_C8.T @ s @ _C8).astype(np.float32)


def idct2_matrix(coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT via matrix multiplication: s = C @ S @ C^T."""
    S = coeffs.astype(np.float64)
    return (_C8 @ S @ _C8.T).astype(np.float32)


def dct2_blocks(blocks: np.ndarray) -> np.ndarray:
    """Apply forward DCT to every 8x8 block in an (Nr, Nc, 8, 8) array."""
    # efficient: einsum for C^T @ block @ C
    return np.einsum("ij,rcjk,kl->rcil", _C8.T, blocks.astype(np.float64), _C8).astype(np.float32)


def idct2_blocks(blocks: np.ndarray) -> np.ndarray:
    """Apply inverse DCT to every 8x8 block in an (Nr, Nc, 8, 8) array."""
    return np.einsum("ij,rcjk,kl->rcil", _C8, blocks.astype(np.float64), _C8.T).astype(np.float32)


# ---------- Quantisation ----------------------------------------------------

# Standard JPEG luminance quantisation table (ITU-T.81 Annex K.1)
STD_LUMA_Q = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int32)

# Standard JPEG chrominance quantisation table (ITU-T.81 Annex K.2)
STD_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.int32)


def quantise(coeffs: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """c'_{yx} = round(c_{yx} / q_{yx}).

    Accepts a single 8x8 block or an (Nr, Nc, 8, 8) stack.
    """
    return np.round(coeffs / q_table).astype(np.int32)


def dequantise(q_coeffs: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """c_{yx} = c'_{yx} * q_{yx}."""
    return (q_coeffs * q_table).astype(np.float32)


def quality_scale_q(q_table: np.ndarray, quality: int) -> np.ndarray:
    """Task 2.6 — rescale a standard Q table for a given quality level 1..99.

    S = 5000/Q         for Q in [1, 50)
    S = 200 - 2*Q      for Q in [50, 100)
    q'_{yx} = ceil(q_{yx} * S / 100)   (clamped to at least 1)
    """
    q = int(quality)
    if q < 1:
        q = 1
    if q > 99:
        q = 99
    if q < 50:
        S = 5000 / q
    else:
        S = 200 - 2 * q
    scaled = np.ceil(q_table.astype(np.float64) * S / 100.0).astype(np.int32)
    scaled = np.clip(scaled, 1, 255)
    return scaled
