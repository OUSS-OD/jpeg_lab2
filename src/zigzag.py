"""
Task 2.2 — Zigzag traversal of a matrix.

Works for square NxN matrices as well as rectangular NxM matrices.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def zigzag_indices(n: int, m: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (rows, cols) index arrays for the zigzag order of an n x m matrix."""
    rows: list[int] = []
    cols: list[int] = []
    # Traverse anti-diagonals. Diagonal index d = r + c goes 0 .. n+m-2.
    # Even d: go up (r decreases, c increases), odd d: go down.
    for d in range(n + m - 1):
        if d % 2 == 0:
            # upward: start from (min(d, n-1), max(0, d-n+1)), moving up-right
            r = min(d, n - 1)
            c = d - r
            while r >= 0 and c < m:
                rows.append(r)
                cols.append(c)
                r -= 1
                c += 1
        else:
            # downward: start from (max(0, d-m+1), min(d, m-1)), moving down-left
            c = min(d, m - 1)
            r = d - c
            while c >= 0 and r < n:
                rows.append(r)
                cols.append(c)
                r += 1
                c -= 1
    return tuple(rows), tuple(cols)


def zigzag(matrix: np.ndarray) -> np.ndarray:
    """Return a 1D array of `matrix` values in zigzag order."""
    n, m = matrix.shape
    rows, cols = zigzag_indices(n, m)
    return matrix[np.array(rows), np.array(cols)].copy()


def inverse_zigzag(vec: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct an n x m matrix from its zigzag-ordered values."""
    n, m = shape
    rows, cols = zigzag_indices(n, m)
    out = np.zeros((n, m), dtype=vec.dtype)
    out[np.array(rows), np.array(cols)] = vec
    return out
