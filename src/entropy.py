"""
Tasks 2.3 + 2.4 — Differential coding, RLE, variable-length categories
per ITU-T.81 (JPEG standard).

DC coefficients:
    differential_encode_dc(dc_list) -> list of differences (dc[i] - dc[i-1], dc[0] kept)
    Then each difference is encoded as (SIZE, VALUE), where SIZE is the
    category (number of bits) per JPEG Table F.1 and VALUE is the VLI-coded bits.

AC coefficients (63 values per 8x8 block after dropping the DC):
    run_length_encode_ac(ac_list) -> list of (RUN, SIZE, VALUE) triples,
        where RUN is the number of zeros preceding the next non-zero coefficient
        (max 15 — longer runs produce ZRL codes = (15, 0, 0)), and SIZE/VALUE
        are the VLI category+bits of that non-zero coefficient.
        Trailing zeros are replaced by a single EOB marker = (0, 0, 0).

Variable-length integer (VLI) coding (JPEG Table F.1):
    SIZE    VALUE range
    0       0
    1       -1, 1
    2       -3..-2, 2..3
    3       -7..-4, 4..7
    ...
    n       -(2^n-1) .. -(2^(n-1)),  2^(n-1) .. (2^n-1)

For negative values, the bit pattern stored is the one's complement of |value|
truncated to SIZE bits (equivalent to value - 1 written in SIZE bits).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------- DC differential coding ------------------------------------------

def differential_encode_dc(dc_list: List[int] | np.ndarray) -> List[int]:
    """Return list of differences: [dc0, dc1-dc0, dc2-dc1, ...]."""
    dc = [int(x) for x in dc_list]
    if not dc:
        return []
    diffs = [dc[0]]
    for i in range(1, len(dc)):
        diffs.append(dc[i] - dc[i - 1])
    return diffs


def differential_decode_dc(diffs: List[int]) -> List[int]:
    """Invert differential_encode_dc."""
    if not diffs:
        return []
    dc = [int(diffs[0])]
    for i in range(1, len(diffs)):
        dc.append(dc[-1] + int(diffs[i]))
    return dc


# ---------- VLI helpers (ITU-T.81 Table F.1) --------------------------------

def vli_category(value: int) -> int:
    """Return the number of bits needed to represent `value` in JPEG VLI form."""
    if value == 0:
        return 0
    v = abs(int(value))
    # smallest n such that 2^n > v  ==>  n = floor(log2(v)) + 1
    n = 0
    while v:
        v >>= 1
        n += 1
    return n


def vli_bits(value: int) -> Tuple[int, int]:
    """Return (SIZE, BITS) for a JPEG VLI-coded integer.

    BITS is an integer whose SIZE low bits are the encoded payload.
    Positive v:  BITS = v
    Negative v:  BITS = v - 1 (masked to SIZE bits; same as (v + (1<<SIZE) - 1))
    Zero: (0, 0)
    """
    if value == 0:
        return 0, 0
    size = vli_category(value)
    if value > 0:
        bits = value
    else:
        bits = value + (1 << size) - 1
    bits &= (1 << size) - 1
    return size, bits


def vli_decode(size: int, bits: int) -> int:
    """Inverse of vli_bits."""
    if size == 0:
        return 0
    # If the top bit is 0, the value is negative.
    if bits & (1 << (size - 1)):
        return int(bits)
    return int(bits) - ((1 << size) - 1)


# ---------- AC RLE ----------------------------------------------------------

EOB = (0, 0, 0)          # (run, size, bits)
ZRL = (15, 0, 0)         # 16-zero run-length marker


def run_length_encode_ac(ac: List[int] | np.ndarray) -> List[Tuple[int, int, int]]:
    """RLE-encode the 63 AC coefficients of a single block.

    Returns a list of (run, size, bits) triples. Uses ZRL = (15, 0, 0) for
    runs of 16 zeros, EOB = (0, 0, 0) at the end if any trailing zeros remain.
    """
    ac = [int(v) for v in ac]
    out: List[Tuple[int, int, int]] = []
    run = 0
    # find last non-zero
    last_nz = -1
    for i in range(len(ac) - 1, -1, -1):
        if ac[i] != 0:
            last_nz = i
            break
    if last_nz == -1:
        return [EOB]

    i = 0
    while i <= last_nz:
        if ac[i] == 0:
            run += 1
            if run == 16:
                out.append(ZRL)
                run = 0
            i += 1
        else:
            size, bits = vli_bits(ac[i])
            out.append((run, size, bits))
            run = 0
            i += 1
    if last_nz < len(ac) - 1:
        out.append(EOB)
    return out


def run_length_decode_ac(triples: List[Tuple[int, int, int]], length: int = 63) -> List[int]:
    """Invert run_length_encode_ac to a length-N list of AC coefficients."""
    out: List[int] = []
    for run, size, bits in triples:
        if (run, size) == (0, 0):
            # EOB — pad with zeros
            while len(out) < length:
                out.append(0)
            break
        if (run, size) == (15, 0):
            # ZRL — 16 zeros
            out.extend([0] * 16)
            continue
        out.extend([0] * run)
        out.append(vli_decode(size, bits))
    # pad if needed (defensive)
    while len(out) < length:
        out.append(0)
    return out[:length]
