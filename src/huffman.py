"""
Task 2.5 — Huffman coding using the standard JPEG Huffman tables
(ITU-T.81 Annex K.3).

We use the fixed canonical tables for baseline JPEG:
    - DC luminance      (K.3, Table K.3)
    - DC chrominance    (K.3, Table K.4)
    - AC luminance      (K.3, Table K.5)
    - AC chrominance    (K.3, Table K.6)

Each table is given as BITS[1..16] = number of codes of each length, and
HUFFVAL[] = the symbols in increasing code-length order. From these we
derive the canonical codes per Annex C.

The encoder / decoder operate on bitstrings (BitWriter / BitReader).

DC symbol  = SIZE (0..11)
AC symbol  = (RUN << 4) | SIZE  (0x00 = EOB, 0xF0 = ZRL)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------- Standard JPEG Huffman tables ------------------------------------

# DC luminance (ITU-T.81 Table K.3)
_DC_LUMA_BITS = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # index 0 unused
_DC_LUMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# DC chrominance (Table K.4)
_DC_CHROMA_BITS = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_DC_CHROMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# AC luminance (Table K.5) — BITS[1..16]
_AC_LUMA_BITS = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]
_AC_LUMA_VALS = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]

# AC chrominance (Table K.6)
_AC_CHROMA_BITS = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
_AC_CHROMA_VALS = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]


def build_huffman_table(bits_counts: List[int], huffvals: List[int]) -> Dict[int, Tuple[int, int]]:
    """Build canonical Huffman codes per JPEG Annex C.

    Parameters
    ----------
    bits_counts : list of 17 integers; bits_counts[L] = number of codes of
                  length L, L = 1..16 (index 0 unused).
    huffvals    : symbols in increasing code-length order.

    Returns
    -------
    dict symbol -> (code, length)
    """
    huffsize: List[int] = []
    for L in range(1, 17):
        huffsize.extend([L] * bits_counts[L])

    huffcode: List[int] = []
    code = 0
    si = huffsize[0] if huffsize else 0
    for size in huffsize:
        while size > si:
            code <<= 1
            si += 1
        huffcode.append(code)
        code += 1

    return {sym: (c, s) for sym, c, s in zip(huffvals, huffcode, huffsize)}


# Pre-built standard tables
DC_LUMA_TABLE   = build_huffman_table(_DC_LUMA_BITS,   _DC_LUMA_VALS)
DC_CHROMA_TABLE = build_huffman_table(_DC_CHROMA_BITS, _DC_CHROMA_VALS)
AC_LUMA_TABLE   = build_huffman_table(_AC_LUMA_BITS,   _AC_LUMA_VALS)
AC_CHROMA_TABLE = build_huffman_table(_AC_CHROMA_BITS, _AC_CHROMA_VALS)

# Inverse tables for decoding: (code, length) -> symbol
def _invert(table: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    return {v: k for k, v in table.items()}

DC_LUMA_DECODE   = _invert(DC_LUMA_TABLE)
DC_CHROMA_DECODE = _invert(DC_CHROMA_TABLE)
AC_LUMA_DECODE   = _invert(AC_LUMA_TABLE)
AC_CHROMA_DECODE = _invert(AC_CHROMA_TABLE)


# ---------- Bit I/O ---------------------------------------------------------

class BitWriter:
    """Append-only bit buffer. `to_bytes()` pads the final byte with 1-bits
    (JPEG convention)."""

    def __init__(self) -> None:
        self._bits: List[int] = []

    def write(self, value: int, nbits: int) -> None:
        for i in range(nbits - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def write_bits_str(self, s: str) -> None:
        for ch in s:
            self._bits.append(1 if ch == "1" else 0)

    def bit_length(self) -> int:
        return len(self._bits)

    def to_bytes(self) -> bytes:
        bits = list(self._bits)
        # pad with 1s to byte boundary
        while len(bits) % 8 != 0:
            bits.append(1)
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i:i + 8]:
                byte = (byte << 1) | b
            out.append(byte)
        return bytes(out)


class BitReader:
    """Read bits in MSB-first order from a bytes buffer."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0  # bit position

    def read(self, nbits: int) -> int:
        value = 0
        for _ in range(nbits):
            byte_idx = self._pos >> 3
            bit_idx = 7 - (self._pos & 7)
            value = (value << 1) | ((self._data[byte_idx] >> bit_idx) & 1)
            self._pos += 1
        return value


# ---------- Block-level encode / decode -------------------------------------

def encode_block(
    diff_dc: int,
    ac_triples: List[Tuple[int, int, int]],
    bw: BitWriter,
    dc_table: Dict[int, Tuple[int, int]],
    ac_table: Dict[int, Tuple[int, int]],
) -> None:
    """Write one block's worth of compressed data to `bw`.

    diff_dc   : the DC coefficient's differential value (already diffed)
    ac_triples: output of run_length_encode_ac (list of (run, size, bits))
    """
    # --- DC ---
    from .entropy import vli_bits  # local import to avoid cycles
    size, bits = vli_bits(diff_dc)
    code, codelen = dc_table[size]
    bw.write(code, codelen)
    if size > 0:
        bw.write(bits, size)

    # --- AC ---
    for run, size, bits in ac_triples:
        symbol = (run << 4) | size
        code, codelen = ac_table[symbol]
        bw.write(code, codelen)
        if size > 0:
            bw.write(bits, size)


def _decode_symbol(br: BitReader, decode_table: Dict[Tuple[int, int], int]) -> int:
    """Read one Huffman-coded symbol using a decode_table {(code,len): sym}."""
    code = 0
    for length in range(1, 17):
        code = (code << 1) | br.read(1)
        sym = decode_table.get((code, length))
        if sym is not None:
            return sym
    raise ValueError("Invalid Huffman code")


def decode_block(
    br: BitReader,
    dc_decode: Dict[Tuple[int, int], int],
    ac_decode: Dict[Tuple[int, int], int],
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Decode one block — returns (diff_dc_value, list_of_ac_triples)."""
    from .entropy import vli_decode  # local import

    # DC
    size = _decode_symbol(br, dc_decode)
    bits = br.read(size) if size > 0 else 0
    diff_dc = vli_decode(size, bits)

    # AC
    ac: List[Tuple[int, int, int]] = []
    count = 0
    while count < 63:
        sym = _decode_symbol(br, ac_decode)
        if sym == 0x00:
            ac.append((0, 0, 0))  # EOB
            break
        if sym == 0xF0:
            ac.append((15, 0, 0))  # ZRL
            count += 16
            continue
        run = sym >> 4
        size = sym & 0x0F
        bits = br.read(size)
        ac.append((run, size, bits))
        count += run + 1
    return diff_dc, ac
