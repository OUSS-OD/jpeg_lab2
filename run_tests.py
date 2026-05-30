"""Quick sanity tests for every module."""

import numpy as np

from src.color_space import rgb_to_ycbcr, ycbcr_to_rgb
from src.resampling import (
    downsample_2x, upsample_2x, linear_interp, linear_spline,
    bilinear_interp, resize_bilinear,
)
from src.dct import (
    STD_LUMA_Q, dct2_naive, idct2_naive, dct2_matrix, idct2_matrix,
    dct2_blocks, idct2_blocks, split_blocks, merge_blocks,
    quantise, dequantise, quality_scale_q,
)
from src.zigzag import zigzag, inverse_zigzag
from src.entropy import (
    differential_encode_dc, differential_decode_dc,
    run_length_encode_ac, run_length_decode_ac,
    vli_bits, vli_decode, vli_category,
)
from src.huffman import (
    BitWriter, BitReader, DC_LUMA_TABLE, AC_LUMA_TABLE,
    DC_LUMA_DECODE, AC_LUMA_DECODE, encode_block, decode_block,
)
from src.compressor import compress, decompress
from src.raw_format import save_raw, load_raw, MODE_RGB, MODE_GRAY, MODE_BW


def ok(msg):
    print(f"  [OK] {msg}")


# --- color space
rgb = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
ycbcr = rgb_to_ycbcr(rgb)
rgb_back = ycbcr_to_rgb(ycbcr)
err = np.abs(rgb.astype(int) - rgb_back.astype(int)).max()
assert err <= 4, f"color roundtrip max error {err}"
ok(f"RGB<->YCbCr roundtrip max err = {err}")

# --- resampling
img = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
down = downsample_2x(img)
up = upsample_2x(down)
assert down.shape == (32, 32)
assert up.shape == (64, 64)
ok("downsample_2x / upsample_2x shapes")

assert abs(linear_interp(0, 10, 0, 100, 5) - 50) < 1e-9
ok("linear_interp")

xs = np.array([0.0, 1.0, 3.0])
ys = np.array([0.0, 1.0, 0.0])
assert abs(linear_spline(xs, ys, 0.5) - 0.5) < 1e-9
assert abs(linear_spline(xs, ys, 2.0) - 0.5) < 1e-9
ok("linear_spline")

v = bilinear_interp(0, 1, 0, 1, 0, 10, 10, 20, 0.5, 0.5)
assert abs(v - 10.0) < 1e-9
ok(f"bilinear_interp = {v}")

resized = resize_bilinear(img, (100, 50))
assert resized.shape == (100, 50)
ok("resize_bilinear shape")

# --- DCT
# Test naive vs matrix on a small 8x8 block
block = np.random.randint(0, 256, size=(8, 8)).astype(np.float32)
c_naive = dct2_naive(block)
c_matrix = dct2_matrix(block)
diff = np.abs(c_naive - c_matrix).max()
print(f"    naive vs matrix DCT max diff: {diff}")
assert diff < 1e-2, f"DCT implementations disagree: {diff}"
ok("dct2_naive == dct2_matrix")

rec = idct2_matrix(c_matrix)
rec_err = np.abs(rec - block).max()
assert rec_err < 1e-2, f"DCT->IDCT error {rec_err}"
ok(f"DCT roundtrip err = {rec_err:.2e}")

# blocks
channel = np.random.randint(0, 256, size=(40, 50), dtype=np.uint8).astype(np.float32)
blocks, orig = split_blocks(channel)
coeffs = dct2_blocks(blocks)
rec_blocks = idct2_blocks(coeffs)
merged = merge_blocks(rec_blocks, orig)
err = np.abs(merged - channel).max()
assert err < 1e-2, f"block roundtrip err {err}"
ok(f"split->DCT->IDCT->merge err = {err:.2e}")

# quantisation + quality
q = quality_scale_q(STD_LUMA_Q, 50)
qc = quantise(coeffs, q)
dqc = dequantise(qc, q)
ok(f"quantise/dequantise shapes: {qc.shape} / {dqc.shape}")

q_high = quality_scale_q(STD_LUMA_Q, 90)
q_low  = quality_scale_q(STD_LUMA_Q, 10)
print(f"    Q(90) mean = {q_high.mean():.1f}, Q(10) mean = {q_low.mean():.1f}")
assert q_high.mean() < q_low.mean(), "higher quality should produce smaller Q values"
ok("quality_scale_q ordering")

# --- zigzag
m = np.arange(64).reshape(8, 8)
zz = zigzag(m)
# JPEG zigzag starts [0, 1, 8, 16, 9, 2, 3, 10, ...]
assert zz[0] == 0 and zz[1] == 1 and zz[2] == 8 and zz[3] == 16
back = inverse_zigzag(zz, (8, 8))
assert np.array_equal(back, m)
ok("zigzag / inverse_zigzag roundtrip")

# rectangular
m_rect = np.arange(12).reshape(3, 4)
zz_r = zigzag(m_rect)
back_r = inverse_zigzag(zz_r, (3, 4))
assert np.array_equal(back_r, m_rect)
ok("zigzag on rectangular matrix")

# --- entropy (VLI + DC diff + AC RLE)
for v in [0, 1, -1, 5, -5, 100, -100, 1023, -1023]:
    s, b = vli_bits(v)
    decoded = vli_decode(s, b)
    assert decoded == v, f"VLI roundtrip failed for {v}: got {decoded}"
ok("VLI roundtrip for several values")

dc = [5, 7, 4, 4, 10]
diffs = differential_encode_dc(dc)
assert diffs == [5, 2, -3, 0, 6]
assert differential_decode_dc(diffs) == dc
ok("DC differential encode/decode")

ac = [3, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5] + [0] * 45
triples = run_length_encode_ac(ac)
back_ac = run_length_decode_ac(triples, length=63)
assert back_ac == ac
ok(f"AC RLE roundtrip ({len(triples)} triples)")

# AC with all zeros
assert run_length_encode_ac([0] * 63) == [(0, 0, 0)]
ok("AC RLE for all-zero block -> EOB")

# --- huffman
bw = BitWriter()
# DC diff = 5, AC = all zeros (EOB)
encode_block(5, [(0, 0, 0)], bw, DC_LUMA_TABLE, AC_LUMA_TABLE)
data = bw.to_bytes()
br = BitReader(data)
diff, ac = decode_block(br, DC_LUMA_DECODE, AC_LUMA_DECODE)
assert diff == 5 and ac == [(0, 0, 0)]
ok(f"Huffman encode_block/decode_block roundtrip ({len(data)} bytes)")

# --- raw format
import tempfile, os
tmp_path = os.path.join(tempfile.gettempdir(), "t.myrw")
rgb_img = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
save_raw(tmp_path, rgb_img, MODE_RGB)
back, mode = load_raw(tmp_path)
assert np.array_equal(back, rgb_img) and mode == MODE_RGB
ok("raw format RGB roundtrip")

gray_img = np.random.randint(0, 256, (7, 9), dtype=np.uint8)
save_raw(tmp_path, gray_img, MODE_GRAY)
back, mode = load_raw(tmp_path)
assert np.array_equal(back, gray_img) and mode == MODE_GRAY
ok("raw format GRAY roundtrip")

# --- end-to-end compressor on a small image
small = (np.random.rand(32, 48, 3) * 255).astype(np.uint8)
for q in [10, 30, 50, 80]:
    packed = compress(small, quality=q)
    recovered = decompress(packed)
    assert recovered.shape == small.shape
    err = np.abs(recovered.astype(int) - small.astype(int)).mean()
    print(f"    q={q}: {len(packed)} bytes, mean abs err = {err:.1f}")
ok("compress/decompress color roundtrip")

# grayscale
small_g = (np.random.rand(32, 48) * 255).astype(np.uint8)
for q in [10, 50, 90]:
    packed = compress(small_g, quality=q)
    recovered = decompress(packed)
    assert recovered.shape == small_g.shape
    err = np.abs(recovered.astype(int) - small_g.astype(int)).mean()
    print(f"    q={q}: {len(packed)} bytes, mean abs err = {err:.1f}")
ok("compress/decompress grayscale roundtrip")

print("\nALL TESTS PASSED")
