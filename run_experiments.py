"""
Run full compression experiments on every test image and produce:
    - compressed .myjpg files  (one per image per quality level)
    - decompressed .png files  (to eyeball quality)
    - size-vs-quality plots    (one PNG per image + combined plot)
    - size_report.csv          (raw numbers)
Outputs go to the ./output directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.compressor import compress_to_file, decompress_from_file
from src.raw_format import save_raw, raw_file_size, MODE_RGB, MODE_GRAY

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def load_png(path):
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def run_for_image(name: str, img: np.ndarray, qualities):
    mode_str = "color" if img.ndim == 3 else "grayscale"
    print(f"\n=== {name} ({mode_str}, shape={img.shape}) ===")
    results = []
    for q in qualities:
        cfile = OUT / f"{name}_q{q:02d}.myjpg"
        nbytes = compress_to_file(img, cfile, quality=q)

        recovered = decompress_from_file(cfile)
        dfile = OUT / f"{name}_q{q:02d}_decoded.png"
        if recovered.ndim == 2:
            Image.fromarray(recovered, mode="L").save(dfile)
        else:
            Image.fromarray(recovered).save(dfile)

        # error metric
        err = np.abs(recovered.astype(int) - img.astype(int)).mean()
        psnr = 20 * np.log10(255 / (np.sqrt(((recovered.astype(float) - img.astype(float)) ** 2).mean()) + 1e-9))
        raw_size = img.size  # total bytes for the uncompressed pixel data
        print(f"  q={q:3d}  size={nbytes:7d} B  "
              f"ratio={raw_size/nbytes:6.2f}x  MAE={err:5.2f}  PSNR={psnr:5.2f} dB")
        results.append((q, nbytes, err, psnr))
    return results


def main():
    imgs_dir = Path("images")
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # load all images
    lena      = load_png(imgs_dir / "lena_synth.png")
    color     = load_png(imgs_dir / "color_pattern.png")
    gray      = np.asarray(Image.open(imgs_dir / "color_gray.png"))
    bw_round  = np.asarray(Image.open(imgs_dir / "color_bw_round.png"))
    bw_dither = np.asarray(Image.open(imgs_dir / "color_bw_dither.png"))

    # Also save raw versions of each (Task 1.1)
    save_raw(OUT / "lena_synth.myrw",       lena,      MODE_RGB)
    save_raw(OUT / "color_pattern.myrw",    color,     MODE_RGB)
    save_raw(OUT / "color_gray.myrw",       gray,      MODE_GRAY)
    save_raw(OUT / "color_bw_round.myrw",   bw_round,  MODE_GRAY)
    save_raw(OUT / "color_bw_dither.myrw",  bw_dither, MODE_GRAY)

    # Print raw vs png size comparison (Task 1.1)
    print("\n--- Raw format size vs PNG (Task 1.1) ---")
    print(f"{'name':<22}{'raw B':>10}{'png B':>10}{'ratio':>10}")
    for name, arr, mode, pngname in [
        ("lena_synth",      lena,      MODE_RGB,  "lena_synth.png"),
        ("color_pattern",   color,     MODE_RGB,  "color_pattern.png"),
        ("color_gray",      gray,      MODE_GRAY, "color_gray.png"),
        ("color_bw_round",  bw_round,  MODE_GRAY, "color_bw_round.png"),
        ("color_bw_dither", bw_dither, MODE_GRAY, "color_bw_dither.png"),
    ]:
        raw_s = raw_file_size(arr, mode)
        png_s = (imgs_dir / pngname).stat().st_size
        print(f"{name:<22}{raw_s:>10}{png_s:>10}{raw_s/png_s:>10.2f}")

    # Run compression experiments
    all_results = {}
    all_results["lena_synth"]      = run_for_image("lena_synth",      lena,      qualities)
    all_results["color_pattern"]   = run_for_image("color_pattern",   color,     qualities)
    all_results["gray"]            = run_for_image("gray",            gray,      qualities)
    all_results["bw_round"]        = run_for_image("bw_round",        bw_round,  qualities)
    all_results["bw_dither"]       = run_for_image("bw_dither",       bw_dither, qualities)

    # Write CSV
    with open(OUT / "size_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "quality", "size_bytes", "mae", "psnr_db"])
        for name, rows in all_results.items():
            for q, nb, err, psnr in rows:
                w.writerow([name, q, nb, f"{err:.3f}", f"{psnr:.3f}"])

    # Combined plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, rows in all_results.items():
        qs = [r[0] for r in rows]
        sizes = [r[1] for r in rows]
        ax.plot(qs, sizes, "o-", label=name)
    ax.set_xlabel("Quality")
    ax.set_ylabel("Compressed size (bytes)")
    ax.set_title("Compressed size vs quality")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "size_vs_quality_all.png", dpi=120)
    plt.close(fig)

    # Individual plots (per Task requirement: one plot per test image)
    for name, rows in all_results.items():
        qs = [r[0] for r in rows]
        sizes = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(qs, sizes, "o-", color="tab:blue")
        ax.set_xlabel("Quality")
        ax.set_ylabel("Compressed size (bytes)")
        ax.set_title(f"Size vs quality — {name}")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUT / f"size_vs_quality_{name}.png", dpi=120)
        plt.close(fig)

    print(f"\nDone. Outputs in {OUT.resolve()}")


if __name__ == "__main__":
    main()
