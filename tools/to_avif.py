#!/usr/bin/env python3
"""Convert chapter figures to AVIF for the web, keeping a JPEG sibling for the PDF.

AVIF is what a reader downloads; LaTeX cannot read it, so every converted file also
gets a `.jpg` beside it that `filters/pdf-images.lua` substitutes when Quarto renders
the book to PDF. Both belong in the repository — the JPEG is never fetched by a
browser, so it costs the reader nothing.

Conversion is not automatic. A PNG of a diagram typically shrinks by 70-80%, which is
the whole point, but a photograph that is *already* a JPEG often barely improves and
can even grow. Files that do not beat the threshold are left exactly as they are.

    python3 tools/to_avif.py figures/01-introduction            # convert in place
    python3 tools/to_avif.py figures/01-introduction --dry-run  # just report

`figures/social-card.png` is skipped by name wherever it appears: it is the Open Graph
image, and the services that unfurl links do not reliably decode AVIF.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

SOURCE_SUFFIXES = {".png", ".jpg", ".jpeg"}
NEVER_CONVERT = {"social-card.png"}
AVIF_QUALITY = "82"  # measured at 42 dB PSNR on line art — visually lossless here
JPEG_QUALITY = "88"  # the PDF sibling; print-ish, never downloaded by a browser


def kb(path: pathlib.Path) -> float:
    """Size of a file in kilobytes."""
    return path.stat().st_size / 1024


def convert(src: pathlib.Path, threshold: float, dry_run: bool) -> tuple[str, float, float]:
    """Encode one file. Returns (verdict, source KB, avif KB)."""
    avif = src.with_suffix(".avif")
    tmp = src.with_suffix(".avif.tmp")
    try:
        # The AVIF: prefix is load-bearing. ImageMagick picks the encoder from the
        # extension, and a temp name ending in .tmp silently gets the wrong one — the
        # output then comes back the same size as the input and nothing converts.
        subprocess.run(["magick", str(src), "-quality", AVIF_QUALITY, f"AVIF:{tmp}"], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return f"FAILED ({exc.stderr.decode()[:60]})", kb(src), 0.0

    before, after = kb(src), kb(tmp)
    saved = 1 - after / before if before else 0

    if saved < threshold:
        tmp.unlink(missing_ok=True)
        return "kept as-is", before, after

    if dry_run:
        tmp.unlink(missing_ok=True)
        return "would convert", before, after

    # The PDF needs a raster LaTeX can read, and LaTeX reads both PNG and JPEG. So the
    # sibling is whichever is smaller: a freshly encoded JPEG, or the original we were
    # about to throw away. Re-encoding a scanned line drawing as JPEG can easily come
    # out larger than the PNG it came from, and then keeping the PNG is simply better.
    jpeg_tmp = src.with_suffix(".jpg.tmp")
    subprocess.run(["magick", str(src), "-quality", JPEG_QUALITY, f"JPEG:{jpeg_tmp}"], check=True, capture_output=True)

    tmp.replace(avif)
    if kb(jpeg_tmp) < before:
        jpeg_tmp.replace(src.with_suffix(".jpg"))
        if src.suffix.lower() != ".jpg":
            src.unlink()
    else:
        jpeg_tmp.unlink(missing_ok=True)  # the original is the better LaTeX sibling
    return "converted", before, after


def main() -> int:
    """Walk the given directories, convert what is worth converting, and report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--threshold", type=float, default=0.20, help="minimum fractional saving to bother (default 0.20)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not shutil.which("magick"):
        print("ImageMagick not found — brew install imagemagick", file=sys.stderr)
        return 1

    total_before = total_after = 0.0
    converted = 0
    for d in args.dirs:
        for src in sorted(p for p in d.rglob("*") if p.suffix.lower() in SOURCE_SUFFIXES):
            if src.name in NEVER_CONVERT:
                print(f"{'skipped':14s} {src}  (Open Graph image — must stay PNG)")
                continue
            if src.with_suffix(".avif").exists():
                # This is the LaTeX sibling of a file already converted, not a source.
                # Re-encoding it would rebuild the AVIF from an already-lossy JPEG.
                print(f"{'sibling':14s} {src}  (kept for the PDF; not a conversion source)")
                continue
            verdict, before, after = convert(src, args.threshold, args.dry_run)
            saved = 1 - after / before if before and after else 0
            print(f"{verdict:14s} {src}  {before:6.0f}K -> {after:5.0f}K  ({saved:+.0%})")
            if verdict in ("converted", "would convert"):
                converted += 1
                total_before += before
                total_after += after

    if converted:
        drop = 1 - total_after / total_before
        print(
            f"\n{converted} converted: {total_before:.0f}K -> {total_after:.0f}K downloaded by a reader ({drop:.0%} less)"
        )
    else:
        print("\nnothing worth converting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
