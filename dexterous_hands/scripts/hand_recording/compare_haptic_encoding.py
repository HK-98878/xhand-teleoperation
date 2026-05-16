"""
Compare old (linear ±100) vs new (linear-xy + sqrt-z) haptic encoding.

Finds frames in a captured session with varied contact levels (zero, light,
intermediate, peak) and renders both encodings side by side as PNGs so you
can visually evaluate whether the new scheme actually improves contact
visibility.

Reads:
  <session_dir>/haptic.bin
  <session_dir>/haptic_index.jsonl
  <session_dir>/session_meta.json

Writes:
  <out_dir>/haptic_compare_<label>_idx<i>.png    one per selected frame

Usage:
    python compare_haptic_encoding.py path/to/session_YYYYMMDD_HHMMSS
                                       [--out haptic_comparison]
                                       [--upscale 20]
                                       [--n-frames-per-tier 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import the new encoding from the export module
sys.path.insert(0, str(Path(__file__).parent))
from parse_recordings import (
    DISPLAY_ROWS, DISPLAY_COLS, GRID_ROWS, GRID_COLS,
    SEPARATOR_PX, SEPARATOR_COLOUR,
    HAPTIC_XY_BOUND, HAPTIC_Z_MIN, HAPTIC_Z_MAX,
    haptic_sample_to_image as encode_new,
)


OLD_VALUE_RANGE = 100.0  # what export.py used before this change


def encode_old(sample: np.ndarray, upscale: int = 1) -> np.ndarray:
    """The previous encoding: linear ±100 on all three channels."""
    n_fingers = sample.shape[0]
    finger_imgs = []
    for f in range(n_fingers):
        v = sample[f]
        n = v.shape[0]
        target = GRID_ROWS * GRID_COLS
        if n < target:
            padded = np.zeros((target, 3), dtype=v.dtype)
            padded[:n] = v
            v = padded
        elif n > target:
            v = v[:target]
        grid = v.reshape(GRID_ROWS, GRID_COLS, 3)
        rotated = np.rot90(grid, k=1)
        scaled = (rotated + OLD_VALUE_RANGE) * (255.0 / (2 * OLD_VALUE_RANGE))
        np.clip(scaled, 0, 255, out=scaled)
        finger_imgs.append(scaled.astype(np.uint8))
    separator = np.zeros((DISPLAY_ROWS, SEPARATOR_PX, 3), dtype=np.uint8)
    separator[:] = SEPARATOR_COLOUR
    parts: list[np.ndarray] = []
    for i, fi in enumerate(finger_imgs):
        if i > 0:
            parts.append(separator)
        parts.append(fi)
    img = np.concatenate(parts, axis=1)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    return img


def select_representative_frames(haptic_mmap: np.ndarray, n_per_tier: int = 2) -> list[tuple[int, str]]:
    """Pick frames at varied contact magnitudes. Returns [(index, label), ...]."""
    # Magnitude per sample: sum of |z| across all taxels (z is the dominant signal).
    z = haptic_mmap[..., 2]  # (n_samples, n_fingers, n_vectors)
    contact_mag = np.abs(z).sum(axis=(1, 2))

    n = len(contact_mag)
    sorted_indices = np.argsort(contact_mag)

    # Quartiles by magnitude — pick spread across the active range.
    # Skip the bottom 50% (mostly zero) — we want frames where SOMETHING is happening.
    tiers = {
        "zero":         sorted_indices[: max(1, n // 100)],                # very lowest
        "light":        sorted_indices[n // 2:    n // 2 + max(1, n // 50)],   # 50th-ish percentile of activity
        "intermediate": sorted_indices[3 * n // 4:3 * n // 4 + max(1, n // 50)],
        "peak":         sorted_indices[-max(1, n // 100):],                # very highest
    }

    selected: list[tuple[int, str]] = []
    for label, candidate_indices in tiers.items():
        # Sample n_per_tier evenly spaced from the candidates
        if len(candidate_indices) == 0:
            continue
        step = max(1, len(candidate_indices) // n_per_tier)
        picks = candidate_indices[::step][:n_per_tier]
        for idx in picks:
            selected.append((int(idx), label))
    return selected


def annotate_image(img: np.ndarray, title: str, stats: str) -> Image.Image:
    """Add a title bar above the image with stats."""
    pil = Image.fromarray(img)
    title_height = 60
    canvas = Image.new("RGB", (pil.width, pil.height + title_height), (40, 40, 40))
    canvas.paste(pil, (0, title_height))
    draw = ImageDraw.Draw(canvas)

    # Try to find a usable font; fall back to default if not available
    font = None
    title_font = None
    for size, var_name in [(22, "title_font"), (14, "font")]:
        for candidate in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                          "/System/Library/Fonts/Helvetica.ttc",
                          "C:/Windows/Fonts/arial.ttf"]:
            try:
                loaded = ImageFont.truetype(candidate, size)
                if var_name == "title_font":
                    title_font = loaded
                else:
                    font = loaded
                break
            except OSError:
                continue
    if title_font is None: title_font = ImageFont.load_default()
    if font is None: font = ImageFont.load_default()

    draw.text((10, 6), title, fill=(255, 255, 255), font=title_font)
    draw.text((10, 36), stats, fill=(180, 180, 180), font=font)
    return canvas


def render_comparison(sample: np.ndarray, upscale: int,
                      frame_idx: int, tier_label: str) -> Image.Image:
    """Render old vs new encoding side by side with stats."""
    old_img = encode_old(sample, upscale=upscale)
    new_img = encode_new(sample, upscale=upscale)

    # Stats on the raw sample
    x = sample[..., 0].flatten()
    y = sample[..., 1].flatten()
    z = sample[..., 2].flatten()

    def axis_stats(name, vals):
        return f"{name}: min={vals.min():+5.1f} max={vals.max():+5.1f} active={int((np.abs(vals) > 0.5).sum())}/{len(vals)}"

    raw_stats = " | ".join([
        axis_stats("x", x),
        axis_stats("y", y),
        axis_stats("z", z),
    ])

    title_old = f"OLD encoding (linear ±100)  [frame {frame_idx}, tier: {tier_label}]"
    title_new = f"NEW encoding (linear xy ±{HAPTIC_XY_BOUND:.0f}, sqrt z [{HAPTIC_Z_MIN:.0f}, {HAPTIC_Z_MAX:.0f}])"

    old_canvas = annotate_image(old_img, title_old, raw_stats)
    new_canvas = annotate_image(new_img, title_new, raw_stats)

    # Stack vertically with a gap
    gap = 20
    combined = Image.new("RGB",
                         (max(old_canvas.width, new_canvas.width),
                          old_canvas.height + new_canvas.height + gap),
                         (20, 20, 20))
    combined.paste(old_canvas, (0, 0))
    combined.paste(new_canvas, (0, old_canvas.height + gap))
    return combined


def load_session_haptic(session_dir: Path) -> tuple[np.ndarray, int, int]:
    meta = json.loads((session_dir / "session_meta.json").read_text())
    haptic_info = meta.get("haptic")
    if haptic_info is None:
        raise RuntimeError(f"Session has no haptic data: {session_dir}")
    n_fingers = haptic_info["n_fingers"]
    n_vectors = haptic_info["n_vectors_per_finger"]
    sample_bytes = n_fingers * n_vectors * 3 * 4  # float32
    file_size = (session_dir / "haptic.bin").stat().st_size
    n_samples = file_size // sample_bytes
    mmap = np.memmap(session_dir / "haptic.bin", dtype="<f4", mode="r",
                     shape=(n_samples, n_fingers, n_vectors, 3))
    print(f"Loaded haptic: {n_samples} samples × {n_fingers} fingers × "
          f"{n_vectors} taxels × 3 axes")
    return mmap, n_fingers, n_vectors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path,
                    help="Path to session_YYYYMMDD_HHMMSS directory")
    ap.add_argument("--out", type=Path, default=Path("haptic_comparison"),
                    help="Output directory for comparison PNGs")
    ap.add_argument("--upscale", type=int, default=20,
                    help="Pixel upscale for visibility (default: 20)")
    ap.add_argument("--n-frames-per-tier", type=int, default=2,
                    help="How many frames per contact tier (default: 2)")
    args = ap.parse_args()

    if not args.session_dir.exists():
        print(f"Session not found: {args.session_dir}", file=sys.stderr)
        return 1

    haptic, _, _ = load_session_haptic(args.session_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    selections = select_representative_frames(haptic, n_per_tier=args.n_frames_per_tier)
    print(f"Selected {len(selections)} frames across tiers:")
    for idx, label in selections:
        print(f"  idx={idx:6d} tier={label}")

    for idx, label in selections:
        sample = np.asarray(haptic[idx])
        img = render_comparison(sample, args.upscale, idx, label)
        out_path = args.out / f"haptic_compare_{label}_idx{idx:06d}.png"
        img.save(out_path)
        print(f"  → {out_path}")

    print(f"\nDone. Open the PNGs in {args.out} to compare encodings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())