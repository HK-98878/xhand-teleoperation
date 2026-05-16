"""
Minimal haptic visualiser for haptic.bin output.

Loads a session and shows one finger at a time:
  - Timeline (top): per-sample total |force| for the current finger, with a
    vertical marker at the current sample
  - 2×2 heatmap grid: x, y, z, and magnitude — each with its own colour scale
    so the magnitudes don't dominate the signed channels. All rotated 90° CCW
    so the top of the grid matches the top of the fingertip. Cell labels show
    the raw flat index.

Keys:
  [  /  ]   — previous / next finger (auto-jumps to that finger's peak sample)
  p         — jump to the current finger's peak-force sample
  q         — quit
Slider scrubs through samples.

Usage:
    python view_haptic.py path/to/session_YYYYMMDD_HHMMSS [--finger 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


GRID_ROWS, GRID_COLS = 10, 12  # fixed; rotated for display so top maps to top


def load_session(session_dir: Path):
    """Return (haptic array of shape (T, n_fingers, n_vectors, 3), meta dict)."""
    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        sys.exit(f"No session_meta.json in {session_dir}")
    meta = json.loads(meta_path.read_text())

    if "haptic" not in meta or meta["haptic"].get("n_vectors_per_finger") is None:
        sys.exit("session_meta.json has no haptic shape — was the robot connected?")

    h = meta["haptic"]
    n_fingers = h["n_fingers"]
    n_vectors = h["n_vectors_per_finger"]

    bin_path = session_dir / "haptic.bin"
    if not bin_path.exists():
        sys.exit(f"No haptic.bin in {session_dir}")

    arr = np.fromfile(bin_path, dtype="<f4")
    floats_per_sample = n_fingers * n_vectors * 3
    if floats_per_sample == 0:
        sys.exit("Zero-size haptic shape in metadata")
    if arr.size % floats_per_sample != 0:
        print(f"Warning: haptic.bin size {arr.size} not divisible by "
              f"{floats_per_sample} floats/sample — truncating")
        arr = arr[: (arr.size // floats_per_sample) * floats_per_sample]

    n_samples = arr.size // floats_per_sample
    haptic = arr.reshape(n_samples, n_fingers, n_vectors, 3)
    print(f"Loaded {n_samples} samples × {n_fingers} fingers × {n_vectors} vectors × 3")
    return haptic, meta


def per_finger_peaks(haptic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (total_abs of shape (T, n_fingers), peak_sample of shape (n_fingers,))."""
    total_abs = np.abs(haptic).sum(axis=(2, 3))   # (T, n_fingers)
    peaks = np.argmax(total_abs, axis=0)          # (n_fingers,)
    return total_abs, peaks


def reshape_for_display(scalar: np.ndarray) -> np.ndarray:
    """(120,) → (12, 10), with 90° CCW rotation so the top of the grid matches
    the physical top of the fingertip. Pads with NaN if undersized."""
    target = GRID_ROWS * GRID_COLS
    n = scalar.size
    if n < target:
        padded = np.full(target, np.nan, dtype=scalar.dtype)
        padded[:n] = scalar
        grid = padded.reshape(GRID_ROWS, GRID_COLS)
    elif n > target:
        grid = scalar[:target].reshape(GRID_ROWS, GRID_COLS)
    else:
        grid = scalar.reshape(GRID_ROWS, GRID_COLS)
    return np.rot90(grid, k=1)  # 90° CCW


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path, help="Path to session_YYYYMMDD_HHMMSS dir")
    ap.add_argument("--finger", type=int, default=0, help="Initial finger 0..n-1")
    args = ap.parse_args()

    haptic, meta = load_session(args.session_dir)
    T, n_fingers, n_vectors, _ = haptic.shape

    if not 0 <= args.finger < n_fingers:
        sys.exit(f"--finger {args.finger} out of range 0..{n_fingers - 1}")

    total_abs_all, peaks = per_finger_peaks(haptic)
    for f in range(n_fingers):
        print(f"  finger {f}: peak |force| {total_abs_all[peaks[f], f]:7.2f} "
              f"at sample {peaks[f]}/{T - 1}")

    # ---- mutable view state -----------------------------------------------
    state = {
        "finger": args.finger,
        "sample": int(peaks[args.finger]),
    }

    # Per-finger colour ranges. Signed channels (x, y, z) get the same range
    # across xyz so cross-channel patterns are comparable; magnitude gets its
    # own range and a sequential colormap since it's non-negative.
    def ranges_for(finger: int) -> tuple[float, float]:
        v = haptic[:, finger]                          # (T, n_vectors, 3)
        signed_vmax = max(1e-6, float(np.abs(v).max()))
        mag_vmax = max(1e-6, float(np.linalg.norm(v, axis=-1).max()))
        return signed_vmax, mag_vmax

    def scalars_for(sample_idx: int, finger: int) -> dict:
        v = haptic[sample_idx, finger, :, :]
        return {
            "x": v[:, 0],
            "y": v[:, 1],
            "z": v[:, 2],
            "m": np.linalg.norm(v, axis=1),
        }

    # ---- figure layout ----------------------------------------------------
    # Top row: timeline. Bottom: 2×2 heatmap grid.
    fig = plt.figure(figsize=(11, 9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.25,
                             left=0.07, right=0.93, top=0.95, bottom=0.10)
    ax_time = fig.add_subplot(outer[0, 0])
    grid_gs = outer[1, 0].subgridspec(2, 2, hspace=0.3, wspace=0.3)
    ax_x = fig.add_subplot(grid_gs[0, 0])
    ax_y = fig.add_subplot(grid_gs[0, 1])
    ax_z = fig.add_subplot(grid_gs[1, 0])
    ax_m = fig.add_subplot(grid_gs[1, 1])

    # Timeline
    time_line, = ax_time.plot(total_abs_all[:, state["finger"]], lw=0.8)
    ax_time.set_xlabel("sample index")
    ax_time.set_ylabel("Σ |xyz|")
    time_marker = ax_time.axvline(state["sample"], color="r", lw=1)

    # Heatmaps
    scalars = scalars_for(state["sample"], state["finger"])
    signed_vmax, mag_vmax = ranges_for(state["finger"])

    panels = {}
    for key, ax, title, cmap, vmin, vmax in [
        ("x", ax_x, "X (shear)",  "RdBu_r", -signed_vmax, signed_vmax),
        ("y", ax_y, "Y (shear)",  "RdBu_r", -signed_vmax, signed_vmax),
        ("z", ax_z, "Z (normal)", "RdBu_r", -signed_vmax, signed_vmax),
        ("m", ax_m, "magnitude",  "viridis", 0.0,         mag_vmax),
    ]:
        img = ax.imshow(reshape_for_display(scalars[key]),
                        vmin=vmin, vmax=vmax, cmap=cmap,
                        interpolation="nearest", aspect="equal")
        cbar = fig.colorbar(img, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)

        # Cell labels — raw flat index, positioned post-rotation
        texts = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                flat = r * GRID_COLS + c
                if flat >= n_vectors:
                    continue
                disp_r, disp_c = GRID_COLS - 1 - c, r
                # White text on the dark viridis panel reads better than black
                txt_color = "white" if key == "m" else "black"
                t = ax.text(disp_c, disp_r, str(flat),
                            ha="center", va="center",
                            fontsize=5.5, color=txt_color, alpha=0.5)
                texts.append(t)
        ax.set_xticks([])
        ax.set_yticks([])
        panels[key] = {"img": img, "cbar": cbar, "texts": texts, "ax": ax}

    suptitle = fig.suptitle("")

    # Slider
    ax_slider = fig.add_axes([0.10, 0.04, 0.78, 0.02]) # type: ignore
    slider = Slider(ax_slider, "sample", 0, T - 1, valinit=state["sample"], valstep=1)

    # ---- refresh helpers --------------------------------------------------
    def title_str() -> str:
        return (f"Finger {state['finger']}/{n_fingers - 1}  |  "
                f"sample {state['sample']}  |  peak at {peaks[state['finger']]}")

    def refresh_for_finger():
        """Finger changed: rescale colour ranges, replot timeline, jump to peak."""
        f = state["finger"]
        state["sample"] = int(peaks[f])
        signed_vmax, mag_vmax = ranges_for(f)
        for key in ("x", "y", "z"):
            panels[key]["img"].set_clim(-signed_vmax, signed_vmax)
        panels["m"]["img"].set_clim(0.0, mag_vmax)
        time_line.set_ydata(total_abs_all[:, f])
        ax_time.relim()
        ax_time.autoscale_view(scalex=False, scaley=True)
        slider.set_val(state["sample"])  # triggers refresh_for_sample via on_slider

    def refresh_for_sample():
        """Sample changed: redraw all four heatmaps + marker + title."""
        scalars = scalars_for(state["sample"], state["finger"])
        for key in ("x", "y", "z", "m"):
            panels[key]["img"].set_data(reshape_for_display(scalars[key]))
        time_marker.set_xdata([state["sample"], state["sample"]])
        suptitle.set_text(title_str())
        fig.canvas.draw_idle()

    suptitle.set_text(title_str())

    # ---- event handlers ---------------------------------------------------
    def on_slider(val):
        state["sample"] = int(val)
        refresh_for_sample()

    def on_key(event):
        if event.key == "]":
            state["finger"] = (state["finger"] + 1) % n_fingers
            refresh_for_finger()
        elif event.key == "[":
            state["finger"] = (state["finger"] - 1) % n_fingers
            refresh_for_finger()
        elif event.key == "p":
            slider.set_val(int(peaks[state["finger"]]))
        elif event.key == "q":
            plt.close(fig)

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())