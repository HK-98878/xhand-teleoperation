"""
Export pipeline: align session data, transcode video, synthesise haptic RGB.

Reads a captured session (see capture.py) and produces an aligned export:

  <export_root>/<session_name>/
    cameras/
      rgb/
        rgb.mp4            H.264 transcode of source MJPEG, at target fps
        frames.jsonl       one record per output frame
        metadata.json      width, height, fps, codec, source info
      haptic/
        rgb.mp4            synthesised tactile image stream, same fps as rgb
        frames.jsonl       one record per output frame
        metadata.json      derivation params (value range, layout, separators)
    robot.jsonl            joint state + targets, one record per output tick
    haptic_vector.jsonl    raw 1800-float tactile per tick
    episode_events.jsonl   copy of session events.jsonl
    session_metadata.json  config snapshot + totals

Alignment strategy:
  - Camera frames drive the timeline. One output tick = one camera frame.
  - Robot streams (state, command, haptic) are sampled with latest-before
    semantics against each frame's host_timestamp_ns.
  - Camera video is encoded at --target-fps (default 28). Frame count is
    preserved exactly; playback duration may differ slightly from wall-clock,
    which is fine for training.

Usage:
    python export.py path/to/session_YYYYMMDD_HHMMSS [--target-fps 28]
                     [--out exports/] [--haptic-upscale 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export")


# ---------------------------------------------------------------------------
# Constants for haptic image synthesis
# ---------------------------------------------------------------------------

GRID_ROWS, GRID_COLS = 10, 12          # raw fingertip taxel grid
DISPLAY_ROWS, DISPLAY_COLS = 12, 10    # after 90° CCW rotation (matches visualiser)
SEPARATOR_PX = 2                       # black bar between fingers
SEPARATOR_COLOUR = (0, 0, 0)

# Per-channel encoding bounds for haptic → RGB conversion. Picked from
# observed force distributions across real grasp data (see analysis script):
#  - x,y (shear) are symmetric and very narrow: 99.9% under ±8, max ±23
#  - z (normal) is essentially one-sided positive: 99.9% under +21, max +53,
#    while negative z is bounded by sensor noise (>-1)
# These bounds maximise pixel resolution where the signal actually lives.
# Note: the encoding is informational for the model only — the raw values
# live in haptic.bin and haptic_vector.jsonl. Change these bounds freely
# but be aware all training data must be re-exported with the same choice.
HAPTIC_XY_BOUND = 30.0                 # symmetric ±30 for x and y (linear)
HAPTIC_Z_MIN = -1.0                    # asymmetric range for z (sqrt-compressed)
HAPTIC_Z_MAX = 50.0                    # see _encode_z_sqrt() below

# Per-channel deadband: values with |force| below the deadband encode to the
# neutral grey (128) regardless of their sign. Suppresses sensor noise and
# small per-taxel offsets that survive calibration, giving the model a clean
# "no contact" baseline. Below the deadband is treated as exactly zero; above
# it the normal scaling resumes. Picked to be slightly above the observed
# sensor noise floor (~±0.5) but well below real light-touch signal (~+2).
HAPTIC_XY_DEADBAND = 1
HAPTIC_Z_DEADBAND = 0.5


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass
class CameraSource:
    """One captured camera, located under <session>/cameras/<name>/."""
    name: str
    dir: Path
    video: Path
    frames: list[dict]
    metadata: dict             # contents of metadata.json written by capture.py


@dataclass
class Session:
    root: Path
    meta: dict
    cameras: list[CameraSource]
    events: list[dict]
    robot_state: list[dict]
    commands: list[dict]
    haptic_index: list[dict]
    haptic_mmap: Optional[np.memmap]   # shape (n_samples, n_fingers, n_vectors, 3) or None
    n_fingers: int
    n_vectors_per_finger: Optional[int]

    @classmethod
    def load(cls, session_dir: Path) -> "Session":
        meta_path = session_dir / "session_meta.json"
        if not meta_path.exists():
            sys.exit(f"No session_meta.json in {session_dir}")
        meta = json.loads(meta_path.read_text())

        cameras_dir = session_dir / "cameras"
        if not cameras_dir.is_dir():
            sys.exit(f"No cameras/ subdir in {session_dir}")
        cameras: list[CameraSource] = []
        for cam_dir in sorted(p for p in cameras_dir.iterdir() if p.is_dir()):
            frames_path = cam_dir / "frames.jsonl"
            video_candidates = list(cam_dir.glob("rgb.*"))
            if not video_candidates or not frames_path.exists():
                log.warning("Skipping %s — missing rgb.* or frames.jsonl", cam_dir.name)
                continue
            metadata_path = cam_dir / "metadata.json"
            cam_meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            cameras.append(CameraSource(
                name=cam_dir.name,
                dir=cam_dir,
                video=video_candidates[0],
                frames=load_jsonl(frames_path),
                metadata=cam_meta,
            ))
        if not cameras:
            sys.exit(f"No usable camera directories in {cameras_dir}")

        events = load_jsonl(session_dir / "events.jsonl")
        robot_state = load_jsonl(session_dir / "robot_state.jsonl")
        commands = load_jsonl(session_dir / "commands.jsonl")
        haptic_index = load_jsonl(session_dir / "haptic_index.jsonl")

        haptic_mmap = None
        n_fingers = 0
        n_vectors_per_finger = None
        haptic_meta = meta.get("haptic")
        if haptic_meta and haptic_meta.get("n_vectors_per_finger"):
            n_fingers = haptic_meta["n_fingers"]
            n_vectors_per_finger = haptic_meta["n_vectors_per_finger"]
            bin_path = session_dir / "haptic.bin"
            if bin_path.exists() and bin_path.stat().st_size > 0:
                floats_per_sample = n_fingers * n_vectors_per_finger * 3
                total_floats = bin_path.stat().st_size // 4  # float32
                n_samples = total_floats // floats_per_sample
                haptic_mmap = np.memmap(
                    bin_path, dtype="<f4", mode="r",
                    shape=(n_samples, n_fingers, n_vectors_per_finger, 3),
                )

        log.info("Loaded session: %d cameras (%s), %d events, %d state, %d cmd, %d haptic",
                 len(cameras), ", ".join(c.name for c in cameras),
                 len(events), len(robot_state), len(commands),
                 0 if haptic_mmap is None else haptic_mmap.shape[0])
        return cls(
            root=session_dir, meta=meta,
            cameras=cameras, events=events,
            robot_state=robot_state, commands=commands,
            haptic_index=haptic_index, haptic_mmap=haptic_mmap,
            n_fingers=n_fingers, n_vectors_per_finger=n_vectors_per_finger,
        )


# ---------------------------------------------------------------------------
# Latest-before alignment
# ---------------------------------------------------------------------------

class LatestBeforeIndex:
    """Given a sorted list of timestamps, returns the index of the latest
    entry with ts <= query. Returns None if no entry exists before the query."""

    def __init__(self, timestamps: list[int]):
        self.ts = timestamps

    def lookup(self, query_ts: int) -> Optional[int]:
        # bisect_right returns first index where ts > query; the latest-before is that - 1
        i = bisect_right(self.ts, query_ts) - 1
        if i < 0:
            return None
        return i


# ---------------------------------------------------------------------------
# Haptic image synthesis
# ---------------------------------------------------------------------------

def _encode_xy_linear(values: np.ndarray) -> np.ndarray:
    """Linear encoding for x and y shear channels with symmetric bound.
    Maps [-HAPTIC_XY_BOUND, +HAPTIC_XY_BOUND] → [0, 255], centred at 128.
    Values below HAPTIC_XY_DEADBAND in magnitude are mapped to neutral 128
    to suppress sensor noise. Outside the range, clipped."""
    scaled = (values + HAPTIC_XY_BOUND) * (255.0 / (2 * HAPTIC_XY_BOUND))
    deadband = np.abs(values) < HAPTIC_XY_DEADBAND
    return np.where(deadband, 128.0, np.clip(scaled, 0, 255))


def _encode_z_sqrt(values: np.ndarray) -> np.ndarray:
    """Square-root encoding for z normal channel, asymmetric range.

    The contact distribution is heavily skewed: 99% of taxels read zero,
    of the active ones most are small (z in [0, +10]), with rare strong
    contacts up to ~+50. Linear encoding wastes most of the 8-bit range
    on values that never occur. Signed sqrt compresses the high end and
    expands the low end where the signal lives.

    Encoding around z=0 (the typical "no contact" baseline):
        |z| < HAPTIC_Z_DEADBAND → 128 (neutral, suppress sensor noise)
        z=1   → 145 (single taxel-unit of contact, clearly visible)
        z=5   → 168 (light contact)
        z=10  → 184 (firm contact)
        z=25  → 217 (very firm)
        z=50  → 255 (saturation, ~0.01% of taxels per measured stats)

    Negative z is rare (sensor noise around the calibrated zero, > -1 in
    practice) but the same sqrt curve applies symmetrically.
    """
    # Normalise to [-1, +1] within each side of the asymmetric bound,
    # apply sqrt magnitude, multiply by 127 for half-range, offset to 128.
    pos_norm = np.clip(np.where(values >= 0, values / HAPTIC_Z_MAX, 0), 0, 1)
    neg_norm = np.clip(np.where(values < 0, values / HAPTIC_Z_MIN, 0), 0, 1)
    # sqrt expands low-magnitude values; sign preserved by which mask fired.
    magnitude = np.sqrt(pos_norm) - np.sqrt(neg_norm)
    scaled = np.clip(128.0 + magnitude * 127.0, 0, 255)
    # Apply deadband: small noise/offset values encode to neutral grey.
    deadband = np.abs(values) < HAPTIC_Z_DEADBAND
    return np.where(deadband, 128.0, scaled)


def haptic_sample_to_image(sample: np.ndarray, upscale: int = 1) -> np.ndarray:
    """Convert a single haptic sample (n_fingers, n_vectors, 3) to a uint8
    RGB image.

    Channel encoding (informed by measured force distributions):
      - x → R, linear, symmetric ±HAPTIC_XY_BOUND
      - y → G, linear, symmetric ±HAPTIC_XY_BOUND
      - z → B, signed sqrt, asymmetric [HAPTIC_Z_MIN, HAPTIC_Z_MAX]

    Layout: 5 fingers side by side, each rotated 90° CCW (top of finger = top
    of image), with a 2px black separator between fingers.

    Returned image shape: (DISPLAY_ROWS, n_fingers*DISPLAY_COLS + (n_fingers-1)*SEPARATOR_PX, 3).
    If upscale > 1, nearest-neighbour upscale by that factor for codec friendliness.
    """
    n_fingers = sample.shape[0]
    finger_imgs = []
    for f in range(n_fingers):
        v = sample[f]  # (n_vectors, 3)
        # Pad if undersized so we always get a clean 10x12 reshape.
        n = v.shape[0]
        target = GRID_ROWS * GRID_COLS
        if n < target:
            padded = np.zeros((target, 3), dtype=v.dtype)
            padded[:n] = v
            v = padded
        elif n > target:
            v = v[:target]

        grid = v.reshape(GRID_ROWS, GRID_COLS, 3)
        rotated = np.rot90(grid, k=1)  # 90° CCW → (DISPLAY_ROWS, DISPLAY_COLS, 3)

        # Encode each channel with its own scheme.
        encoded = np.empty_like(rotated, dtype=np.float32)
        encoded[..., 0] = _encode_xy_linear(rotated[..., 0])  # x → R
        encoded[..., 1] = _encode_xy_linear(rotated[..., 1])  # y → G
        encoded[..., 2] = _encode_z_sqrt(rotated[..., 2])     # z → B
        finger_imgs.append(encoded.astype(np.uint8))

    # Concatenate with separators
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



# ---------------------------------------------------------------------------
# Video I/O via ffmpeg pipes
# ---------------------------------------------------------------------------

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def measure_actual_fps(frames: list[dict]) -> float:
    """Compute the camera's actual delivered fps from host timestamps in
    frames.jsonl. Returns NaN if fewer than 2 frames."""
    if len(frames) < 2:
        return float("nan")
    duration_s = (frames[-1]["host_timestamp_ns"]
                  - frames[0]["host_timestamp_ns"]) / 1e9
    if duration_s <= 0:
        return float("nan")
    return (len(frames) - 1) / duration_s


def transcode_rgb_video(src: Path, dst: Path, output_fps: float) -> None:
    """Transcode session rgb.mkv (MJPEG-in-Matroska) → H.264 mp4.

    Every input frame is preserved 1:1 in the output. We use setpts to assign
    output PTS that produce the desired playback fps (decoupling output
    timing from the source's tagged framerate, which capture.py wrote as a
    fixed value that may not match actual delivery), and fps_mode=vfr so
    ffmpeg doesn't pad or drop frames trying to hit a constant rate.

    output_fps should be the camera's actual measured fps, so that playback
    duration matches wall-clock capture duration.

    Pixel format is yuvj420p (full-range luma, 0-255). The camera captures
    full range; using full range in the export preserves every value and
    ensures the training-time data matches what the model will see at
    inference if both decode paths are consistent.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(src),
        "-vf", f"setpts=N/{output_fps}/TB",
        "-fps_mode", "vfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuvj420p",
        "-color_range", "pc",
        str(dst),
    ]
    log.info("Transcoding %s → %s @ %.3f fps (every frame preserved)",
             src.name, dst.name, output_fps)
    subprocess.run(cmd, check=True)


def write_haptic_video(frames_iter, width: int, height: int,
                       fps: float, dst: Path) -> int:
    """Pipe raw uint8 RGB frames to ffmpeg → H.264 mp4 with yuv444p
    (no chroma subsampling — critical for tiny image where 2×2 chroma blocks
    would smear adjacent taxel values) AND lossless encoding (qp=0).

    Lossless because the image is small and each pixel carries information —
    even mild lossy compression smooths the taxel-scale spikes that are the
    whole point of the data. Lossless H.264 at this resolution costs ~tens of
    KB per second, so size is not a concern.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-qp", "0",              # lossless — preserves every taxel value exactly
        "-pix_fmt", "yuv444p",
        "-r", f"{fps}",
        str(dst),
    ]
    log.info("Writing haptic video %s (%dx%d @ %.3f fps, lossless)",
             dst.name, width, height, fps)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n_written = 0
    try:
        for frame in frames_iter:
            assert frame.shape == (height, width, 3), \
                f"Frame shape mismatch: {frame.shape} vs ({height}, {width}, 3)"
            assert frame.dtype == np.uint8
            proc.stdin.write(frame.tobytes())
            n_written += 1
        proc.stdin.close()
    except BrokenPipeError:
        log.error("ffmpeg closed pipe early — likely a config error")
    err = proc.stderr.read().decode(errors="replace")
    proc.wait()
    if proc.returncode != 0:
        log.error("ffmpeg failed:\n%s", err)
        raise RuntimeError(f"ffmpeg exit {proc.returncode}")
    return n_written


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def compute_rezero_offset(
    session: "SessionData",
    episode_index: int,
    window_s: float,
) -> Optional[np.ndarray]:
    """Compute per-taxel-per-axis haptic offset from the first `window_s`
    seconds after the Nth episode_start event. Returns an array of shape
    (n_fingers, n_vectors_per_finger, 3) to be SUBTRACTED from every haptic
    sample, or None if the window can't be resolved.

    The window is chosen on the assumption that the user has placed the hand
    in a known no-contact pose during the early seconds of the named episode
    — same convention as the original sensor zeroing. No automatic validation
    of that assumption is performed; the user verifies visually."""
    if session.haptic_mmap is None:
        log.error("--rezero-from-episode requested but session has no haptic data")
        return None

    starts = [e for e in session.events
              if (e.get("event") or "").lower() == "episode_start"
              and e.get("robot_timestamp_ns") is not None]
    if episode_index < 0 or episode_index >= len(starts):
        log.error("--rezero-from-episode %d out of range (session has %d episode_start events)",
                  episode_index, len(starts))
        return None

    start_ts = int(starts[episode_index]["robot_timestamp_ns"])
    end_ts = start_ts + int(window_s * 1e9)

    # Find haptic samples in [start_ts, end_ts]. haptic_index entries are
    # timestamp-ordered; do a linear scan since the window is short.
    sample_indices: list[int] = []
    for i, idx_entry in enumerate(session.haptic_index):
        ts = idx_entry.get("robot_timestamp_ns")
        if ts is None:
            continue
        ts = int(ts)
        if ts < start_ts:
            continue
        if ts > end_ts:
            break
        sample_indices.append(i)

    if not sample_indices:
        log.error("--rezero-from-episode found no haptic samples in [%d, %d] "
                  "(episode %d, window %.2fs)",
                  start_ts, end_ts, episode_index, window_s)
        return None

    # Stack the selected samples and mean over time → per-taxel-per-axis offset.
    samples = np.stack([np.asarray(session.haptic_mmap[i]) for i in sample_indices],
                       axis=0)
    offset = samples.mean(axis=0).astype(np.float32)

    log.info("Computed rezero offset from %d haptic samples in episode %d's first %.2fs",
             len(sample_indices), episode_index, window_s)
    log.info("  offset stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
             float(offset.mean()), float(offset.std()),
             float(offset.min()), float(offset.max()))
    return offset


def export(session_dir: Path, out_root: Path, target_fps: float,
           haptic_upscale: int, primary_camera_name: Optional[str],
           rezero_from_episode: Optional[int] = None,
           rezero_window_s: float = 3.0,
           zero_non_z: bool = False) -> int:
    if not _have_ffmpeg():
        log.error("ffmpeg not found. Install with: sudo apt install ffmpeg")
        return 1
    if haptic_upscale < 1:
        log.error("--haptic-upscale must be >= 1")
        return 1

    session = Session.load(session_dir)

    # --- Optional re-zero -------------------------------------------------
    # Computes a per-taxel offset from the first N seconds of the chosen
    # episode and subtracts it from every haptic sample read downstream
    # (both haptic_vector.jsonl and haptic.mp4). Raw haptic.bin is untouched.
    rezero_offset: Optional[np.ndarray] = None
    if rezero_from_episode is not None:
        rezero_offset = compute_rezero_offset(session, rezero_from_episode, rezero_window_s)
        if rezero_offset is None:
            return 1

    def read_haptic(hi: int) -> np.ndarray:
        """Read a haptic sample, applying re-zero offset if configured.
        Always returns a float32 array of shape (n_fingers, n_vectors, 3)."""
        sample = np.asarray(session.haptic_mmap[hi]).astype(np.float32)
        if rezero_offset is not None:
            sample[:,:,2] -= rezero_offset[:,:,2] # Z only
        if zero_non_z:
            sample[:,:,0:2] -= sample[:,:,0:2]
        return sample

    # Pick primary camera. This camera's frame timestamps drive the
    # robot.jsonl / haptic_vector.jsonl / haptic-image tick timeline.
    # Other cameras are transcoded with their own frames.jsonl but don't
    # influence robot/haptic alignment.
    if primary_camera_name is None:
        primary = session.cameras[0]
        log.info("No --primary-camera given; using first available: %s", primary.name)
    else:
        primary = next((c for c in session.cameras if c.name == primary_camera_name), None)
        if primary is None:
            log.error("Primary camera '%s' not found. Available: %s",
                      primary_camera_name,
                      ", ".join(c.name for c in session.cameras))
            return 1

    out_dir = out_root / session_dir.name
    cameras_out_dir = out_dir / "cameras"
    cameras_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build alignment indices ------------------------------------------
    state_ts = [r["robot_timestamp_ns"] for r in session.robot_state]
    cmd_ts = [r["robot_timestamp_ns"] for r in session.commands]
    haptic_ts = [r["robot_timestamp_ns"] for r in session.haptic_index]
    state_idx = LatestBeforeIndex(state_ts)
    cmd_idx = LatestBeforeIndex(cmd_ts)
    haptic_idx = LatestBeforeIndex(haptic_ts)

    # --- Robot/haptic records aligned to PRIMARY camera ------------------
    primary_frames = primary.frames
    n_primary = len(primary_frames)
    if n_primary == 0:
        log.error("Primary camera %s has no frames", primary.name)
        return 1
    log.info("Aligning robot/haptic to primary camera '%s' (%d frames; "
             "actual fps determined per-camera, see warnings if any)",
             primary.name, n_primary)

    robot_records = []
    haptic_vector_records = []
    missing_state = 0
    missing_cmd = 0
    missing_haptic = 0

    for out_idx, frame in enumerate(primary_frames):
        host_ts = frame["host_timestamp_ns"]
        si = state_idx.lookup(host_ts)
        ci = cmd_idx.lookup(host_ts)
        hi = haptic_idx.lookup(host_ts)
        if si is None: missing_state += 1
        if ci is None: missing_cmd += 1
        if hi is None: missing_haptic += 1

        state = session.robot_state[si] if si is not None else None
        cmd = session.commands[ci] if ci is not None else None
        robot_records.append({
            "frame_index": out_idx,
            "host_timestamp_ns": host_ts,
            "joint_state_position": state["position"] if state else None,
            "joint_state_effort": state["effort"] if state else None,
            "joint_state_temperature": state["temperature"] if state else None,
            "joint_target_position": cmd["position"] if cmd else None,
        })

        if hi is not None and session.haptic_mmap is not None:
            raw = read_haptic(hi).reshape(-1).tolist()
        else:
            raw = None
        haptic_vector_records.append({
            "frame_index": out_idx,
            "host_timestamp_ns": host_ts,
            "raw_force": raw,
        })

    if missing_state or missing_cmd or missing_haptic:
        log.warning("Frames without latest-before data on primary: state=%d cmd=%d haptic=%d",
                    missing_state, missing_cmd, missing_haptic)

    with open(out_dir / "robot.jsonl", "w") as f:
        for r in robot_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "haptic_vector.jsonl", "w") as f:
        for r in haptic_vector_records:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote robot.jsonl + haptic_vector.jsonl (%d records each, keyed to primary)",
             n_primary)

    # --- Transcode every camera ------------------------------------------
    # Each camera is encoded at its own measured fps so every captured frame
    # is preserved 1:1. This is essential for keeping rgb.mp4 and haptic.mp4
    # frame-aligned downstream — any frame-drop logic in ffmpeg would
    # decorrelate them.
    per_camera_summary = []
    primary_output_fps: Optional[float] = None  # set when we hit the primary
    for cam in session.cameras:
        cam_out_dir = cameras_out_dir / cam.name
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        rgb_out = cam_out_dir / "rgb.mp4"

        measured_fps = measure_actual_fps(cam.frames)
        if not (measured_fps == measured_fps):  # NaN guard
            log.error("Camera %s has unmeasurable fps (too few frames); skipping",
                      cam.name)
            continue

        # Warn if user supplied --target-fps and it disagrees with reality.
        # We never honour --target-fps — we always encode at measured — but a
        # mismatch tells you something is off about your assumptions.
        if abs(measured_fps - target_fps) > 0.5:
            log.warning("Camera %s actually delivered %.3f fps but --target-fps is %.1f. "
                        "Ignoring --target-fps and encoding at measured rate "
                        "(needed to keep rgb.mp4 and haptic.mp4 frame-aligned).",
                        cam.name, measured_fps, target_fps)

        transcode_rgb_video(cam.video, rgb_out, measured_fps)

        if cam.name == primary.name:
            primary_output_fps = measured_fps

        # frames.jsonl mirrors source, but pointing at rgb.mp4 instead of source ext
        with open(cam_out_dir / "frames.jsonl", "w") as f:
            for i, frame in enumerate(cam.frames):
                rec = {
                    "camera": cam.name,
                    "depth_file": None,
                    "frame_index": i,
                    "height": frame["height"],
                    "host_timestamp_ns": frame["host_timestamp_ns"],
                    "rgb_video": "rgb.mp4",
                    "rgb_video_frame": i,
                    "width": frame["width"],
                }
                f.write(json.dumps(rec) + "\n")

        n_frames_cam = len(cam.frames)
        cam_metadata = {
            "name": cam.name,
            "is_primary": cam.name == primary.name,
            "width": cam.frames[0]["width"] if cam.frames else None,
            "height": cam.frames[0]["height"] if cam.frames else None,
            "fps": measured_fps,
            "source_fps_measured": measured_fps,
            "codec": "h264",
            "pix_fmt": "yuvj420p",
            "color_range": "pc",
            "n_frames": n_frames_cam,
            "source_video": cam.video.name,
            "capture_metadata": cam.metadata,  # snapshot of capture-time metadata.json
        }
        (cam_out_dir / "metadata.json").write_text(json.dumps(cam_metadata, indent=2))
        per_camera_summary.append({
            "name": cam.name,
            "is_primary": cam.name == primary.name,
            "n_frames": n_frames_cam,
            "source_fps_measured": measured_fps,
        })
        log.info("  exported %s: %d frames, measured %.3f fps",
                 cam.name, n_frames_cam, measured_fps)

    # --- Haptic video synthesis (aligned to primary camera) --------------
    haptic_out_dir = cameras_out_dir / "haptic"
    if session.haptic_mmap is None:
        log.warning("No haptic data recorded — skipping haptic video synthesis")
        haptic_written = 0
    elif primary_output_fps is None:
        log.error("Primary camera was not transcoded — cannot synthesise haptic video")
        haptic_written = 0
    else:
        haptic_out_dir.mkdir(parents=True, exist_ok=True)
        n_fingers = session.n_fingers
        base_w = n_fingers * DISPLAY_COLS + (n_fingers - 1) * SEPARATOR_PX
        base_h = DISPLAY_ROWS
        out_w = base_w * haptic_upscale
        out_h = base_h * haptic_upscale

        def frame_iter():
            for frame in primary_frames:
                hi = haptic_idx.lookup(frame["host_timestamp_ns"])
                if hi is None:
                    yield np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    continue
                sample = read_haptic(hi)
                yield haptic_sample_to_image(sample, upscale=haptic_upscale)

        haptic_video_path = haptic_out_dir / "rgb.mp4"
        haptic_written = write_haptic_video(
            frame_iter(), out_w, out_h, primary_output_fps, haptic_video_path,
        )
        log.info("Wrote %d haptic frames (aligned to primary camera %s @ %.3f fps)",
                 haptic_written, primary.name, primary_output_fps)

        # frames.jsonl mirrors primary camera's timestamps
        with open(haptic_out_dir / "frames.jsonl", "w") as f:
            for i, frame in enumerate(primary_frames):
                rec = {
                    "camera": "haptic_rgb",
                    "depth_file": None,
                    "frame_index": i,
                    "height": out_h,
                    "host_timestamp_ns": frame["host_timestamp_ns"],
                    "rgb_video": "rgb.mp4",
                    "rgb_video_frame": i,
                    "width": out_w,
                }
                f.write(json.dumps(rec) + "\n")

        haptic_metadata = {
            "name": "haptic_rgb",
            "aligned_to_primary_camera": primary.name,
            "width": out_w,
            "height": out_h,
            "fps": primary_output_fps,
            "codec": "h264",
            "pix_fmt": "yuv444p",
            "encoding": "lossless (qp=0)",
            "n_frames": haptic_written,
            "derivation": {
                "source": "haptic.bin (raw FingerSensorState.raw_force)",
                "rezero": (
                    None if rezero_offset is None else {
                        "applied": True,
                        "from_episode_index": rezero_from_episode,
                        "window_s": rezero_window_s,
                        "offset_stats": {
                            "mean": float(rezero_offset.mean()),
                            "std": float(rezero_offset.std()),
                            "min": float(rezero_offset.min()),
                            "max": float(rezero_offset.max()),
                        },
                    }
                ),
                "n_fingers": n_fingers,
                "n_vectors_per_finger": session.n_vectors_per_finger,
                "channel_order": "x→R, y→G, z→B",
                "xy_encoding": "linear, symmetric, with deadband",
                "xy_input_bound": [-HAPTIC_XY_BOUND, HAPTIC_XY_BOUND],
                "xy_deadband": HAPTIC_XY_DEADBAND,
                "xy_mapping": f"(v + {HAPTIC_XY_BOUND}) * 255 / {2 * HAPTIC_XY_BOUND}, clipped; |v| < {HAPTIC_XY_DEADBAND} → 128",
                "z_encoding": "signed sqrt, asymmetric, with deadband",
                "z_input_range": [HAPTIC_Z_MIN, HAPTIC_Z_MAX],
                "z_deadband": HAPTIC_Z_DEADBAND,
                "z_mapping": (
                    "128 + sign(z) * sqrt(|z| / Z_bound) * 127  "
                    f"(positive Z_bound={HAPTIC_Z_MAX}, negative Z_bound={abs(HAPTIC_Z_MIN)}); "
                    f"|z| < {HAPTIC_Z_DEADBAND} → 128"
                ),
                "value_range_output": [0, 255],
                "finger_layout": "side by side, left to right",
                "rotation": "90deg CCW per finger (top of grid = top of fingertip)",
                "separator_px": SEPARATOR_PX,
                "upscale": haptic_upscale,
            },
        }
        (haptic_out_dir / "metadata.json").write_text(json.dumps(haptic_metadata, indent=2))

    # --- Episode events + session metadata --------------------------------
    shutil.copyfile(session.root / "events.jsonl",
                    out_dir / "episode_events.jsonl")

    joint_name = (session.robot_state[0]["joint_name"]
                  if session.robot_state and "joint_name" in session.robot_state[0]
                  else None)

    session_metadata = {
        "source_session": str(session.root.name),
        "target_fps_arg": target_fps,
        "primary_camera": primary.name,
        "primary_output_fps": primary_output_fps,
        "cameras": per_camera_summary,
        "n_robot_state_samples": len(session.robot_state),
        "n_command_samples": len(session.commands),
        "n_haptic_samples": (0 if session.haptic_mmap is None
                             else int(session.haptic_mmap.shape[0])),
        "missing_state_frames_primary": missing_state,
        "missing_cmd_frames_primary": missing_cmd,
        "missing_haptic_frames_primary": missing_haptic,
        "joint_name": joint_name,
        "n_joints": session.meta.get("config", {}).get("robot", {}).get("n_joints"),
        "n_fingers": session.n_fingers,
        "haptic_n_vectors_per_finger": session.n_vectors_per_finger,
        "alignment": "latest-before; one robot/haptic tick per primary-camera frame; "
                     "other cameras transcoded with own timestamps",
        "session_meta_snapshot": session.meta,
    }
    (out_dir / "session_metadata.json").write_text(json.dumps(session_metadata, indent=2))

    log.info("Export complete → %s", out_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path,
                    help="Path to session_YYYYMMDD_HHMMSS dir")
    ap.add_argument("--out", type=Path, default=Path("exports"),
                    help="Export root (default: ./exports/)")
    ap.add_argument("--target-fps", type=float, default=28.0,
                    help="Expected output fps. Used only as a sanity-check reference: "
                         "if a camera's actual measured fps differs by more than 0.5, "
                         "you'll see a warning. The output is ALWAYS encoded at the "
                         "camera's actual measured rate to preserve every captured frame "
                         "and keep rgb/haptic videos frame-aligned. (default: 28)")
    ap.add_argument("--haptic-upscale", type=int, default=1,
                    help="Nearest-neighbour upscale for haptic video (default: 1)")
    ap.add_argument("--primary-camera", type=str, default=None,
                    help="Name of camera whose frame timeline drives robot/haptic "
                         "alignment. Defaults to first camera in directory order.")
    ap.add_argument("--rezero-from-episode", type=int, default=None,
                    help="Re-zero the haptic sensors session-wide using the first "
                         "--rezero-window-s seconds after this episode_start event. "
                         "Subtracts a per-taxel offset from every haptic sample in "
                         "the exported haptic_vector.jsonl and haptic.mp4. The raw "
                         "haptic.bin is untouched. Use when the original sensor "
                         "zero (via the z key during capture) was inadequate. "
                         "User must visually verify the chosen window is contact-free.")
    ap.add_argument("--rezero-window-s", type=float, default=3.0,
                    help="Duration of the calibration window after the chosen "
                         "episode_start (default: 3.0 seconds).")
    ap.add_argument("--zero-non-z", action="store_true",
                    help="Disable xy haptic readings - useful for erroneous non-zeroed "
                    "data which is hard to correct with averaging, since these readings "
                    "are often less relevant")
    args = ap.parse_args()

    if not args.session_dir.is_dir():
        log.error("Not a directory: %s", args.session_dir)
        return 1

    return export(args.session_dir, args.out, args.target_fps,
                  args.haptic_upscale, args.primary_camera,
                  rezero_from_episode=args.rezero_from_episode,
                  rezero_window_s=args.rezero_window_s,
                  zero_non_z=args.zero_non_z)


if __name__ == "__main__":
    sys.exit(main())