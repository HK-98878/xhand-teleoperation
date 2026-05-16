"""
Threaded WiLoR-mini teleop scaffold with one-euro filtering and
dex-retargeting onto an XHand left hand.

Architecture:
  [camera thread]    -> latest_frame (single-slot, last-write-wins)
  [inference thread] -> reads latest_frame, runs WiLoR, writes latest_pose
  [command thread]   -> reads latest_pose at fixed rate, applies one-euro
                        filtering, rotates MANO->XHand frame, retargets
                        to robot joint angles, writes filtered_pose
  [main thread]      -> snapshots filtered_pose, draws overlay+HUD, UI

Inference runs at ~12-15 Hz (GPU-bound); commands tick at 50 Hz with
smoothly filtered poses and freshly-retargeted qpos. Visualisation uses
WiLoR's own 2D keypoints.

Press q to quit and dump final stats. Press s mid-run for snapshot.
"""

import contextlib
import datetime
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from dex_retargeting.retargeting_config import RetargetingConfig


# --- configuration ---------------------------------------------------------

# Adjust these for your machine.
XHAND_CONFIG = "./urdf/xhand_left_vector.yml"
DEX_URDF_DIR = "./urdf"

# Frame rotation, MANO -> XHand URDF.
# MANO: fingers along +x, palm normal +z, "across-palm" along +y.
# XHand URDF: fingers along -z, thumb-side along +x, palm normal along +y.
R_MANO_TO_XHAND = np.array([
    [ 1,  0,  0],   # XHand x = MANO x
    [ 0,  0,  1],   # XHand y = MANO z
    [ 0,  1,  0],   # XHand z = MANO y
], dtype=np.float64)


# --- topology --------------------------------------------------------------

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]

FINGER_COLOURS_BGR = {
    "thumb":  (28, 26, 228),
    "index":  (184, 126, 55),
    "middle": (74, 175, 77),
    "ring":   (163, 78, 152),
    "pinky":  (0, 127, 255),
}


def bone_colour_bgr(a: int, b: int) -> tuple:
    j = max(a, b)
    if j <= 4:  return FINGER_COLOURS_BGR["thumb"]
    if j <= 8:  return FINGER_COLOURS_BGR["index"]
    if j <= 12: return FINGER_COLOURS_BGR["middle"]
    if j <= 16: return FINGER_COLOURS_BGR["ring"]
    return FINGER_COLOURS_BGR["pinky"]


# --- output suppression ----------------------------------------------------

@contextlib.contextmanager
def suppress_output():
    """OS-level redirect of stdout/stderr to /dev/null. Catches even
    C-extension prints. Used to silence WiLoR-mini's per-frame logging."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


# --- timing infrastructure -------------------------------------------------

@dataclass
class StageStats:
    """Rolling-window timing buffer for one pipeline stage. Times in ms."""
    name: str
    window: int = 300
    samples: deque = field(default_factory=lambda: deque(maxlen=300))

    def __post_init__(self):
        self.samples = deque(maxlen=self.window)

    def add(self, ms: float):
        self.samples.append(ms)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"n": 0}
        a = np.fromiter(self.samples, dtype=np.float64)
        return {
            "n":    int(a.size),
            "mean": float(a.mean()),
            "p50":  float(np.percentile(a, 50)),
            "p95":  float(np.percentile(a, 95)),
            "p99":  float(np.percentile(a, 99)),
            "max":  float(a.max()),
            "std":  float(a.std()),
        }


class Timer:
    """Context manager for stage timing. cuda_sync forces GPU completion
    before stopping the clock; necessary for accurate inference timing."""
    def __init__(self, stats: StageStats, cuda_sync: bool = False):
        self.stats = stats
        self.cuda_sync = cuda_sync and torch.cuda.is_available()

    def __enter__(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.stats.add((time.perf_counter() - self.t0) * 1000.0)
        return False


def print_stats(stages: Dict[str, StageStats], header: str = ""):
    if header:
        print(f"\n=== {header} ===")
    print(f"{'stage':<14} {'n':>5} {'mean':>7} {'p50':>7} {'p95':>7} "
          f"{'p99':>7} {'max':>7} {'std':>6}")
    print("-" * 66)
    for name, s in stages.items():
        d = s.summary()
        if d["n"] == 0:
            print(f"{name:<14} {0:>5}")
            continue
        print(f"{name:<14} {d['n']:>5} {d['mean']:>7.2f} {d['p50']:>7.2f} "
              f"{d['p95']:>7.2f} {d['p99']:>7.2f} {d['max']:>7.2f} {d['std']:>6.2f}")
    print()


# --- one-euro filter -------------------------------------------------------

class OneEuroScalar:
    """One-euro filter for a single scalar value.

    Reference: Casiez, Roussel, Vogel (2012), "1€ Filter".

    Parameters:
      min_cutoff: cutoff frequency at zero velocity. Lower = smoother but
                  more lag at rest. Sensible range 0.3-2.0 Hz.
      beta:       how aggressively cutoff increases with speed. Higher =
                  less lag during fast motion but more jitter passes through.
                  Sensible range 0.001-0.1.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroVector:
    """Per-element one-euro filter for an arbitrary-shape array. Applies
    the same parameters to every element. Shape is fixed on first call."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self._params = (min_cutoff, beta, d_cutoff)
        self._filters: Optional[np.ndarray] = None
        self._shape = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._filters is None:
            self._shape = x.shape
            self._filters = np.empty(x.size, dtype=object)
            for i in range(x.size):
                self._filters[i] = OneEuroScalar(*self._params)
        flat_in = x.reshape(-1)
        flat_out = np.empty_like(flat_in)
        for i, val in enumerate(flat_in):
            flat_out[i] = self._filters[i](float(val), t)
        return flat_out.reshape(self._shape)


# --- shared state ----------------------------------------------------------

@dataclass
class LatestFrame:
    """Most recent camera frame. Single-slot, last-write-wins."""
    rgb: Optional[np.ndarray] = None
    bgr: Optional[np.ndarray] = None
    capture_ts: float = 0.0
    seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LatestPose:
    """Most recent inference output (raw or filtered). Single-slot.

    Used for both the inference thread's raw output and the command
    thread's filtered output. The filtered version additionally carries
    retargeted robot qpos.
    """
    joints_3d: Optional[np.ndarray] = None    # (21, 3) MANO local frame
    kpts_2d: Optional[np.ndarray] = None      # (21, 2) image pixels
    is_right: bool = True
    capture_ts: float = 0.0    # frame capture time (perf_counter)
    inference_ts: float = 0.0  # inference completion time (perf_counter)
    seq: int = 0
    qpos: Optional[np.ndarray] = None    # (12,) XHand joint angles, rad
    qpos_clipped: bool = False           # True if any joint hit a limit
    lock: threading.Lock = field(default_factory=threading.Lock)


# --- threads ---------------------------------------------------------------

def camera_thread(latest_frame: LatestFrame, stop_event: threading.Event,
                  stats: StageStats):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[camera] failed to open camera")
        stop_event.set()
        return
    seq = 0
    try:
        while not stop_event.is_set():
            with Timer(stats):
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.001)
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts = time.perf_counter()
                with latest_frame.lock:
                    latest_frame.rgb = frame_rgb
                    latest_frame.bgr = frame_bgr
                    latest_frame.capture_ts = ts
                    latest_frame.seq = seq
                seq += 1
    finally:
        cap.release()


def inference_thread(pipe, latest_frame: LatestFrame, latest_pose: LatestPose,
                     stop_event: threading.Event, stats: StageStats,
                     idle_stats: StageStats):
    last_seq = -1
    while not stop_event.is_set():
        with latest_frame.lock:
            seq = latest_frame.seq
            if seq == last_seq or latest_frame.rgb is None:
                rgb = None
            else:
                rgb = latest_frame.rgb
                capture_ts = latest_frame.capture_ts
        if rgb is None:
            with Timer(idle_stats):
                time.sleep(0.001)
            continue
        last_seq = seq

        with Timer(stats, cuda_sync=True):
            with suppress_output():
                outputs = pipe.predict(rgb)

        if len(outputs) == 0:
            stats.samples.pop()
            continue

        out = outputs[0]
        preds = out["wilor_preds"]
        joints_3d = np.asarray(preds["pred_keypoints_3d"][0]).copy()
        kpts_2d = (np.asarray(preds["pred_keypoints_2d"][0]).copy()
                   if "pred_keypoints_2d" in preds else None)
        is_right = bool(out.get("is_right", 1))

        # Mirror left hands so downstream sees right-hand MANO convention.
        # if not is_right:
        #     joints_3d[:, 0] *= -1.0

        if (np.isnan(joints_3d).any() or np.isinf(joints_3d).any()
                or (kpts_2d is not None
                    and (np.isnan(kpts_2d).any() or np.isinf(kpts_2d).any()))):
            continue

        # In inference_thread, after the handedness mirror:
        if seq < 5:  # only first few frames
            fingertips_relative_to_wrist = joints_3d[[4, 8, 12, 16, 20]] - joints_3d[0]
            print(f"\n[debug] WiLoR output, fingertips relative to wrist:")
            for i, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
                v = fingertips_relative_to_wrist[i]
                print(f"  {name:7s} [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")

        with latest_pose.lock:
            latest_pose.joints_3d = joints_3d
            latest_pose.kpts_2d = kpts_2d
            latest_pose.is_right = is_right
            latest_pose.capture_ts = capture_ts
            latest_pose.inference_ts = time.perf_counter()
            latest_pose.seq += 1


def command_thread(latest_pose: LatestPose, filtered_pose: LatestPose,
                   retargeting, stop_event: threading.Event,
                   stats: StageStats, retarget_stats: StageStats,
                   target_hz: float = 50.0,
                   filter_params: dict = None):
    """Reads latest_pose at fixed rate; filters, rotates, retargets;
    writes filtered_pose with both filtered keypoints (for visualisation)
    and retargeted qpos (for robot)."""
    period = 1.0 / target_hz
    filter_params = filter_params or dict(min_cutoff=0.5, beta=0.05)
    joint_filter = OneEuroVector(**filter_params)
    kpts2d_filter = OneEuroVector(**filter_params)

    # Pre-extract retargeter config for the hot path.
    indices = retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0]
    task_indices = indices[1]
    joint_limits = retargeting.joint_limits  # (12, 2)
    limit_lo = joint_limits[:, 0]
    limit_hi = joint_limits[:, 1]
    margin = 0.001  # rad, keep last_qpos slightly off bounds

    next_tick = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        dt = next_tick - now
        if dt > 0:
            time.sleep(dt)
        next_tick += period
        if next_tick < time.perf_counter() - period:
            next_tick = time.perf_counter() + period

        with Timer(stats):
            with latest_pose.lock:
                if latest_pose.joints_3d is None:
                    continue
                joints = latest_pose.joints_3d
                kpts_2d = latest_pose.kpts_2d
                seq = latest_pose.seq
                capture_ts = latest_pose.capture_ts
                inference_ts = latest_pose.inference_ts

            t = time.perf_counter()
            filtered_3d = joint_filter(joints, t)
            filtered_2d = (kpts2d_filter(kpts_2d, t)
                           if kpts_2d is not None else None)

            # Rotate MANO -> XHand frame, compute target vectors, retarget.
            joints_xhand = filtered_3d @ R_MANO_TO_XHAND.T
            target_vectors = (joints_xhand[task_indices]
                              - joints_xhand[origin_indices])

            with Timer(retarget_stats):
                qpos = retargeting.retarget(target_vectors)

            qpos_clipped = np.clip(qpos, limit_lo, limit_hi)
            hit_limit = not np.allclose(qpos, qpos_clipped)

            # Keep warm-start state slightly off bounds to avoid optimiser
            # stalls on subsequent calls.
            retargeting.last_qpos = np.clip(
                retargeting.last_qpos, limit_lo + margin, limit_hi - margin)

            with filtered_pose.lock:
                filtered_pose.joints_3d = filtered_3d
                filtered_pose.kpts_2d = filtered_2d
                filtered_pose.qpos = qpos_clipped
                filtered_pose.qpos_clipped = hit_limit
                filtered_pose.capture_ts = capture_ts
                filtered_pose.inference_ts = inference_ts
                filtered_pose.seq = seq

            # TODO: send qpos_clipped to robot here.


# --- main thread visualisation --------------------------------------------

def draw_overlay(frame_bgr: np.ndarray, kpts_2d: np.ndarray,
                 hud_lines: list = None) -> np.ndarray:
    """Draw skeleton bones, joint markers, and an optional HUD on a frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    pts = kpts_2d.astype(int)

    if (pts[:, 0].min() < -w or pts[:, 0].max() > 2 * w
            or pts[:, 1].min() < -h or pts[:, 1].max() > 2 * h):
        return out

    for a, b in HAND_BONES:
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), bone_colour_bgr(a, b), 2)
    for x, y in pts:
        cv2.circle(out, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(out, (x, y), 4, (0, 0, 0), 1)

    y0 = 20
    if hud_lines:
        for line in hud_lines:
            cv2.putText(out, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            y0 += 18
    return out


def qpos_hud_lines(qpos: np.ndarray, hit_limit: bool) -> list:
    """Format qpos as compact per-finger HUD lines.
    XHand joint order:
      0:thumb_bend  1:thumb_rota1  2:thumb_rota2
      3:index_bend  4:index_j1     5:index_j2
      6:mid_j1      7:mid_j2
      8:ring_j1     9:ring_j2
     10:pinky_j1   11:pinky_j2
    """
    qd = np.rad2deg(qpos)
    lines = [
        f"thumb {qd[0]:+4.0f} {qd[1]:+4.0f} {qd[2]:+4.0f}",
        f"index {qd[3]:+4.0f} {qd[4]:+4.0f} {qd[5]:+4.0f}",
        f"mid        {qd[6]:+4.0f} {qd[7]:+4.0f}",
        f"ring       {qd[8]:+4.0f} {qd[9]:+4.0f}",
        f"pinky     {qd[10]:+4.0f} {qd[11]:+4.0f}",
    ]
    if hit_limit:
        lines.append("!! at limit")
    return lines


def main():



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[init] device={device}, dtype={dtype}")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    # Build retargeter (slow cold call - do this before launching threads)
    print(f"[init] building XHand retargeter from {XHAND_CONFIG}")
    RetargetingConfig.set_default_urdf_dir(DEX_URDF_DIR)
    retargeting = RetargetingConfig.load_from_file(XHAND_CONFIG).build()
    n_joints = len(retargeting.optimizer.target_joint_names)
    print(f"[init] retargeter ready, {n_joints} target joints")

    stages: Dict[str, StageStats] = {
        name: StageStats(name) for name in [
            "camera",       # camera thread per-frame
            "inference",    # inference thread per-prediction
            "infer_idle",   # inference thread waiting for new frames
            "command",      # command thread per-tick (filter+retarget+write)
            "retarget",     # subset of command: retargeting alone
            "viz_2d",       # main thread overlay drawing
            "loop_total",   # main thread per-iteration
        ]
    }

    latest_frame = LatestFrame()
    latest_pose = LatestPose()
    filtered_pose = LatestPose()
    stop_event = threading.Event()

    # Warmup the WiLoR model so first-frame jitter doesn't pollute stats.
    print("[warmup] priming WiLoR...")
    cap_warm = cv2.VideoCapture(0)
    for _ in range(5):
        ok, f = cap_warm.read()
        if ok:
            with suppress_output():
                _ = pipe.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap_warm.release()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[warmup] done.")

    # Filter parameters - tune via observation.
    filter_params = dict(min_cutoff=0.5, beta=0.05)

    threads = [
        threading.Thread(target=camera_thread, name="camera", daemon=True,
                         args=(latest_frame, stop_event, stages["camera"])),
        threading.Thread(target=inference_thread, name="inference", daemon=True,
                         args=(pipe, latest_frame, latest_pose, stop_event,
                               stages["inference"], stages["infer_idle"])),
        threading.Thread(target=command_thread, name="command", daemon=True,
                         args=(latest_pose, filtered_pose, retargeting,
                               stop_event, stages["command"],
                               stages["retarget"]),
                         kwargs=dict(target_hz=50.0,
                                     filter_params=filter_params)),
    ]
    for t in threads:
        t.start()

    PRINT_EVERY = 2.0
    last_print = time.perf_counter()
    t_start = time.perf_counter()
    n_displayed = 0
    n_with_pose = 0

    try:
        while not stop_event.is_set():
            with Timer(stages["loop_total"]):
                with latest_frame.lock:
                    bgr = (latest_frame.bgr.copy()
                           if latest_frame.bgr is not None else None)
                with filtered_pose.lock:
                    fp_kpts_2d = (filtered_pose.kpts_2d.copy()
                                  if filtered_pose.kpts_2d is not None else None)
                    fp_qpos = (filtered_pose.qpos.copy()
                               if filtered_pose.qpos is not None else None)
                    fp_clipped = filtered_pose.qpos_clipped
                    fp_capture = filtered_pose.capture_ts
                    fp_inf_ts = filtered_pose.inference_ts

                if bgr is not None:
                    hud = []
                    if fp_capture:
                        age_ms = (time.perf_counter() - fp_capture) * 1000.0
                        hud.append(f"pose age {age_ms:5.1f} ms")
                    if fp_capture and fp_inf_ts:
                        inf_lat_ms = (fp_inf_ts - fp_capture) * 1000.0
                        hud.append(f"infer lat {inf_lat_ms:5.1f} ms")
                    if fp_qpos is not None:
                        hud.extend(qpos_hud_lines(fp_qpos, fp_clipped))

                    if fp_kpts_2d is not None:
                        n_with_pose += 1
                        with Timer(stages["viz_2d"]):
                            bgr = draw_overlay(bgr, fp_kpts_2d, hud_lines=hud)

                    cv2.imshow("WiLoR teleop", bgr)
                    n_displayed += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                if key == ord("s"):
                    print_stats(stages, header="midrun snapshot")

            now = time.perf_counter()
            if now - last_print >= PRINT_EVERY:
                window = now - last_print
                infer_p50 = stages["inference"].summary().get("p50", 0)
                cmd_p50 = stages["command"].summary().get("p50", 0)
                ret_p50 = stages["retarget"].summary().get("p50", 0)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"infer p50={infer_p50:5.1f} ms  "
                      f"cmd p50={cmd_p50:5.2f} ms  "
                      f"retarget p50={ret_p50:5.2f} ms  "
                      f"display {n_displayed/window:5.1f} Hz "
                      f"({n_with_pose}/{n_displayed} pose)")
                last_print = now
                n_displayed = 0
                n_with_pose = 0
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        print_stats(stages, header="FINAL")
        elapsed = time.perf_counter() - t_start
        print(f"runtime: {elapsed:.1f}s")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()