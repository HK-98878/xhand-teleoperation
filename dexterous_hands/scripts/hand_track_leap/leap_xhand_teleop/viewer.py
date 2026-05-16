"""
Real-time hand wireframe visualizer with diagnostic overlay.

Two panels:
  1. Open3D 3D viewer:  21 keypoints (spheres) + bones (line set), with axis
     gizmo at the wrist showing the wrist-local frame.
  2. Matplotlib diagnostic panel (separate window):
        - FPS / latency / dropped frames
        - Tracking confidence
        - Per-keypoint jitter (std dev over a sliding window)
        - 12-bar chart of current XHAND1 joint commands w/ limit shading

Both panels are non-blocking and run cooperatively from the main thread.
Either can be toggled off via constructor flags.

The Open3D viewer in particular is intentionally tolerant of late/dropped
frames -- if no new keypoints arrive, it just keeps the last geometry.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Optional

import numpy as np

from .keypoints import HAND_EDGES, KP, FINGER_CHAINS
from .retarget_xhand1 import XHAND1_JOINT_NAMES, JOINT_LIMITS_RAD

try:
    import open3d as o3d  # type: ignore
    _HAVE_O3D = True
except ImportError:
    _HAVE_O3D = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


_FINGER_COLORS = {
    "thumb":  [0.95, 0.55, 0.25],
    "index":  [0.30, 0.75, 0.95],
    "middle": [0.40, 0.85, 0.40],
    "ring":   [0.85, 0.85, 0.30],
    "pinky":  [0.85, 0.40, 0.85],
}


class HandWireframeViewer:
    """Combined 3D wireframe + diagnostics viewer.

    Args:
        show_3d:    enable Open3D wireframe window
        show_diag:  enable matplotlib diagnostics window
        keypoint_radius: sphere radius in meters (cosmetic)
    """

    def __init__(
        self,
        show_3d: bool = True,
        show_diag: bool = True,
        keypoint_radius: float = 0.006,
    ):
        self.show_3d = show_3d and _HAVE_O3D
        self.show_diag = show_diag and _HAVE_MPL
        if show_3d and not _HAVE_O3D:
            print("[viewer] open3d not installed; 3D view disabled "
                  "(`pip install open3d`)")
        if show_diag and not _HAVE_MPL:
            print("[viewer] matplotlib not installed; diagnostics disabled "
                  "(`pip install matplotlib`)")

        self._radius = keypoint_radius

        # ---- 3D state ----
        self._vis = None
        self._line_set = None
        self._spheres: list = []
        self._wrist_frame = None
        self._initialized_3d = False

        # ---- Diagnostics state ----
        self._fps_window = deque(maxlen=60)  # frame timestamps
        self._jitter_buffer = deque(maxlen=30)  # last N keypoint arrays
        self._dropped = 0
        self._dropped_total = 0
        self._last_recv_time: Optional[float] = None
        self._fig = None
        self._axes = None
        self._joint_bars = None
        self._jitter_line = None
        self._info_text = None

    # ---------------------------------------------------------------- 3D
    def open_now(self):
        """Open both viewer windows immediately with a placeholder pose.

        Call this BEFORE the main loop so the user sees windows even when no
        tracking data is arriving (e.g. Leap not delivering frames). The
        placeholder is a flat right hand at the origin.
        """
        # Placeholder: flat right hand spread across +x with thumb to +z.
        placeholder = np.array([
            [0.000, 0.0, 0.000],   # wrist
            [0.014, 0.0, 0.014],   # thumb_cmc
            [0.035, 0.0, 0.035],   # thumb_mcp
            [0.050, 0.0, 0.050],   # thumb_ip
            [0.062, 0.0, 0.062],   # thumb_tip
            [0.085, 0.0, 0.020],   # index_mcp
            [0.115, 0.0, 0.020],
            [0.140, 0.0, 0.020],
            [0.160, 0.0, 0.020],
            [0.085, 0.0, 0.000],   # middle_mcp
            [0.115, 0.0, 0.000],
            [0.140, 0.0, 0.000],
            [0.160, 0.0, 0.000],
            [0.085, 0.0, -0.020],  # ring_mcp
            [0.115, 0.0, -0.020],
            [0.140, 0.0, -0.020],
            [0.160, 0.0, -0.020],
            [0.085, 0.0, -0.040],  # pinky_mcp
            [0.115, 0.0, -0.040],
            [0.140, 0.0, -0.040],
            [0.160, 0.0, -0.040],
        ], dtype=np.float32)
        if self.show_3d:
            self._init_3d(placeholder)
            # Run a few render passes so the window actually appears.
            for _ in range(5):
                self._vis.poll_events()
                self._vis.update_renderer()
        if self.show_diag:
            self._init_diag()
            for _ in range(3):
                try:
                    self._fig.canvas.flush_events()
                except Exception:
                    pass

    def _init_3d(self, points: np.ndarray):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="LeapMotion -> XHAND1 Wireframe",
                                width=1100, height=750)

        # Coordinate frame at origin (camera frame)
        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self._vis.add_geometry(self._cam_frame)

        # Wrist coordinate frame (small, will be moved/rotated each frame)
        self._wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        self._vis.add_geometry(self._wrist_frame)
        self._wrist_frame_prev_T = np.eye(4)

        # 21 spheres -- one per keypoint, colored by which finger they belong to
        kp_to_finger = {}
        for fname, chain in FINGER_CHAINS.items():
            for k in chain:
                kp_to_finger[int(k)] = fname
        kp_to_finger[int(KP.WRIST)] = "wrist"

        for i in range(21):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=self._radius, resolution=8)
            color = _FINGER_COLORS.get(kp_to_finger.get(i, "wrist"), [0.85, 0.85, 0.85])
            if i == int(KP.WRIST):
                color = [1.0, 1.0, 1.0]
            sph.paint_uniform_color(color)
            sph.translate(points[i], relative=False)
            sph.compute_vertex_normals()
            self._spheres.append(sph)
            self._vis.add_geometry(sph)
        # Track sphere positions in a parallel array because Open3D mesh
        # objects (pybind11) don't accept arbitrary Python attributes.
        self._sphere_positions = points.copy()

        # Bones as a single LineSet
        self._line_set = o3d.geometry.LineSet()
        self._line_set.points = o3d.utility.Vector3dVector(points)
        self._line_set.lines = o3d.utility.Vector2iVector(np.array(HAND_EDGES))
        line_colors = []
        # Color each edge by the finger it belongs to.
        finger_for_edge = {}
        for fname, chain in FINGER_CHAINS.items():
            for a, b in zip(chain[:-1], chain[1:]):
                finger_for_edge[(int(a), int(b))] = fname
                finger_for_edge[(int(b), int(a))] = fname
        for a, b in HAND_EDGES:
            fname = finger_for_edge.get((a, b), "palm")
            line_colors.append(_FINGER_COLORS.get(fname, [0.6, 0.6, 0.6]))
        self._line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
        self._vis.add_geometry(self._line_set)

        # Camera framing
        opt = self._vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.07])
        opt.line_width = 4.0
        opt.point_size = 4.0

        ctr = self._vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_front([0.0, -0.3, -1.0])
        ctr.set_lookat(points[int(KP.MIDDLE_MCP)].tolist())
        ctr.set_up([0.0, 1.0, 0.0])

        self._initialized_3d = True

    def _update_3d(self, points: np.ndarray, kp):
        if not self._initialized_3d:
            self._init_3d(points)
            return

        # Move spheres (translate by delta to avoid recomputing the mesh)
        for i, sph in enumerate(self._spheres):
            delta = points[i] - self._sphere_positions[i]
            sph.translate(delta, relative=True)
            self._vis.update_geometry(sph)
        self._sphere_positions = points.copy()

        # Update bone lines
        self._line_set.points = o3d.utility.Vector3dVector(points)
        self._vis.update_geometry(self._line_set)

        # Wrist frame: build R from the same wrist-local basis used by retargeter
        wrist = points[int(KP.WRIST)]
        mid_mcp = points[int(KP.MIDDLE_MCP)]
        idx_mcp = points[int(KP.INDEX_MCP)]
        pky_mcp = points[int(KP.PINKY_MCP)]
        x = mid_mcp - wrist
        x /= (np.linalg.norm(x) + 1e-9)
        n = np.cross(idx_mcp - wrist, pky_mcp - wrist)
        if kp.is_left:
            n = -n
        y = n / (np.linalg.norm(n) + 1e-9)
        z = np.cross(x, y); z /= (np.linalg.norm(z) + 1e-9)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0] = x; T[:3, 1] = y; T[:3, 2] = z
        T[:3, 3] = wrist
        # Apply T relative to previous: T_new = T @ inv(T_prev)
        T_rel = T @ np.linalg.inv(self._wrist_frame_prev_T)
        self._wrist_frame.transform(T_rel)
        self._wrist_frame_prev_T = T
        self._vis.update_geometry(self._wrist_frame)

    # ------------------------------------------------------- Diagnostics
    def _init_diag(self):
        plt.ion()
        self._fig, axes = plt.subplots(3, 1, figsize=(9, 7),
                                       gridspec_kw={"height_ratios": [1, 2, 2]})
        self._fig.canvas.manager.set_window_title("Teleop Diagnostics")
        self._axes = axes

        # Panel 0: text info
        axes[0].axis("off")
        self._info_text = axes[0].text(
            0.01, 0.5, "", family="monospace", fontsize=10, va="center"
        )

        # Panel 1: jitter (per-keypoint std dev over last N frames, mm)
        axes[1].set_title("Per-keypoint jitter (mm, std over ~1 s)")
        axes[1].set_xlim(-0.5, 20.5)
        axes[1].set_ylim(0, 5)
        axes[1].set_xticks(range(21))
        axes[1].set_xticklabels([f"{i}" for i in range(21)], fontsize=7)
        axes[1].set_xlabel("keypoint index")
        axes[1].grid(True, alpha=0.3)
        self._jitter_bars = axes[1].bar(range(21), [0]*21, color="#4ab3d4")

        # Panel 2: joint commands w/ limit envelope
        axes[2].set_title("XHAND1 joint commands (rad) with limit envelope")
        axes[2].set_xticks(range(12))
        axes[2].set_xticklabels(XHAND1_JOINT_NAMES, rotation=30, ha="right", fontsize=7)
        axes[2].axhline(0, color="k", linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
        # Plot limits as a translucent band
        lo = JOINT_LIMITS_RAD[:, 0]
        hi = JOINT_LIMITS_RAD[:, 1]
        axes[2].fill_between(range(12), lo, hi, alpha=0.15, color="green", step="mid",
                             label="joint range")
        self._joint_bars = axes[2].bar(range(12), [0]*12, color="#e67e22", width=0.6)
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_ylim(JOINT_LIMITS_RAD[:, 0].min() - 0.3,
                         JOINT_LIMITS_RAD[:, 1].max() + 0.3)

        plt.tight_layout()
        self._fig.show()

    def _update_diag(self, points: np.ndarray, joint_cmd: np.ndarray, confidence: float):
        if self._fig is None:
            self._init_diag()

        # FPS
        now = time.monotonic()
        self._fps_window.append(now)
        if len(self._fps_window) >= 2:
            fps = (len(self._fps_window) - 1) / (self._fps_window[-1] - self._fps_window[0])
        else:
            fps = 0.0
        latency_ms = (now - self._last_recv_time) * 1000 if self._last_recv_time else 0
        self._last_recv_time = now

        # Jitter
        self._jitter_buffer.append(points.copy())
        if len(self._jitter_buffer) >= 5:
            arr = np.stack(self._jitter_buffer, axis=0)  # (T, 21, 3)
            stds_mm = (arr.std(axis=0).mean(axis=1)) * 1000.0  # mean std over xyz, mm
        else:
            stds_mm = np.zeros(21)

        info = (
            f"FPS: {fps:5.1f}    "
            f"frame age: {latency_ms:5.1f} ms    "
            f"confidence: {confidence:.2f}    "
            f"dropped (rolling): {self._dropped}    "
            f"dropped (total): {self._dropped_total}\n"
            f"mean jitter: {stds_mm.mean():.2f} mm   "
            f"max jitter: {stds_mm.max():.2f} mm @ kp{int(stds_mm.argmax())}   "
            f"|joint_cmd|_inf: {np.abs(joint_cmd).max():.3f} rad"
        )
        self._info_text.set_text(info)

        # Update jitter bars
        for bar, h in zip(self._jitter_bars, stds_mm):
            bar.set_height(h)
        ymax = max(5.0, float(stds_mm.max()) * 1.2)
        self._axes[1].set_ylim(0, ymax)

        # Update joint command bars; color red if at/near limit
        margins = np.minimum(joint_cmd - JOINT_LIMITS_RAD[:, 0],
                             JOINT_LIMITS_RAD[:, 1] - joint_cmd)
        for i, (bar, v, m) in enumerate(zip(self._joint_bars, joint_cmd, margins)):
            bar.set_height(float(v))
            bar.set_color("#c0392b" if m < 0.05 else "#e67e22")

        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    # ----------------------------------------------------------- public
    def update(self, points: np.ndarray, kp, joint_cmd: np.ndarray):
        """Update both panels with a new frame.

        Args:
            points:    (21, 3) keypoints in world frame (post-filter)
            kp:        the HandKeypoints object (for is_left, confidence)
            joint_cmd: (12,) clipped joint commands
        """
        if self.show_3d:
            self._update_3d(points, kp)
        if self.show_diag:
            self._update_diag(points, joint_cmd, kp.confidence)

    def poll(self):
        """Pump GUI events. Call once per main-loop iteration. Returns False
        if the user has closed a window (signal to exit)."""
        keep_running = True
        if self.show_3d and self._vis is not None:
            ok = self._vis.poll_events()
            self._vis.update_renderer()
            if not ok:
                keep_running = False
        if self.show_diag and self._fig is not None:
            try:
                self._fig.canvas.flush_events()
                if not plt.fignum_exists(self._fig.number):
                    keep_running = False
            except Exception:
                pass
        return keep_running

    def note_drop(self):
        """Call when a frame is dropped (no new keypoints this loop)."""
        self._dropped += 1
        self._dropped_total += 1

    def close(self):
        if self._vis is not None:
            try: self._vis.destroy_window()
            except Exception: pass
        if self._fig is not None:
            try: plt.close(self._fig)
            except Exception: pass