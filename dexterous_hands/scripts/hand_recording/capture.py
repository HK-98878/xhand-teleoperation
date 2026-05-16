"""
VLA data capture script.

Records 1080p30 MJPEG video from one or more UVC webcams alongside xhand
robot state + dense tactile data via rosbridge. Each camera writes to its own
subdir; episodes are marked by spacebar in a single session-level events file.

Controls (when preview window is focused):
    SPACE   start / end an episode
    z       zero haptic sensors
    q       quit

Output layout per session:

    session_YYYYMMDD_HHMMSS/
        cameras/
            <name1>/
                rgb.mkv
                frames.jsonl
                metadata.json
            <name2>/
                ...
        events.jsonl
        robot_state.jsonl
        commands.jsonl
        haptic.bin
        haptic_index.jsonl
        session_meta.json

Cameras run independently; their frame timestamps don't align exactly.
Post-processing (export.py) re-times them against a common tick if needed.

Usage:
    python capture.py
    python capture.py --no-robot
    python capture.py --no-camera
    python capture.py --list-controls --camera <name>
    python capture.py --config other.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")


# ---------------------------------------------------------------------------
# v4l2 helpers
# ---------------------------------------------------------------------------

def _have_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None


def list_v4l2_controls(device: str) -> None:
    """Print all controls the device supports."""
    if not _have_v4l2_ctl():
        log.error("v4l2-ctl not found. Install with: sudo apt install v4l-utils")
        sys.exit(1)
    log.info("Controls for %s:", device)
    subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"], check=False)
    log.info("Supported formats:")
    subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"], check=False)


def apply_v4l2_controls(device: str, controls: dict) -> None:
    """Apply controls. Auto toggles are set first so manual values stick."""
    if not _have_v4l2_ctl():
        log.warning("v4l2-ctl not found — skipping camera control setup. "
                    "Install with: sudo apt install v4l-utils")
        return

    def is_auto_toggle(name: str) -> bool:
        return "auto" in name.lower()

    items = list(controls.items())
    items.sort(key=lambda kv: 0 if is_auto_toggle(kv[0]) else 1)

    for name, value in items:
        if value is None:
            continue
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={name}={value}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning("Could not set %s=%s on %s: %s",
                        name, value, device,
                        result.stderr.strip() or result.stdout.strip())
        else:
            log.info("[%s] set %s = %s", Path(device).name[:40], name, value)


# ---------------------------------------------------------------------------
# Session paths (camera dirs are created on demand by CameraRecorder)
# ---------------------------------------------------------------------------

@dataclass
class SessionPaths:
    root: Path
    cameras_dir: Path
    events: Path
    robot_state: Path
    commands: Path
    haptic_bin: Path
    haptic_index: Path
    meta: Path

    @classmethod
    def for_session(cls, output_dir: Path,
                    session_name: Optional[str] = None) -> "SessionPaths":
        if session_name is None:
            session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        root = output_dir / session_name
        cameras_dir = root / "cameras"
        cameras_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            cameras_dir=cameras_dir,
            events=root / "events.jsonl",
            robot_state=root / "robot_state.jsonl",
            commands=root / "commands.jsonl",
            haptic_bin=root / "haptic.bin",
            haptic_index=root / "haptic_index.jsonl",
            meta=root / "session_meta.json",
        )


# ---------------------------------------------------------------------------
# Camera recorder (one per camera; threaded grab + write)
# ---------------------------------------------------------------------------

class CameraRecorder:
    """Threaded camera capture. Records continuously for the whole session
    into <name>/rgb.<container>; per-frame metadata goes to <name>/frames.jsonl."""

    def __init__(self, cfg: dict, camera_dir: Path, container: str,
                 warn_dropped_frames: bool):
        if "name" not in cfg:
            raise ValueError("camera config missing required 'name' field")
        if "device" not in cfg:
            raise ValueError(f"camera '{cfg.get('name')}' missing 'device' field")

        self.cfg = cfg
        self.name = cfg["name"]
        self.device = cfg["device"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.fps = cfg["fps"]
        self.fourcc = cfg["fourcc"]
        self.warn_dropped_frames = warn_dropped_frames

        self.dir = camera_dir
        self.video_basename = f"rgb.{container}"
        self.video_path = camera_dir / self.video_basename
        self.frames_path = camera_dir / "frames.jsonl"
        self.metadata_path = camera_dir / "metadata.json"

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_fp = None

        self._grab_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._frame_queue: Queue = Queue(maxsize=64)

        self._running = threading.Event()
        self._latest_frame = None
        self._latest_lock = threading.Lock()

        # Stats
        self.frames_grabbed = 0
        self.frames_written = 0
        self.frames_dropped_writer = 0
        self.actual_width: Optional[int] = None
        self.actual_height: Optional[int] = None
        self.actual_fps_reported: Optional[float] = None

    def _opencv_open_arg(self):
        """OpenCV's V4L2 backend accepts integer indexes or device path strings.
        For /dev/v4l/by-id/... symlinks we pass the string path directly.
        For /dev/videoN we convert to integer N (slightly faster path)."""
        d = self.device
        if d.startswith("/dev/video"):
            try:
                return int(d.replace("/dev/video", ""))
            except ValueError:
                pass
        return d

    def open(self) -> None:
        # Apply v4l2 controls *before* opening with OpenCV — once OpenCV grabs
        # the device, some control writes get rejected.
        apply_v4l2_controls(self.device, self.cfg.get("controls", {}))

        self.dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(self._opencv_open_arg(), cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"[{self.name}] could not open {self.device}")

        # Order matters: fourcc first, then size, then fps.
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps_reported = cap.get(cv2.CAP_PROP_FPS)
        log.info("[%s] opened: %dx%d @ %.1f fps (requested %dx%d @ %d)",
                 self.name, self.actual_width, self.actual_height,
                 self.actual_fps_reported, self.width, self.height, self.fps)

        if (self.actual_width, self.actual_height) != (self.width, self.height):
            log.warning("[%s] camera did not honour requested resolution — got %dx%d",
                        self.name, self.actual_width, self.actual_height)

        writer_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            str(self.video_path), writer_fourcc, self.fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"[{self.name}] could not open VideoWriter at {self.video_path}")

        self._cap = cap
        self._writer = writer
        self._frames_fp = open(self.frames_path, "w", buffering=1)

        self._running.set()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True,
            name=f"grab-{self.name}",
        )
        self._write_thread = threading.Thread(
            target=self._write_loop, daemon=True,
            name=f"write-{self.name}",
        )
        self._grab_thread.start()
        self._write_thread.start()
        log.info("[%s] recording → %s", self.name, self.video_path)

    def _grab_loop(self) -> None:
        expected_dt = 1.0 / self.fps
        warn_threshold = expected_dt * 1.5
        last_t = None

        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok:
                log.warning("[%s] camera read failed; retrying...", self.name)
                time.sleep(0.01)
                continue

            host_ts_ns = time.time_ns()
            self.frames_grabbed += 1

            if self.warn_dropped_frames and last_t is not None:
                gap = (host_ts_ns - last_t) / 1e9
                if gap > warn_threshold:
                    log.warning("[%s] frame gap %.3fs (expected ~%.3fs) — possible drop",
                                self.name, gap, expected_dt)
            last_t = host_ts_ns

            with self._latest_lock:
                self._latest_frame = frame

            try:
                self._frame_queue.put_nowait((frame.copy(), host_ts_ns))
            except Exception:
                self.frames_dropped_writer += 1
                log.warning("[%s] writer queue full — dropped frame (total: %d)",
                            self.name, self.frames_dropped_writer)

    def _write_loop(self) -> None:
        while self._running.is_set() or not self._frame_queue.empty():
            try:
                frame, host_ts_ns = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue
            if self._writer is None or self._frames_fp is None:
                continue
            self._writer.write(frame)
            record = {
                "camera": self.name,
                "depth_file": None,
                "frame_index": self.frames_written,
                "height": self.height,
                "host_timestamp_ns": host_ts_ns,
                "rgb_video": self.video_basename,
                "rgb_video_frame": self.frames_written,
                "width": self.width,
            }
            self._frames_fp.write(json.dumps(record) + "\n")
            self.frames_written += 1

    def get_preview_frame(self):
        with self._latest_lock:
            return None if self._latest_frame is None else self._latest_frame

    def close(self, write_metadata: bool = True) -> int:
        self._running.clear()
        if self._grab_thread:
            self._grab_thread.join(timeout=2)
        if self._write_thread:
            self._write_thread.join(timeout=2)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._frames_fp is not None:
            self._frames_fp.close()
            self._frames_fp = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("[%s] closed (%d grabbed, %d written, %d dropped at writer)",
                 self.name, self.frames_grabbed, self.frames_written,
                 self.frames_dropped_writer)
        if write_metadata:
            self._write_metadata()
        return self.frames_written

    def _write_metadata(self) -> None:
        """Snapshot the resolved camera config + capture stats at session end."""
        meta = {
            "name": self.name,
            "device": self.device,
            "configured": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "fourcc": self.fourcc,
                "controls": self.cfg.get("controls", {}),
            },
            "actual": {
                "width": self.actual_width,
                "height": self.actual_height,
                "fps_reported": self.actual_fps_reported,
            },
            "frames_grabbed": self.frames_grabbed,
            "frames_written": self.frames_written,
            "frames_dropped_writer": self.frames_dropped_writer,
            "video": self.video_basename,
            "frames_jsonl": "frames.jsonl",
        }
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Event logger (episode boundaries) — session-level, applies to all cameras
# ---------------------------------------------------------------------------

class TerminalKeyReader:
    """Headless single-key input via raw terminal mode + select().

    Used when there's no OpenCV preview window (--no-camera mode). Puts the
    controlling terminal into cbreak (no line buffering, no echo) so single
    keypresses are immediately readable; restores normal mode on close.

    Returns the same key codes cv2.waitKey() would: ord(' '), ord('q'), etc.
    Returns 0xFF when no key is pressed within the poll timeout.

    Requires a real TTY on stdin; if stdin is piped/redirected, falls back to
    a non-blocking no-op so the script doesn't crash but also doesn't receive
    input."""

    NO_KEY = 0xFF

    def __init__(self, poll_timeout_s: float = 0.05):
        self.poll_timeout_s = poll_timeout_s
        self._fd = sys.stdin.fileno()
        self._is_tty = sys.stdin.isatty()
        self._orig_settings = None

    def __enter__(self) -> "TerminalKeyReader":
        if self._is_tty:
            self._orig_settings = termios.tcgetattr(self._fd)
            # cbreak rather than full raw: keeps Ctrl-C → SIGINT working so
            # the signal handler still fires on Ctrl-C.
            tty.setcbreak(self._fd)
        else:
            log.warning("stdin is not a TTY; key input disabled. Use Ctrl-C to quit.")
        return self

    def __exit__(self, *_exc) -> None:
        if self._orig_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._orig_settings)
            self._orig_settings = None

    def read_key(self) -> int:
        """Block up to poll_timeout_s for one keypress, returning its byte
        value (0-255) or NO_KEY if the timeout expired. Multi-byte sequences
        (arrow keys, function keys) are consumed but reported as their first
        byte; we don't currently use any of those."""
        if not self._is_tty:
            time.sleep(self.poll_timeout_s)
            return self.NO_KEY
        ready, _, _ = select.select([self._fd], [], [], self.poll_timeout_s)
        if not ready:
            return self.NO_KEY
        ch = sys.stdin.read(1)
        if not ch:
            return self.NO_KEY
        return ord(ch) & 0xFF


# ---------------------------------------------------------------------------
# Event logger (episode boundaries) — session-level, applies to all cameras
# ---------------------------------------------------------------------------

class EventLogger:
    def __init__(self, path: Path):
        self.path = path
        self._fp = open(path, "w", buffering=1)
        self._lock = threading.Lock()
        self._count = 0

    def log(self, event: str, **extra) -> None:
        record = {"event": event, "robot_timestamp_ns": time.time_ns()}
        record.update(extra)
        with self._lock:
            self._fp.write(json.dumps(record) + "\n")
            self._count += 1

    def close(self) -> int:
        with self._lock:
            self._fp.close()
        return self._count


# ---------------------------------------------------------------------------
# Robot recorder
# ---------------------------------------------------------------------------

class RobotRecorder:
    """Subscribes to /xhand_control/xhand_state (XHandStateArray) and
    /xhand_control/xhand_command (XHandCommand) via roslibpy.

    Writes:
      - robot_state.jsonl: joint state, effort, temperature, errors
      - commands.jsonl: joint targets from teleop
      - haptic.bin: raw float32 stream of all FingerSensorState.raw_force vectors
      - haptic_index.jsonl: timestamps + byte offsets per haptic sample

    Also exposes zero_sensors() which calls /xhand_control/reset_hand_sensor
    sequentially for each finger.
    """

    HAND_ID = 0
    SENSOR_IDS = [17, 18, 19, 20, 21]

    STATE_TOPIC = "/xhand_control/xhand_state"
    STATE_TYPE = "xhand_control_ros/XHandStateArray"

    COMMAND_TOPIC = "/xhand_control/xhand_command"
    COMMAND_TYPE = "xhand_control_ros/XHandCommand"

    RESET_SERVICE = "/xhand_control/reset_hand_sensor"
    RESET_SERVICE_TYPE = "xhand_control_ros/ResetSensor"

    def __init__(self, cfg: dict, paths: SessionPaths):
        self.cfg = cfg or {}
        self.paths = paths

        self.host = self.cfg.get("host", "localhost")
        self.port = int(self.cfg.get("port", 9090))
        self.hand_index = int(self.cfg.get("hand_index", 0))
        self.expected_n_fingers = int(self.cfg.get("n_fingers", 5))
        self.expected_n_joints = int(self.cfg.get("n_joints", 12))

        self._n_vectors_per_finger: Optional[int] = None

        self._ros = None
        self._state_topic = None
        self._command_topic = None
        self._reset_service = None

        self._state_fp = None
        self._command_fp = None
        self._haptic_fp = None
        self._haptic_idx_fp = None

        self._lock = threading.Lock()
        self._connected = threading.Event()

        self.samples = 0
        self.command_samples = 0
        self.haptic_byte_offset = 0
        self.last_msg_wall_ns: Optional[int] = None
        self._last_warn_t = 0.0

    def open(self) -> None:
        try:
            import roslibpy  # noqa: F401
        except ImportError:
            log.error("roslibpy not installed. pip install roslibpy")
            raise

        self._state_fp = open(self.paths.robot_state, "w", buffering=1)
        self._command_fp = open(self.paths.commands, "w", buffering=1)
        self._haptic_fp = open(self.paths.haptic_bin, "wb", buffering=0)
        self._haptic_idx_fp = open(self.paths.haptic_index, "w", buffering=1)

        import roslibpy
        self._ros = roslibpy.Ros(host=self.host, port=self.port)
        self._ros.on_ready(self._on_ready, run_in_thread=True)
        self._ros.run()

        if not self._connected.wait(timeout=5.0):
            log.warning("rosbridge connection to %s:%d not confirmed after 5s — "
                        "still trying in background", self.host, self.port)

    def _on_ready(self) -> None:
        import roslibpy
        log.info("rosbridge connected at %s:%d", self.host, self.port)
        self._connected.set()

        self._state_topic = roslibpy.Topic(
            self._ros, self.STATE_TOPIC, self.STATE_TYPE,
            queue_size=1, throttle_rate=0,
        )
        self._state_topic.subscribe(self._on_state_message)
        log.info("Subscribed to %s", self.STATE_TOPIC)

        self._command_topic = roslibpy.Topic(
            self._ros, self.COMMAND_TOPIC, self.COMMAND_TYPE,
            queue_size=1, throttle_rate=0,
        )
        self._command_topic.subscribe(self._on_command_message)
        log.info("Subscribed to %s", self.COMMAND_TOPIC)

        self._reset_service = roslibpy.Service(
            self._ros, self.RESET_SERVICE, self.RESET_SERVICE_TYPE,
        )

    def close(self) -> int:
        for topic_attr in ("_state_topic", "_command_topic"):
            topic = getattr(self, topic_attr, None)
            if topic is not None:
                try:
                    topic.unsubscribe()
                except Exception as e:
                    log.warning("%s unsubscribe failed: %s", topic_attr, e)
        if self._ros is not None:
            try:
                self._ros.terminate()
            except Exception as e:
                log.warning("ros terminate failed: %s", e)

        with self._lock:
            for fp in (self._state_fp, self._command_fp,
                       self._haptic_fp, self._haptic_idx_fp):
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
            self._state_fp = None
            self._command_fp = None
            self._haptic_fp = None
            self._haptic_idx_fp = None

        log.info("RobotRecorder closed (%d state samples, %d command samples, %d haptic bytes)",
                 self.samples, self.command_samples, self.haptic_byte_offset)
        return self.samples

    def _on_state_message(self, msg: dict) -> None:
        recv_wall_ns = time.time_ns()
        try:
            stamp = msg.get("header", {}).get("stamp", {}) or {}
            secs = int(stamp.get("secs", 0))
            nsecs = int(stamp.get("nsecs", 0))
            sensor_ts_ns = secs * 1_000_000_000 + nsecs

            hand_states = msg.get("hand_states", []) or []
            sensor_states = msg.get("sensor_states", []) or []

            if (self.hand_index >= len(hand_states)
                    or self.hand_index >= len(sensor_states)):
                self._warn_throttled(
                    "hand_index %d out of range (got %d hand_states, %d sensor_states)",
                    self.hand_index, len(hand_states), len(sensor_states),
                )
                return

            hs = hand_states[self.hand_index]
            ss = sensor_states[self.hand_index]

            finger_states = ss.get("finger_sensor_states", []) or []
            if len(finger_states) != self.expected_n_fingers:
                self._warn_throttled(
                    "expected %d finger_sensor_states, got %d",
                    self.expected_n_fingers, len(finger_states),
                )

            haptic_floats: list[float] = []
            per_finger_n_vectors: list[int] = []
            for fs in finger_states:
                raw_force = fs.get("raw_force", []) or []
                per_finger_n_vectors.append(len(raw_force))
                for v in raw_force:
                    haptic_floats.append(float(v.get("x", 0.0)))
                    haptic_floats.append(float(v.get("y", 0.0)))
                    haptic_floats.append(float(v.get("z", 0.0)))

            if self._n_vectors_per_finger is None and per_finger_n_vectors:
                self._n_vectors_per_finger = per_finger_n_vectors[0]
                if not all(n == self._n_vectors_per_finger
                           for n in per_finger_n_vectors):
                    log.warning("Inconsistent per-finger taxel counts: %s",
                                per_finger_n_vectors)
                else:
                    log.info("Haptic shape locked: %d fingers × %d vectors × 3 = %d floats/sample",
                             len(per_finger_n_vectors), self._n_vectors_per_finger,
                             len(per_finger_n_vectors) * self._n_vectors_per_finger * 3)
            elif self._n_vectors_per_finger is not None:
                if any(n != self._n_vectors_per_finger
                       for n in per_finger_n_vectors):
                    self._warn_throttled(
                        "haptic shape changed mid-session: %s (expected %d/finger)",
                        per_finger_n_vectors, self._n_vectors_per_finger,
                    )

            buf = (struct.pack(f"<{len(haptic_floats)}f", *haptic_floats)
                   if haptic_floats else b"")

            state_record = {
                "sample_index": self.samples,
                "robot_timestamp_ns": recv_wall_ns,
                "sensor_timestamp_ns": sensor_ts_ns,
                "hand_id": (msg.get("hand_id") or [None])[self.hand_index]
                            if msg.get("hand_id") else None,
                "hand_name": (msg.get("hand_name") or [None])[self.hand_index]
                              if msg.get("hand_name") else None,
                "hand_type": (msg.get("hand_type") or [None])[self.hand_index]
                              if msg.get("hand_type") else None,
                "joint_name": hs.get("name", []),
                "position": hs.get("position", []),
                "effort": hs.get("effort", []),
                "temperature": hs.get("temperature", []),
                "error_code": hs.get("error_code", []),
                "finger_locations": [fs.get("location", "") for fs in finger_states],
                "finger_calc_temperature": [fs.get("calc_temperature", 0)
                                             for fs in finger_states],
            }

            haptic_idx_record = {
                "sample_index": self.samples,
                "robot_timestamp_ns": recv_wall_ns,
                "sensor_timestamp_ns": sensor_ts_ns,
                "byte_offset": self.haptic_byte_offset,
                "n_floats": len(haptic_floats),
                "per_finger_n_vectors": per_finger_n_vectors,
            }

            with self._lock:
                if self._state_fp is None or self._haptic_fp is None:
                    return
                self._state_fp.write(json.dumps(state_record) + "\n")
                if buf:
                    self._haptic_fp.write(buf)
                self._haptic_idx_fp.write(json.dumps(haptic_idx_record) + "\n")
                self.haptic_byte_offset += len(buf)
                self.samples += 1

            self.last_msg_wall_ns = recv_wall_ns

        except Exception as e:
            self._warn_throttled("Error processing XHandStateArray: %s", e)

    def _on_command_message(self, msg: dict) -> None:
        recv_wall_ns = time.time_ns()
        try:
            stamp = msg.get("header", {}).get("stamp", {}) or {}
            secs = int(stamp.get("secs", 0))
            nsecs = int(stamp.get("nsecs", 0))
            sensor_ts_ns = secs * 1_000_000_000 + nsecs

            record = {
                "sample_index": self.command_samples,
                "robot_timestamp_ns": recv_wall_ns,
                "sensor_timestamp_ns": sensor_ts_ns,
                "hand_id": msg.get("hand_id"),
                "position": msg.get("position", []),
            }

            with self._lock:
                if self._command_fp is None:
                    return
                self._command_fp.write(json.dumps(record) + "\n")
                self.command_samples += 1
        except Exception as e:
            self._warn_throttled("Error processing XHandCommand: %s", e)

    def _warn_throttled(self, fmt: str, *args) -> None:
        now = time.monotonic()
        if now - self._last_warn_t > 1.0:
            log.warning(fmt, *args)
            self._last_warn_t = now

    def zero_sensors(self, events: EventLogger) -> bool:
        if (self._reset_service is None or self._ros is None
                or not self._ros.is_connected):
            log.error("Cannot zero sensors: rosbridge not connected")
            return False

        import roslibpy
        all_ok = True
        for sensor_id in self.SENSOR_IDS:
            req = roslibpy.ServiceRequest({"hand_id": self.HAND_ID, "sensor_id": sensor_id})
            try:
                done = threading.Event()
                result_holder = {"resp": None, "err": None}

                def _cb(resp, _h=result_holder, _d=done):
                    _h["resp"] = resp
                    _d.set()

                def _err(err, _h=result_holder, _d=done):
                    _h["err"] = err
                    _d.set()

                self._reset_service.call(req, _cb, _err)
                if not done.wait(timeout=2.0):
                    log.error("Reset sensor %d: timeout", sensor_id)
                    all_ok = False
                    continue
                if result_holder["err"] is not None:
                    log.error("Reset sensor %d: %s", sensor_id, result_holder["err"])
                    all_ok = False
                else:
                    log.info("Reset sensor %d: ok", sensor_id)
            except Exception as e:
                log.error("Reset sensor %d failed: %s", sensor_id, e)
                all_ok = False

        if all_ok:
            events.log("haptic_zeroed", sensor_ids=self.SENSOR_IDS, hand_id=self.HAND_ID)
        return all_ok


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    in_episode: bool = False
    episode_count: int = 0


def write_session_meta(paths: SessionPaths, state: SessionState,
                       cams: list[CameraRecorder],
                       robot: Optional[RobotRecorder],
                       config: dict, start_wall_ns: int,
                       n_events: int, n_robot: int) -> None:
    end_wall_ns = time.time_ns()
    meta = {
        "start_wall_ns": start_wall_ns,
        "end_wall_ns": end_wall_ns,
        "duration_s": (end_wall_ns - start_wall_ns) / 1e9,
        "cameras": [
            {"name": c.name, "frames_written": c.frames_written,
             "frames_dropped_writer": c.frames_dropped_writer}
            for c in cams
        ],
        "events": n_events,
        "episodes_completed": state.episode_count,
        "robot_samples": n_robot,
        "camera_enabled": len(cams) > 0,
        "robot_enabled": robot is not None,
        "config": config,
    }
    if robot is not None:
        meta["haptic"] = {
            "dtype": "float32",
            "byte_order": "little",
            "n_fingers": robot.expected_n_fingers,
            "n_vectors_per_finger": robot._n_vectors_per_finger,
            "components_per_vector": 3,
            "total_floats_per_sample": (
                robot.expected_n_fingers * robot._n_vectors_per_finger * 3
                if robot._n_vectors_per_finger else None
            ),
            "total_bytes_written": robot.haptic_byte_offset,
            "note": "raw_force is stored unreshaped — taxel ordering is not a clean grid",
        }
    with open(paths.meta, "w") as f:
        json.dump(meta, f, indent=2)


def open_cameras(camera_configs: list[dict], cameras_dir: Path,
                 container: str, warn_dropped_frames: bool) -> list[CameraRecorder]:
    """Open all enabled cameras. If any fails, close those already opened
    and re-raise — caller must handle cleanup of robot/events."""
    cams: list[CameraRecorder] = []
    try:
        for cam_cfg in camera_configs:
            if not cam_cfg.get("enabled", True):
                log.info("[%s] disabled in config, skipping", cam_cfg.get("name", "?"))
                continue
            camera_dir = cameras_dir / cam_cfg["name"]
            recorder = CameraRecorder(
                cam_cfg, camera_dir, container, warn_dropped_frames,
            )
            recorder.open()
            cams.append(recorder)
    except Exception:
        # Bail: close any cameras that opened so we don't leak file handles.
        for c in cams:
            try:
                c.close(write_metadata=False)
            except Exception:
                pass
        raise
    return cams


def initial_preview_index(cams: list[CameraRecorder]) -> Optional[int]:
    """Return the index of the first camera with preview: true in its config,
    or 0 if there are cameras but none explicitly preferred, or None if no
    cameras. Used as the starting view; Tab cycles to the next from there."""
    if not cams:
        return None
    for i, c in enumerate(cams):
        if c.cfg.get("preview", False):
            return i
    return 0


def run(args: argparse.Namespace, config: dict) -> int:
    output_dir = Path(config["recording"]["output_dir"]).expanduser().resolve()
    container = config["recording"]["container"]
    warn_dropped = config["recording"].get("warn_dropped_frames", True)

    paths = SessionPaths.for_session(output_dir, args.session_name)
    log.info("Session directory: %s", paths.root)

    cams: list[CameraRecorder] = []
    robot: Optional[RobotRecorder] = None
    events = EventLogger(paths.events)
    start_wall_ns = time.time_ns()

    try:
        if not args.no_camera:
            camera_configs = config.get("cameras", [])
            if not camera_configs:
                log.error("No cameras configured (expected list under 'cameras:'). "
                          "Use --no-camera to skip camera capture.")
                return 1
            cams = open_cameras(camera_configs, paths.cameras_dir, container, warn_dropped)
            if not cams:
                log.error("All cameras are disabled in config. "
                          "Enable at least one or use --no-camera.")
                return 1

        if not args.no_robot:
            robot = RobotRecorder(config.get("robot", {}), paths)
            robot.open()
    except Exception as e:
        log.error("Startup failed: %s", e)
        # Cleanup partials
        for c in cams:
            try:
                c.close(write_metadata=False)
            except Exception:
                pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        events.close()
        return 1

    preview_idx = initial_preview_index(cams)
    if preview_idx is not None:
        log.info("Preview: %s (Tab to cycle through %d cameras)",
                 cams[preview_idx].name, len(cams))
    else:
        log.info("No cameras to preview.")

    state = SessionState()
    timed_mode = args.episode_seconds is not None
    if timed_mode:
        log.info("Timed episode mode: %.1fs pre-roll → %.1fs episode → repeat. "
                 "Spacebar starts a cycle; ignored once a cycle is running.",
                 args.pre_episode_seconds, args.episode_seconds)

    # Phase machine for timed mode. In manual mode (timed_mode=False) we only
    # ever use "idle" / "recording" without a deadline.
    #   "idle"       — waiting for spacebar
    #   "countdown"  — between spacebar and episode_start (timed mode only)
    #   "recording"  — between episode_start and episode_end
    phase = "idle"
    phase_deadline_monotonic: Optional[float] = None

    keys_help = "SPACE = episode start/end"
    if timed_mode:
        keys_help = "SPACE = begin timed cycle (ignored during cycle)"
    if robot is not None:
        keys_help += ", z = zero haptic sensors"
    if preview_idx is not None and len(cams) > 1:
        keys_help += ", Tab = cycle preview"
    keys_help += ", q = quit"
    log.info("Keys: %s", keys_help)
    if robot is not None:
        log.info("⚠  Press 'z' to zero haptic sensors before recording any episodes.")

    stop_flag = {"stop": False}
    def _sigint(_sig, _frm):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _sigint)

    last_fps_t = time.monotonic()
    last_fps_count = 0
    fps_display = 0.0
    haptic_zeroed = False

    # Headless-mode periodic countdown logging — log one line per second while
    # in countdown or recording phases (preview-mode users see this in the
    # OpenCV overlay so we don't double-log there).
    headless = preview_idx is None
    last_countdown_log_s: Optional[int] = None

    # Choose key input mechanism. In headless mode we use raw terminal stdin;
    # in preview mode we read from the OpenCV window via cv2.waitKey().
    key_reader_ctx = TerminalKeyReader() if headless else None

    try:
        if key_reader_ctx is not None:
            key_reader_ctx.__enter__()

        while not stop_flag["stop"]:
            # Phase transitions driven by deadlines (timed mode only).
            if timed_mode and phase_deadline_monotonic is not None:
                now_mono = time.monotonic()
                if now_mono >= phase_deadline_monotonic:
                    if phase == "countdown":
                        # Pre-roll done → start episode
                        events.log("episode_start")
                        state.in_episode = True
                        phase = "recording"
                        phase_deadline_monotonic = now_mono + args.episode_seconds
                        last_countdown_log_s = None  # reset so first sec logs
                        log.info("episode_start logged (#%d) — running for %.1fs",
                                 state.episode_count, args.episode_seconds)
                    elif phase == "recording":
                        # Episode duration elapsed → end episode
                        events.log("episode_end")
                        state.episode_count += 1
                        state.in_episode = False
                        phase = "idle"
                        phase_deadline_monotonic = None
                        last_countdown_log_s = None
                        log.info("episode_end logged (auto, episode %d complete)",
                                 state.episode_count)

            # Headless periodic countdown log (per-second), only during timed phases.
            if (headless and timed_mode and phase != "idle"
                    and phase_deadline_monotonic is not None):
                remaining = max(0.0, phase_deadline_monotonic - time.monotonic())
                remaining_s_int = int(remaining)
                if last_countdown_log_s != remaining_s_int:
                    last_countdown_log_s = remaining_s_int
                    if phase == "countdown":
                        log.info("  pre-roll: %ds until episode_start", remaining_s_int)
                    elif phase == "recording":
                        log.info("  recording: %ds until episode_end", remaining_s_int)

            if preview_idx is not None:
                preview_cam = cams[preview_idx]
                preview_scale = float(preview_cam.cfg.get("preview_scale", 0.5))
                frame = preview_cam.get_preview_frame()
                if frame is not None:
                    if preview_scale != 1.0:
                        frame = cv2.resize(
                            frame, (0, 0), fx=preview_scale, fy=preview_scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        frame = frame.copy()

                    now = time.monotonic()
                    if now - last_fps_t >= 1.0:
                        fps_display = ((preview_cam.frames_grabbed - last_fps_count)
                                       / (now - last_fps_t))
                        last_fps_count = preview_cam.frames_grabbed
                        last_fps_t = now

                    # Phase-aware label + colour
                    if phase == "recording":
                        color = (0, 0, 255)  # red
                        if timed_mode and phase_deadline_monotonic is not None:
                            remaining = max(0.0, phase_deadline_monotonic - now)
                            label = f"REC ep #{state.episode_count}  {remaining:4.1f}s left"
                        else:
                            label = f"REC ep #{state.episode_count}"
                    elif phase == "countdown":
                        color = (0, 200, 255)  # amber
                        remaining = (max(0.0, phase_deadline_monotonic - now)
                                     if phase_deadline_monotonic is not None else 0.0)
                        label = f"starting ep #{state.episode_count} in {remaining:4.1f}s"
                    else:
                        color = (0, 200, 0)  # green
                        label = f"idle (next ep: #{state.episode_count})"

                    cam_tag = (f"[{preview_cam.name}]" if len(cams) == 1
                               else f"[{preview_cam.name} {preview_idx + 1}/{len(cams)}]")
                    overlay = f"{cam_tag} {label}  {fps_display:5.1f} fps"
                    if robot is not None and not haptic_zeroed:
                        overlay += "  [HAPTIC NOT ZEROED — press z]"
                    cv2.putText(frame, overlay,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("vla_capture", frame)

                key = cv2.waitKey(1) & 0xFF
            else:
                # Headless: read from terminal stdin with a short poll timeout.
                # 50ms keeps the deadline-check loop responsive enough for the
                # 1Hz countdown logging.
                assert key_reader_ctx is not None
                key = key_reader_ctx.read_key()

            if key == ord("q"):
                break
            if key == ord(" "):
                if timed_mode:
                    # Spacebar only valid in idle. Ignored otherwise.
                    if phase == "idle":
                        if robot is not None and not haptic_zeroed:
                            log.warning("Starting cycle but haptic sensors have not been "
                                        "zeroed. Press 'z' first for clean data.")
                        phase = "countdown"
                        phase_deadline_monotonic = (time.monotonic()
                                                    + args.pre_episode_seconds)
                        log.info("Cycle armed — %.1fs pre-roll, then %.1fs episode",
                                 args.pre_episode_seconds, args.episode_seconds)
                    else:
                        log.info("Spacebar ignored (cycle in progress, phase=%s)", phase)
                else:
                    # Manual mode: spacebar toggles episode_start/end directly.
                    if not state.in_episode:
                        if robot is not None and not haptic_zeroed:
                            log.warning("Starting episode but haptic sensors have not been "
                                        "zeroed. Press 'z' first for clean data.")
                        events.log("episode_start")
                        state.in_episode = True
                        phase = "recording"
                        log.info("episode_start logged (#%d)", state.episode_count)
                    else:
                        events.log("episode_end")
                        state.episode_count += 1
                        state.in_episode = False
                        phase = "idle"
                        log.info("episode_end logged")
            if key == ord("z") and robot is not None:
                if phase != "idle":
                    log.warning("Refusing to zero sensors — phase=%s "
                                "(zero only while idle, between cycles).", phase)
                else:
                    log.info("Zeroing haptic sensors...")
                    if robot.zero_sensors(events):
                        haptic_zeroed = True
                        log.info("Haptic sensors zeroed.")
                    else:
                        log.error("Sensor zeroing failed — see errors above.")
            if key == 9 and preview_idx is not None and len(cams) > 1:
                # Tab — cycle to next camera. Reset fps tracking so the readout
                # isn't computed off the previous camera's frame counter.
                preview_idx = (preview_idx + 1) % len(cams)
                last_fps_t = time.monotonic()
                last_fps_count = cams[preview_idx].frames_grabbed
                fps_display = 0.0
                log.info("Preview switched to: %s", cams[preview_idx].name)

    finally:
        if key_reader_ctx is not None:
            try:
                key_reader_ctx.__exit__(None, None, None)
            except Exception:
                pass

        if state.in_episode:
            log.info("Logging episode_end before exit (in-progress episode)...")
            events.log("episode_end")
            state.episode_count += 1

        for c in cams:
            c.close(write_metadata=True)
        n_robot = robot.close() if robot is not None else 0
        n_events = events.close()

        write_session_meta(paths, state, cams, robot, config,
                           start_wall_ns, n_events, n_robot)
        log.info("Session saved → %s", paths.root)
        log.info("  %d cameras, %d events, %d episodes, %d robot samples",
                 len(cams), n_events, state.episode_count, n_robot)
        for c in cams:
            log.info("  %s: %d frames", c.name, c.frames_written)

        if preview_idx is not None:
            cv2.destroyAllWindows()

    return 0


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable all camera capture")
    parser.add_argument("--no-robot", action="store_true",
                        help="Disable robot state capture")
    parser.add_argument("--list-controls", action="store_true",
                        help="List v4l2 controls + formats for a configured camera and exit "
                             "(use --camera to pick which; defaults to first enabled)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Name of camera to act on (for --list-controls)")
    parser.add_argument("--session-name", type=str, default=None,
                        help="Override the auto-generated session_YYYYMMDD_HHMMSS dir name. "
                             "Use this to coordinate with capture_cameras.py running on "
                             "another machine: both pass the same --session-name and write "
                             "to identically-named directories that can be merged in post.")
    parser.add_argument("--episode-seconds", type=float, default=None,
                        help="Fixed-length episodes. With this set, spacebar triggers a "
                             "pre-roll countdown then auto-logs episode_start, runs for "
                             "this many seconds, then auto-logs episode_end. Spacebar is "
                             "ignored while a cycle is in flight. Without it, episodes are "
                             "fully manual.")
    parser.add_argument("--pre-episode-seconds", type=float, default=5.0,
                        help="Pre-roll countdown before each auto-timed episode "
                             "(default 5). Only used with --episode-seconds.")
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        return 1

    config = load_config(args.config)

    if args.list_controls:
        camera_configs = config.get("cameras", [])
        if not camera_configs:
            log.error("No cameras configured")
            return 1
        if args.camera:
            target = next((c for c in camera_configs if c.get("name") == args.camera), None)
            if target is None:
                log.error("Camera '%s' not found in config. Available: %s",
                          args.camera,
                          ", ".join(c.get("name", "?") for c in camera_configs))
                return 1
        else:
            target = next((c for c in camera_configs if c.get("enabled", True)),
                          camera_configs[0])
            log.info("No --camera specified, using first enabled: %s",
                     target.get("name"))
        list_v4l2_controls(target["device"])
        return 0

    if args.no_camera and args.no_robot:
        log.error("Both --no-camera and --no-robot set — nothing to do.")
        return 1

    if args.episode_seconds is not None:
        if args.episode_seconds <= 0:
            log.error("--episode-seconds must be > 0")
            return 1
        if args.pre_episode_seconds < 0:
            log.error("--pre-episode-seconds must be >= 0")
            return 1

    return run(args, config)


if __name__ == "__main__":
    sys.exit(main())