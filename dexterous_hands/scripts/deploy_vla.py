#!/usr/bin/env python3
from __future__ import annotations
"""
deploy_vla.py — deploy a trained SmolVLA policy on the XHand robot.

Observation keys expected by the model:
  observation.state               — 12D joint positions
  observation.images.top          — ugreen_main camera  (1920×1080)
  observation.images.c920_side    — C920 wrist camera   (1920×1080)
  observation.images.haptic       — haptic tactile image (120×580)

Controls:
  p  — pause / resume policy (holds last position while paused)
  r  — reset hand to home position
  q  — quit (resets to home before exit)
"""

import argparse
import builtins
import csv
import datetime
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import roslibpy
import torch

# Run with the lerobot venv active:
#   source /home/hsken/Documents/smolvla-testing/lerobot/.venv/bin/activate
#   pip install roslibpy  # one-time, if not already present
#   python deploy_vla.py --checkpoint <path> --task "<task>"

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

from hand_recording.parse_recordings import haptic_sample_to_image
from hand_track.utils import XHandPublisher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HAPTIC_UPSCALE = 10
HAPTIC_IMG_H = 120   # 12 rows × upscale
HAPTIC_IMG_W = 580   # 58 cols × upscale  (5 fingers × 10 cols + 4 × 2 separators)

CAMERA_UGREEN = "/dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0"
CAMERA_C920   = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_77807C6F-video-index0"

DEFAULT_HOME = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
N_JOINTS = 12

HAPTIC_SENSOR_IDS = [17, 18, 19, 20, 21]
HAPTIC_RESET_SERVICE      = "/xhand_control/reset_hand_sensor"
HAPTIC_RESET_SERVICE_TYPE = "xhand_control_ros/ResetSensor"


# ---------------------------------------------------------------------------
# Thread-safe buffers
# ---------------------------------------------------------------------------
@dataclass
class LatestFrame:
    bgr: np.ndarray | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LatestRobotState:
    """Populated by a single ROS subscriber for both joint state and haptic."""
    joint_positions: np.ndarray | None = None   # (12,) float32
    haptic_image: np.ndarray | None = None      # (120, 580, 3) uint8 RGB
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------
def camera_capture_thread(
    device_path: str, name: str, buf: LatestFrame, stop_event: threading.Event
) -> None:
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print(f"[ERROR] {name}: could not open {device_path}", flush=True)
        return
    print(f"[{name}] camera opened", flush=True)
    while not stop_event.is_set():
        ok, frame = cap.read()
        if ok:
            with buf.lock:
                buf.bgr = frame
    cap.release()


# ---------------------------------------------------------------------------
# ROS state subscriber (joint positions + haptic in one callback)
# ---------------------------------------------------------------------------
def ros_state_thread(
    ros_client: roslibpy.Ros,
    state_buf: LatestRobotState,
    use_haptic: bool,
    stop_event: threading.Event,
) -> None:
    hand_index = 0

    def on_state(msg: dict) -> None:
        hand_states = msg.get("hand_states", []) or []
        sensor_states = msg.get("sensor_states", []) or []

        new_positions = None
        new_haptic = None

        if hand_index < len(hand_states):
            raw_pos = hand_states[hand_index].get("position", []) or []
            if len(raw_pos) == N_JOINTS:
                new_positions = np.array([float(p) for p in raw_pos], dtype=np.float32)

        if use_haptic and hand_index < len(sensor_states):
            finger_states = sensor_states[hand_index].get("finger_sensor_states", []) or []
            if finger_states:
                n_fingers = len(finger_states)
                n_vecs = len(finger_states[0].get("raw_force", []))
                if n_vecs > 0:
                    sample = np.zeros((n_fingers, n_vecs, 3), dtype=np.float32)
                    for fi, fs in enumerate(finger_states):
                        for vi, v in enumerate(fs.get("raw_force", [])):
                            sample[fi, vi, 0] = float(v.get("x", 0.0))
                            sample[fi, vi, 1] = float(v.get("y", 0.0))
                            sample[fi, vi, 2] = float(v.get("z", 0.0))
                    new_haptic = haptic_sample_to_image(sample, upscale=HAPTIC_UPSCALE)

        with state_buf.lock:
            if new_positions is not None:
                state_buf.joint_positions = new_positions
            if new_haptic is not None:
                state_buf.haptic_image = new_haptic

    topic = roslibpy.Topic(
        ros_client, "/xhand_control/xhand_state",
        "xhand_control_ros/XHandStateArray", throttle_rate=0,
    )
    topic.subscribe(on_state)
    stop_event.wait()
    topic.unsubscribe()


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → RGB float32 CHW in [0, 1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)


def _rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 HWC → float32 CHW in [0, 1]."""
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)


def build_obs(
    frame_top: np.ndarray,
    frame_c920: np.ndarray,
    state_buf: LatestRobotState,
    task: str,
    use_haptic: bool,
) -> dict:
    with state_buf.lock:
        joint_pos = (
            state_buf.joint_positions.copy()
            if state_buf.joint_positions is not None
            else np.zeros(N_JOINTS, dtype=np.float32)
        )
        haptic_img = (
            state_buf.haptic_image.copy()
            if state_buf.haptic_image is not None
            else None
        )

    if use_haptic and haptic_img is not None:
        haptic_t = _rgb_to_tensor(haptic_img)
    else:
        haptic_t = torch.full((3, HAPTIC_IMG_H, HAPTIC_IMG_W), 128.0 / 255.0)

    return {
        "observation.state":              torch.tensor(joint_pos, dtype=torch.float32),
        "observation.images.top":         _bgr_to_tensor(frame_top),
        "observation.images.c920_side":   _bgr_to_tensor(frame_c920),
        "observation.images.haptic":      haptic_t,
        "task":                           task,
    }


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------
def inference_thread(
    policy: SmolVLAPolicy,
    preprocessor: DataProcessorPipeline,
    postprocessor: DataProcessorPipeline,
    frame_top_buf: LatestFrame,
    frame_c920_buf: LatestFrame,
    state_buf: LatestRobotState,
    action_deque: deque,
    deque_lock: threading.Lock,
    pause_event: threading.Event,
    stop_event: threading.Event,
    task: str,
    use_haptic: bool,
    control_hz: float,
) -> None:
    n_chunk = policy.config.n_action_steps
    refill_threshold = n_chunk // 2
    budget_ms = 1000.0 / control_hz

    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.1)
            continue

        with deque_lock:
            remaining = len(action_deque)
        if remaining > refill_threshold:
            time.sleep(0.005)
            continue

        # Wait for first frames to arrive before inferencing
        with frame_top_buf.lock:
            ready_top = frame_top_buf.bgr is not None
        with frame_c920_buf.lock:
            ready_c920 = frame_c920_buf.bgr is not None
        if not (ready_top and ready_c920):
            time.sleep(0.05)
            continue

        with frame_top_buf.lock:
            frame_top = frame_top_buf.bgr.copy()
        with frame_c920_buf.lock:
            frame_c920 = frame_c920_buf.bgr.copy()

        obs = build_obs(frame_top, frame_c920, state_buf, task, use_haptic)
        batch = preprocessor(obs)

        t0 = time.perf_counter()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)  # (1, n_chunk, 12)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if infer_ms > budget_ms:
            print(
                f"[WARN] inference took {infer_ms:.0f}ms "
                f"(>{budget_ms:.0f}ms per-tick budget at {control_hz:.0f}Hz)",
                flush=True,
            )

        actions_unnorm = postprocessor(actions)                     # (1, n_chunk, 12)
        actions_np = actions_unnorm.squeeze(0).cpu().numpy()        # (n_chunk, 12)

        if not np.isfinite(actions_np).all():
            print("[WARN] NaN/Inf in action chunk — discarding", flush=True)
            continue

        with deque_lock:
            action_deque.extend(actions_np)


# ---------------------------------------------------------------------------
# Command thread (fixed-rate 30 Hz)
# ---------------------------------------------------------------------------
def command_thread(
    publisher: XHandPublisher,
    action_deque: deque,
    deque_lock: threading.Lock,
    pause_event: threading.Event,
    stop_event: threading.Event,
    last_action: list,          # mutable single-element list for cross-thread state
    control_hz: float,
    state_buf: LatestRobotState | None = None,
    log_writer: csv.writer | None = None,
) -> None:
    period = 1.0 / control_hz
    next_tick = time.perf_counter()
    stall_count = 0

    while not stop_event.is_set():
        now = time.perf_counter()
        dt = next_tick - now
        if dt > 0:
            time.sleep(dt)
        next_tick += period
        if next_tick < time.perf_counter() - period:
            next_tick = time.perf_counter() + period

        # Mode-3 position controller holds the last commanded position on the
        # hardware side, so we don't need to resend while paused. Resending
        # would fight reset commands sent concurrently from the main thread.
        if pause_event.is_set():
            continue

        with deque_lock:
            action = action_deque.popleft() if action_deque else None

        if action is None:
            stall_count += 1
            if stall_count == 1:
                print("[WARN] inference stall — holding position (action deque empty)", flush=True)
            if last_action[0] is not None:
                publisher.send(last_action[0].tolist())
            continue

        if stall_count > 0:
            print(f"[INFO] resumed after {stall_count} stalled tick(s)", flush=True)
            stall_count = 0

        clipped = np.clip(
            action,
            XHandPublisher.JOINT_LIMITS[:, 0],
            XHandPublisher.JOINT_LIMITS[:, 1],
        )
        diff = action - clipped
        if diff.max() > 1e-4 or diff.max() < -0.2:
            # only report significant clip below zero - we get frequent small oscillations below 0
            idxs = np.where(diff > 1e-4 | diff < -0.2)[0] 
            names = [XHandPublisher.JOINT_NAMES[i] for i in idxs]
            print(
                f"[WARN] clipped joints {names}: "
                f"raw={action[idxs].round(3).tolist()}, "
                f"clipped={clipped[idxs].round(3).tolist()}",
                flush=True,
            )

        publisher.send(clipped.tolist())
        last_action[0] = clipped

        if log_writer is not None and state_buf is not None:
            with state_buf.lock:
                current_state = (
                    state_buf.joint_positions.copy()
                    if state_buf.joint_positions is not None
                    else np.zeros(N_JOINTS, dtype=np.float32)
                )
            log_writer.writerow(
                [time.time()] + current_state.tolist() + clipped.tolist()
            )


# ---------------------------------------------------------------------------
# Haptic sensor zeroing
# ---------------------------------------------------------------------------
def zero_haptic_sensors(ros_client: roslibpy.Ros) -> None:
    """Call /xhand_control/reset_hand_sensor for each finger sensor."""
    if not ros_client.is_connected:
        print("[WARN] cannot zero sensors: ROS bridge not connected", flush=True)
        return
    service = roslibpy.Service(ros_client, HAPTIC_RESET_SERVICE, HAPTIC_RESET_SERVICE_TYPE)
    all_ok = True
    for sensor_id in HAPTIC_SENSOR_IDS:
        done = threading.Event()
        result: dict = {}

        def _cb(resp, _r=result, _d=done):
            _r["resp"] = resp
            _d.set()

        def _err(err, _r=result, _d=done):
            _r["err"] = err
            _d.set()

        service.call(roslibpy.ServiceRequest({"hand_id": 0, "sensor_id": sensor_id}), _cb, _err)
        if not done.wait(timeout=2.0):
            print(f"[WARN] zero sensor {sensor_id}: timeout", flush=True)
            all_ok = False
        elif "err" in result:
            print(f"[WARN] zero sensor {sensor_id}: {result['err']}", flush=True)
            all_ok = False
        else:
            print(f"[zero] sensor {sensor_id}: ok", flush=True)
    if all_ok:
        print("[zero] all haptic sensors zeroed", flush=True)


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------
def _publish_raw_command(
    ros_client: roslibpy.Ros,
    position: list,
    kp: float,
    ki: float,
    kd: float,
    effort_limit: float,
    mode: int,
) -> None:
    topic = roslibpy.Topic(
        ros_client,
        "/xhand_control/xhand_command",
        "xhand_control_ros/XHandCommand",
    )
    topic.publish(
        roslibpy.Message({
            "header": {"stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
            "hand_id":      0,
            "name":         XHandPublisher.JOINT_NAMES,
            "position":     [float(p) for p in position],
            "kp":           [float(kp)] * N_JOINTS,
            "ki":           [float(ki)] * N_JOINTS,
            "kd":           [float(kd)] * N_JOINTS,
            "effort_limit": [float(effort_limit)] * N_JOINTS,
            "mode":         [int(mode)] * N_JOINTS,
        })
    )


def do_reset(
    ros_client: roslibpy.Ros,
    home_position: np.ndarray,
    policy: SmolVLAPolicy,
    action_deque: deque,
    deque_lock: threading.Lock,
) -> None:
    """Move to home with soft gains, disengage, then clear the action queue."""
    home = np.clip(
        home_position,
        XHandPublisher.JOINT_LIMITS[:, 0],
        XHandPublisher.JOINT_LIMITS[:, 1],
    )
    print("[reset] moving to home...", flush=True)
    _publish_raw_command(ros_client, home.tolist(), kp=25, ki=12000, kd=0, effort_limit=250, mode=3)
    time.sleep(2.0)
    _publish_raw_command(ros_client, home.tolist(), kp=25, ki=12000, kd=0, effort_limit=250, mode=0)
    print("[reset] done — hand disengaged", flush=True)

    with deque_lock:
        action_deque.clear()
    policy.reset()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy a SmolVLA policy on the XHand robot via roslibpy."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to local LeRobot checkpoint directory")
    parser.add_argument("--task", required=True,
                        help="Language task description passed to the model")
    parser.add_argument("--ros-host", default="localhost")
    parser.add_argument("--ros-port", type=int, default=9090)
    parser.add_argument("--device", default="cuda",
                        help="Torch device for inference (cuda or cpu)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override n_action_steps from the checkpoint config")
    parser.add_argument("--no-haptic", action="store_true",
                        help="Feed a neutral haptic image instead of live sensor data")
    parser.add_argument("--home", type=float, nargs=N_JOINTS, default=DEFAULT_HOME,
                        metavar="RAD",
                        help=f"Home joint positions in radians ({N_JOINTS} values)")
    parser.add_argument("--control-hz", type=float, default=30.0,
                        help="Command loop frequency in Hz (default: 30)")
    parser.add_argument("--log-file", default=None,
                        help="CSV file for joint state/target logging "
                             "(default: joint_log_<timestamp>.csv)")
    args = parser.parse_args()

    use_haptic = not args.no_haptic
    home_pos = np.array(args.home, dtype=np.float32)

    # ------------------------------------------------------------------
    # Joint log file
    # ------------------------------------------------------------------
    log_path = args.log_file or (
        f"joint_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    _log_file = open(log_path, "w", newline="", buffering=1)
    log_writer = csv.writer(_log_file)
    log_writer.writerow(
        ["timestamp"]
        + [f"state_{i}" for i in range(N_JOINTS)]
        + [f"target_{i}" for i in range(N_JOINTS)]
    )
    print(f"[init] logging joint state/target to {log_path}", flush=True)

    # ------------------------------------------------------------------
    # Load policy and processors
    # ------------------------------------------------------------------
    print(f"[init] loading policy from {args.checkpoint}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.eval()

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    if args.chunk_size is not None:
        policy.config.n_action_steps = args.chunk_size
    print(f"[init] n_action_steps={policy.config.n_action_steps}", flush=True)

    # ------------------------------------------------------------------
    # ROS connection (XHandPublisher owns the roslibpy.Ros client)
    # ------------------------------------------------------------------
    print(f"[init] connecting ROS bridge {args.ros_host}:{args.ros_port}", flush=True)
    publisher = XHandPublisher(host=args.ros_host, port=args.ros_port)
    ros_client = publisher.client  # reuse same WebSocket for subscriptions

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    frame_top_buf  = LatestFrame()
    frame_c920_buf = LatestFrame()
    robot_state    = LatestRobotState()
    action_deque: deque = deque()
    deque_lock     = threading.Lock()
    last_action    = [None]     # single-element list — mutable reference
    pause_event    = threading.Event()
    pause_event.set()   # start paused — press p to begin
    stop_event     = threading.Event()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------
    threads = [
        threading.Thread(
            target=camera_capture_thread, daemon=True, name="cam_top",
            args=(CAMERA_UGREEN, "ugreen_main", frame_top_buf, stop_event),
        ),
        threading.Thread(
            target=camera_capture_thread, daemon=True, name="cam_c920",
            args=(CAMERA_C920, "c920_wrist", frame_c920_buf, stop_event),
        ),
        threading.Thread(
            target=ros_state_thread, daemon=True, name="ros_state",
            args=(ros_client, robot_state, use_haptic, stop_event),
        ),
        threading.Thread(
            target=inference_thread, daemon=True, name="inference",
            args=(
                policy, preprocessor, postprocessor,
                frame_top_buf, frame_c920_buf, robot_state,
                action_deque, deque_lock,
                pause_event, stop_event,
                args.task, use_haptic, args.control_hz,
            ),
        ),
        threading.Thread(
            target=command_thread, daemon=True, name="command",
            args=(
                publisher, action_deque, deque_lock,
                pause_event, stop_event, last_action, args.control_hz,
                robot_state, log_writer,
            ),
        ),
    ]
    for t in threads:
        t.start()

    # ------------------------------------------------------------------
    # Keyboard handler (main thread, raw terminal mode for single-keypress)
    # ------------------------------------------------------------------
    print("\nControls: [p] pause/resume  [r] reset to home  [z] zero haptic sensors  [q] quit", flush=True)
    print("Starting PAUSED — press p to begin\n", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # In raw mode \n no longer implies \r, causing a staircase from all threads.
    # Patch builtins.print globally so every thread gets correct line endings.
    _orig_print = builtins.print
    def _raw_print(*args, **kwargs):
        end = kwargs.pop("end", "\n")
        builtins.print = _orig_print          # avoid recursion
        _orig_print(*args, end=end.replace("\n", "\r\n"), **kwargs)
        builtins.print = _raw_print           # re-apply patch
    try:
        tty.setraw(fd)
        builtins.print = _raw_print
        while not stop_event.is_set():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key in ("q", "Q", "\x03"):   # q or Ctrl-C
                print("\r[ctrl] quitting...", flush=True)
                stop_event.set()
            elif key in ("p", "P"):
                if pause_event.is_set():
                    print("\r[ctrl] resuming policy", flush=True)
                    with deque_lock:
                        action_deque.clear()
                    policy.reset()
                    pause_event.clear()
                else:
                    print("\r[ctrl] paused", flush=True)
                    pause_event.set()
            elif key in ("r", "R"):
                print("\r[ctrl] resetting...", flush=True)
                was_paused = pause_event.is_set()
                pause_event.set()
                do_reset(ros_client, home_pos, policy, action_deque, deque_lock)
                if not was_paused:
                    pause_event.clear()
            elif key in ("z", "Z"):
                print("\r[ctrl] zeroing haptic sensors...", flush=True)
                zero_haptic_sensors(ros_client)
    finally:
        builtins.print = _orig_print
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    stop_event.set()
    print("[shutdown] resetting hand to home...", flush=True)
    try:
        do_reset(ros_client, home_pos, policy, action_deque, deque_lock)
    except Exception as exc:
        print(f"[shutdown] reset failed: {exc}", flush=True)
    for t in threads:
        t.join(timeout=2.0)
    publisher.shutdown()
    _log_file.close()
    print(f"[shutdown] joint log saved to {log_path}", flush=True)
    print("[shutdown] done", flush=True)


if __name__ == "__main__":
    main()
