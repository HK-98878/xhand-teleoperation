"""
Leap Motion 2 source using Ultraleap's Gemini Python bindings.

Setup (Ubuntu):
    1. Install Gemini Hand Tracking Software (>= 5.17) from
       https://leap2.ultraleap.com/downloads/  (the .deb package).
    2. Start the daemon:    sudo leapd
       Or enable as service: sudo systemctl enable --now ultraleap-hand-tracking-service
    3. Clone the bindings:  git clone https://github.com/ultraleap/leapc-python-bindings
    4. From that repo, in your venv:
         pip install -r requirements.txt
         pip install -e leapc-python-api
    5. Verify with:        python examples/tracking_event_example.py

Coordinate system reminder (from Ultraleap docs):
    Right-handed Cartesian, origin at top-center of the device.
    +X along camera baseline, +Y up, +Z toward the user.
    Native units are MILLIMETERS — we convert to meters here.

The Gemini API exposes each finger as a `digit` containing 4 bones
(metacarpal, proximal, intermediate, distal). Each bone has `prev_joint`
and `next_joint` xyz. Note: the thumb's "metacarpal" bone is zero-length
in the Leap model — the thumb effectively starts at the proximal bone.
We map this into our 21-keypoint convention below.
"""
from __future__ import annotations

import threading
import time
from queue import Queue, Empty, Full
from typing import Optional

import numpy as np

from .keypoints import HandKeypoints, KP

try:
    import leap  # type: ignore
    from leap import datatypes as ldt  # type: ignore
    # Verify a real binding loaded (not a stub) by checking for a known symbol.
    _HAVE_LEAP = hasattr(leap, "Listener") and hasattr(leap, "Connection")
except ImportError:
    _HAVE_LEAP = False
except Exception:
    _HAVE_LEAP = False


def _vec3(v) -> np.ndarray:
    """Convert a Leap Vector to a numpy array in METERS (Leap is mm)."""
    return np.array([v.x, v.y, v.z], dtype=np.float32) * 0.001


_BaseListener = leap.Listener if _HAVE_LEAP else object


class _TrackingListener(_BaseListener):
    """Internal Leap event listener that pushes HandKeypoints into a queue."""

    def __init__(self, queue: Queue, prefer_hand: str = "right", verbose: bool = False):
        super().__init__()
        self._queue = queue
        self._prefer_hand = prefer_hand  # "left", "right", or "any"
        self._last_ts = 0.0
        self._verbose = verbose
        self._tracking_event_count = 0
        self._hand_event_count = 0

    def on_connection_event(self, event):
        print("[leap] connection event: connected")

    def on_connection_lost_event(self, event):
        print("[leap] connection LOST")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
            print(f"[leap] device event: serial={info.serial}")
        except Exception as e:
            print(f"[leap] device event (info unavailable: {e})")

    def on_tracking_mode_event(self, event):
        print(f"[leap] tracking mode set: {event.current_tracking_mode}")

    def on_tracking_event(self, event):
        self._tracking_event_count += 1
        if self._verbose and self._tracking_event_count % 60 == 1:
            n_hands = len(event.hands) if event.hands else 0
            print(f"[leap] tracking event #{self._tracking_event_count}, hands={n_hands}")

        if not event.hands:
            return

        self._hand_event_count += 1
        if self._verbose and self._hand_event_count == 1:
            print(f"[leap] FIRST hand detected after {self._tracking_event_count} tracking events")

        # Pick the preferred hand if multiple are present.
        chosen = None
        for h in event.hands:
            # Mirror the reference example's robust string-based check rather
            # than enum equality (the enum import path varies across versions).
            chirality = "left" if str(h.type) == "HandType.Left" else "right"
            if self._prefer_hand == "any" or chirality == self._prefer_hand:
                chosen = h
                break
        if chosen is None:
            if self._verbose and self._hand_event_count <= 3:
                print(f"[leap] hand visible but doesn't match preferred='{self._prefer_hand}'")
            return

        kp = self._hand_to_keypoints(chosen)
        # Drop frames if consumer is slow; teleop wants newest, not all.
        try:
            self._queue.put_nowait(kp)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(kp)
            except Full:
                pass

    @staticmethod
    def _hand_to_keypoints(hand) -> HandKeypoints:
        pts = np.zeros((21, 3), dtype=np.float32)

        # Wrist comes from the palm/arm. Leap exposes `arm.next_joint` as the
        # wrist and `palm.position` as the palm centroid; we use arm.next_joint.
        pts[KP.WRIST] = _vec3(hand.arm.next_joint)

        # digits[0]=thumb, [1]=index, [2]=middle, [3]=ring, [4]=pinky
        # bones: 0=metacarpal, 1=proximal, 2=intermediate, 3=distal

        # ---- Thumb (only 3 phalanges anatomically; Leap pads with zero-length metacarpal) ----
        thumb = hand.digits[0]
        # CMC = base of metacarpal (anatomically the trapezium joint)
        pts[KP.THUMB_CMC] = _vec3(thumb.metacarpal.prev_joint)
        # MCP = base of proximal phalanx
        pts[KP.THUMB_MCP] = _vec3(thumb.proximal.prev_joint)
        # IP = base of distal phalanx (since intermediate is zero-length on thumb)
        pts[KP.THUMB_IP] = _vec3(thumb.distal.prev_joint)
        pts[KP.THUMB_TIP] = _vec3(thumb.distal.next_joint)

        # ---- Other four fingers ----
        finger_kp_bases = {
            1: (KP.INDEX_MCP, KP.INDEX_PIP, KP.INDEX_DIP, KP.INDEX_TIP),
            2: (KP.MIDDLE_MCP, KP.MIDDLE_PIP, KP.MIDDLE_DIP, KP.MIDDLE_TIP),
            3: (KP.RING_MCP, KP.RING_PIP, KP.RING_DIP, KP.RING_TIP),
            4: (KP.PINKY_MCP, KP.PINKY_PIP, KP.PINKY_DIP, KP.PINKY_TIP),
        }
        for d_idx, (mcp, pip, dip, tip) in finger_kp_bases.items():
            digit = hand.digits[d_idx]
            pts[mcp] = _vec3(digit.proximal.prev_joint)   # MCP
            pts[pip] = _vec3(digit.intermediate.prev_joint)  # PIP
            pts[dip] = _vec3(digit.distal.prev_joint)     # DIP
            pts[tip] = _vec3(digit.distal.next_joint)     # TIP

        is_left = (str(hand.type) == "HandType.Left")
        palm_normal = np.array(
            [hand.palm.normal.x, hand.palm.normal.y, hand.palm.normal.z], dtype=np.float32
        )
        palm_dir = np.array(
            [hand.palm.direction.x, hand.palm.direction.y, hand.palm.direction.z], dtype=np.float32
        )

        # Leap doesn't give a per-frame confidence in the public API, so we
        # synthesize one from `grab_strength` not being NaN; otherwise 1.0.
        confidence = 1.0
        try:
            if hand.grab_strength != hand.grab_strength:  # NaN check
                confidence = 0.0
        except Exception:
            pass

        return HandKeypoints(
            points=pts,
            is_left=is_left,
            timestamp=time.monotonic(),
            confidence=confidence,
            palm_normal=palm_normal,
            palm_direction=palm_dir,
        )


class LeapSource:
    """High-level Leap Motion 2 hand tracking source.

    Usage:
        src = LeapSource(prefer_hand="right")
        src.start()
        while running:
            kp = src.get(timeout=0.05)
            if kp is not None: ...
        src.stop()

    Set `mock=True` to run without hardware (for software-only dev). It will
    emit a synthetic, gently-moving hand so you can wire up the rest of the
    pipeline before plugging in the camera.
    """

    def __init__(
        self,
        prefer_hand: str = "right",
        queue_size: int = 2,
        mock: bool = False,
        hmd_mode: bool = False,
        verbose: bool = False,
    ):
        self.prefer_hand = prefer_hand
        self.mock = mock
        self.hmd_mode = hmd_mode
        self.verbose = verbose
        self._queue: Queue[HandKeypoints] = Queue(maxsize=queue_size)
        self._connection: Optional[object] = None
        self._listener: Optional[object] = None
        self._mock_thread: Optional[threading.Thread] = None
        self._leap_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    def start(self):
        if self.mock:
            self._stop.clear()
            self._mock_thread = threading.Thread(target=self._mock_loop, daemon=True)
            self._mock_thread.start()
            print("[leap] running in MOCK mode (no hardware)")
            return

        if not _HAVE_LEAP:
            raise RuntimeError(
                "Ultraleap Python bindings not installed. Either install them "
                "(see leap_source.py docstring) or run with mock=True."
            )

        self._stop.clear()
        self._ready.clear()
        # The Ultraleap binding spins up its message-pump thread only while we
        # are inside `with connection.open(): ...`. Outside the `with` block,
        # tracking events never fire. So we run that whole pattern on a
        # dedicated thread that lives for the source's lifetime.
        self._leap_thread = threading.Thread(target=self._leap_loop, daemon=True)
        self._leap_thread.start()
        # Wait briefly for connection to come up so subsequent calls succeed.
        if not self._ready.wait(timeout=3.0):
            print("[leap] WARNING: connection did not signal ready within 3s "
                  "(continuing anyway)")

    def _leap_loop(self):
        """Runs on a background thread for the lifetime of the source."""
        try:
            self._listener = _TrackingListener(
                self._queue, prefer_hand=self.prefer_hand, verbose=self.verbose
            )
            self._connection = leap.Connection()
            self._connection.add_listener(self._listener)

            with self._connection.open():
                # Set tracking mode once we're inside the with-block (the pump
                # thread is now running and will accept policy changes).
                try:
                    mode = leap.TrackingMode.HMD if self.hmd_mode else leap.TrackingMode.Desktop
                    self._connection.set_tracking_mode(mode)
                except Exception as e:
                    print(f"[leap] could not set tracking mode: {e}")

                self._ready.set()
                # Block here until stop() is called. The listener's
                # on_tracking_event runs on the binding's pump thread and
                # populates self._queue meanwhile.
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception as e:
            print(f"[leap] fatal error in leap thread: {e}")
            self._ready.set()  # unblock start()

    def stop(self):
        self._stop.set()
        if self._mock_thread:
            self._mock_thread.join(timeout=1.0)
        if self._leap_thread:
            self._leap_thread.join(timeout=2.0)
        # Connection is closed automatically by the `with` block exiting.
        self._connection = None

    def get(self, timeout: float = 0.05) -> Optional[HandKeypoints]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    # ---------------- MOCK MODE -----------------
    def _mock_loop(self):
        """Generate a synthetic right hand that opens and closes slowly."""
        t0 = time.monotonic()
        # Reference open-hand pose (right hand, palm down, fingers along +Z),
        # in meters in a "wrist-local" frame, then we'll lift it into world.
        base = np.array([
            [0.000, 0.000, 0.000],   # wrist
            [0.020, 0.005, 0.020],   # thumb_cmc
            [0.040, 0.010, 0.040],   # thumb_mcp
            [0.055, 0.015, 0.060],   # thumb_ip
            [0.065, 0.020, 0.080],   # thumb_tip
            [0.030, 0.000, 0.080],   # index_mcp
            [0.032, 0.005, 0.115],   # index_pip
            [0.033, 0.008, 0.140],   # index_dip
            [0.034, 0.010, 0.160],   # index_tip
            [0.010, 0.000, 0.085],   # middle_mcp
            [0.011, 0.005, 0.123],   # middle_pip
            [0.011, 0.008, 0.150],   # middle_dip
            [0.011, 0.010, 0.172],   # middle_tip
            [-0.010, 0.000, 0.082],  # ring_mcp
            [-0.011, 0.005, 0.115],  # ring_pip
            [-0.011, 0.008, 0.140],  # ring_dip
            [-0.011, 0.010, 0.160],  # ring_tip
            [-0.030, 0.000, 0.075],  # pinky_mcp
            [-0.031, 0.005, 0.103],  # pinky_pip
            [-0.031, 0.008, 0.122],  # pinky_dip
            [-0.031, 0.010, 0.138],  # pinky_tip
        ], dtype=np.float32)

        finger_chains = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }

        while not self._stop.is_set():
            t = time.monotonic() - t0
            # 0..1 closing factor (sinusoidal)
            close = 0.5 * (1.0 - np.cos(0.6 * t))
            pts = base.copy()
            # Curl each non-thumb finger by rotating PIP/DIP/TIP around MCP about local x-axis.
            for name in ("index", "middle", "ring", "pinky"):
                ids = finger_chains[name]
                mcp = pts[ids[0]].copy()
                # Rotation angle grows along the chain
                for i, jid in enumerate(ids[1:], start=1):
                    angle = close * (0.6 + 0.3 * i)  # rad
                    rel = pts[jid] - mcp
                    c, s = np.cos(angle), np.sin(angle)
                    # rotate in (z, y) plane (curl forward/down)
                    new = np.array([rel[0], rel[1] * c - rel[2] * s, rel[1] * s + rel[2] * c])
                    pts[jid] = mcp + new
            # Thumb: simple opposition motion
            for jid in finger_chains["thumb"][1:]:
                pts[jid, 0] -= 0.015 * close

            # Lift hand to a reasonable position above the camera and add slow drift.
            pts[:, 1] += 0.20 + 0.02 * np.sin(0.4 * t)
            pts[:, 0] += 0.03 * np.sin(0.3 * t)

            kp = HandKeypoints(
                points=pts,
                is_left=False,
                timestamp=time.monotonic(),
                confidence=1.0,
                palm_normal=np.array([0, 1, 0], dtype=np.float32),
                palm_direction=np.array([0, 0, 1], dtype=np.float32),
            )
            try:
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                self._queue.put_nowait(kp)
            except Full:
                pass

            time.sleep(1.0 / 60.0)