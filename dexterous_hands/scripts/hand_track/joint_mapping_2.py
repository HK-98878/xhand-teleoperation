import numpy as np
import math
import roslibpy
import sys, os
sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
# Wrist = 0
# Thumb:  1(CMC/j0), 2(MCP/j1), 3(IP/j2),  4(tip)
# Index:  5(MCP/j1), 6(PIP/j2), 7(DIP/j3),  8(tip)
# Middle: 9(MCP/j1), 10(PIP/j2),11(DIP/j3), 12(tip)
# Ring:  13(MCP/j1), 14(PIP/j2),15(DIP/j3), 16(tip)
# Pinky: 17(MCP/j1), 18(PIP/j2),19(DIP/j3), 20(tip)

FINGER_INDICES = {
    "index":  [5,  6,  7,  8],   # j1, j2, j3, tip
    "middle": [9,  10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

FINGER_NAMES = ["index", "middle", "ring", "pinky"]

# ---------------------------------------------------------------------------
# Joint names (must match driver exactly)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
]

# ---------------------------------------------------------------------------
# Joint limits (radians)
# ---------------------------------------------------------------------------
JOINT_LIMITS = {
    "thumb_bend_joint":  (0.0,   1.57),
    "thumb_rota_joint1": (-1.05, 1.57),
    "thumb_rota_joint2": (0.0,   1.57),
    "index_bend_joint":  (-0.09, 0.3),
    "index_joint1":      (0.0,   1.92),
    "index_joint2":      (0.0,   1.92),
    "mid_joint1":        (0.0,   1.92),
    "mid_joint2":        (0.0,   1.92),
    "ring_joint1":       (0.0,   1.92),
    "ring_joint2":       (0.0,   1.92),
    "pinky_joint1":      (0.0,   1.92),
    "pinky_joint2":      (0.0,   1.92),
}

# ---------------------------------------------------------------------------
# BLEND WEIGHTS — adjust these if finger motion doesn't feel right
# Each finger has two output joints (joint1=base, joint2=tip curl).
# The three input measures are:
#   mcp_flex: angle between palm forward direction and proximal bone
#   pip_flex: angle between proximal and middle bone
#   dip_flex: angle between middle and distal bone
# ---------------------------------------------------------------------------
FLEX_BLEND = {
    # joint1 weights: (mcp_flex, pip_flex, dip_flex)
    # joint2 weights: (mcp_flex, pip_flex, dip_flex)
    "index":  dict(j1=(0.70, 0.20, 0.10), j2=(0.10, 0.45, 0.45)),
    "middle": dict(j1=(0.70, 0.20, 0.10), j2=(0.10, 0.45, 0.45)),
    "ring":   dict(j1=(0.70, 0.20, 0.10), j2=(0.10, 0.45, 0.45)),
    "pinky":  dict(j1=(0.70, 0.20, 0.10), j2=(0.10, 0.45, 0.45)),
}

# Remapping input range for finger flexion → robot joint range
FLEX_REMAP = {
    "index":  dict(j1=(0.02, 1.05), j2=(0.02, 1.35)),
    "middle": dict(j1=(0.02, 1.15), j2=(0.02, 1.45)),
    "ring":   dict(j1=(0.02, 1.25), j2=(0.02, 1.45)),
    "pinky":  dict(j1=(0.02, 1.15), j2=(0.02, 1.45)),
}

# Spread (abduction) remapping per finger
# in_lo/in_hi are the expected human spread signal range (radians)
SPREAD_REMAP = {
    "index":  dict(in_lo=-0.05, in_hi=0.13, out_lo=-0.04, out_hi=0.28),
}

# Neutral spread correction scale per finger (damps raw spread signal)
SPREAD_SCALE = {
    "index":  0.45,
}
SPREAD_DEADZONE = {
    "index":  0.035,
}

# Thumb blend weights
# Input signals:
#   thumb_splay:      in-plane sweep of metacarpal relative to across-palm axis
#   thumb_opposition: out-of-plane lift of thumb base
#   thumb_root_flex:  angle between metacarpal and proximal bone
#   thumb_mid_flex:   angle between proximal and distal bone
#   thumb_closure:    angle between metacarpal direction and overall thumb ray
THUMB_BLEND = {
    # bend  weights: (splay, closure)
    "bend":  (0.70, 0.30),
    # rota1 weights: (opposition, root_flex, closure)
    "rota1": (0.20, 0.65, 0.15),
    # rota2 weights: (root_flex, mid_flex)
    "rota2": (0.25, 0.75),
}
THUMB_REMAP = {
    "bend":  (0.40, 1.0),
    "rota1": (0.0, 0.8),
    "rota2": (0.05, 1.45),
}

# ---------------------------------------------------------------------------
# EMA Smoother
# ---------------------------------------------------------------------------
class EMASmoother:
    def __init__(self, alpha=0.2, num_joints=12):
        self.alpha = alpha
        self.state = None

    def update(self, new_angles):
        if self.state is None:
            self.state = np.array(new_angles, dtype=float)
        else:
            self.state = self.alpha * np.array(new_angles) + (1 - self.alpha) * self.state
        return self.state.copy()


# ---------------------------------------------------------------------------
# Math helpers (ported from colleague's script)
# ---------------------------------------------------------------------------
def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else v / n

def safe_angle(a, b, eps=1e-8):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return float(math.acos(max(-1.0, min(1.0, c))))

def safe_asin(x):
    return float(math.asin(max(-1.0, min(1.0, float(x)))))

def project_to_plane(v, normal):
    return v - np.dot(v, normal) * normal

def signed_angle_on_axis(a, b, axis):
    axis = normalize(axis)
    a_p = normalize(project_to_plane(a, axis))
    b_p = normalize(project_to_plane(b, axis))
    if np.linalg.norm(a_p) < 1e-8 or np.linalg.norm(b_p) < 1e-8:
        return 0.0
    x = float(np.dot(a_p, b_p))
    y = float(np.dot(axis, np.cross(a_p, b_p)))
    return float(math.atan2(y, x))

def remap_clip(x, in_lo, in_hi, out_lo, out_hi):
    if abs(in_hi - in_lo) < 1e-8:
        return float(0.5 * (out_lo + out_hi))
    t = max(0.0, min(1.0, (x - in_lo) / (in_hi - in_lo)))
    return float(out_lo + t * (out_hi - out_lo))

def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


# ---------------------------------------------------------------------------
# Landmark extraction from MediaPipe
# ---------------------------------------------------------------------------
def extract_landmarks(hand_landmarks):
    """Convert MediaPipe landmarks to a list of np.ndarray (x, y, z)."""
    return [np.array([l.x, l.y, l.z], dtype=np.float64) for l in hand_landmarks]


def compute_palm_frame(lm):
    """
    Compute palm coordinate frame from MediaPipe landmarks.
    Returns (across, forward, normal) unit vectors.
    across:  index MCP -> pinky MCP (left to right across palm)
    forward: wrist -> middle MCP (finger direction)
    normal:  out of palm
    """
    wrist      = lm[0]
    index_mcp  = lm[5]
    middle_mcp = lm[9]
    pinky_mcp  = lm[17]

    across  = normalize(index_mcp - pinky_mcp)
    forward = normalize(middle_mcp - wrist)
    normal  = normalize(np.cross(across, forward))
    # Reorthogonalise forward against normal
    forward = normalize(np.cross(normal, across))
    return across, forward, normal


# ---------------------------------------------------------------------------
# Finger flex measures
# ---------------------------------------------------------------------------
def finger_flex_measures(forward, across, j1, j2, j3, tip):
    """
    Compute three raw flexion angles along a finger.
    Returns (mcp_flex, pip_flex, dip_flex) in radians.
    """
    v12 = j2  - j1
    v23 = j3  - j2
    v34 = tip - j3

    # Project into sagittal plane (remove across-finger component)
    forward_sag = normalize(project_to_plane(forward, across))
    prox_sag    = normalize(project_to_plane(v12, across))
    mid_sag     = normalize(project_to_plane(v23, across))
    dist_sag    = normalize(project_to_plane(v34, across))

    mcp_flex = safe_angle(forward_sag, prox_sag)
    pip_flex = safe_angle(prox_sag,    mid_sag)
    dip_flex = safe_angle(mid_sag,     dist_sag)

    return mcp_flex, pip_flex, dip_flex


def apply_flex_blend(finger, mcp_flex, pip_flex, dip_flex):
    """
    Blend raw flex measures into two joint commands using FLEX_BLEND weights.
    Returns (joint1_val, joint2_val) before remapping.
    """
    w1 = FLEX_BLEND[finger]["j1"]
    w2 = FLEX_BLEND[finger]["j2"]
    j1_raw = w1[0]*mcp_flex + w1[1]*pip_flex + w1[2]*dip_flex
    j2_raw = w2[0]*mcp_flex + w2[1]*pip_flex + w2[2]*dip_flex
    return j1_raw, j2_raw


# ---------------------------------------------------------------------------
# Spread (abduction) measures
# ---------------------------------------------------------------------------
def compute_spread(lm, across, normal):
    """
    Compute raw lateral spread signal for index and middle fingers.
    Uses the direction of the proximal bone projected onto the palm plane.
    """
    spread = {}
    for finger, (j1_i, j2_i, _, _) in FINGER_INDICES.items():
        if finger not in ("index", "middle"):
            continue
        j1 = lm[j1_i]
        j2 = lm[j2_i]
        base_dir = normalize(project_to_plane(j2 - j1, normal))
        spread[finger] = float(np.dot(base_dir, across))

    # Relative spread: index relative to middle
    spread["index"]  = spread["index"]  - spread["middle"]
    spread["middle"] = 0.0  # middle is the reference

    return spread


def correct_spread(finger, value, neutral):
    corrected = value - neutral.get(finger, 0.0)
    scale = SPREAD_SCALE.get(finger, 1.0)
    dz    = SPREAD_DEADZONE.get(finger, 0.0)
    corrected *= scale
    if abs(corrected) < dz:
        corrected = 0.0
    return corrected


# ---------------------------------------------------------------------------
# Thumb
# ---------------------------------------------------------------------------
def compute_thumb(lm, across, forward, normal):
    """
    Compute thumb joint commands from MediaPipe landmarks.
    Returns (bend, rota1, rota2) raw values before clamping.
    """
    t0, t1, t2, t3, t4 = lm[1], lm[2], lm[3], lm[4], lm[4]
    # Note: MediaPipe thumb has CMC(1), MCP(2), IP(3), tip(4)
    tv01 = t1 - lm[1]   # CMC -> MCP
    tv01 = lm[2] - lm[1]
    tv12 = lm[3] - lm[2]
    tv23 = lm[4] - lm[3]

    thumb_base = normalize(tv01 if np.linalg.norm(tv01) > 1e-8 else tv12)
    thumb_long = normalize((lm[4] - lm[1]) if np.linalg.norm(lm[4] - lm[1]) > 1e-8 else tv12)

    thumb_base_in_plane = normalize(project_to_plane(thumb_base, normal))
    if np.linalg.norm(thumb_base_in_plane) < 1e-8:
        thumb_base_in_plane = normalize(project_to_plane(thumb_long, normal))

    thumb_splay      = safe_angle(across, thumb_base_in_plane)
    thumb_opposition = abs(safe_asin(float(np.dot(thumb_base, normal))))
    thumb_root_flex  = safe_angle(tv01, tv12)
    thumb_mid_flex   = safe_angle(tv12, tv23)
    thumb_closure    = safe_angle(thumb_base, thumb_long)

    wb = THUMB_BLEND["bend"]
    wr = THUMB_BLEND["rota1"]
    wm = THUMB_BLEND["rota2"]

    bend_measure  = wb[0]*thumb_splay      + wb[1]*thumb_closure
    rota1_measure = wr[0]*thumb_opposition + wr[1]*thumb_root_flex + wr[2]*thumb_closure
    rota2_measure = wm[0]*thumb_root_flex  + wm[1]*thumb_mid_flex

    lo, hi = JOINT_LIMITS["thumb_bend_joint"]
    bend  = remap_clip(bend_measure,  *THUMB_REMAP["bend"],  lo, hi)
    lo, hi = JOINT_LIMITS["thumb_rota_joint1"]
    rota1 = remap_clip(rota1_measure, *THUMB_REMAP["rota1"], lo, hi)
    lo, hi = JOINT_LIMITS["thumb_rota_joint2"]
    rota2 = remap_clip(rota2_measure, *THUMB_REMAP["rota2"], lo, hi)

    return bend, rota1, rota2


# ---------------------------------------------------------------------------
# Main retargeting function
# ---------------------------------------------------------------------------
def landmarks_to_angles(hand_landmarks, neutral_spread):
    """
    Convert MediaPipe hand landmarks to 12 joint angles.
    neutral_spread: dict of {finger: float} for spread bias correction,
                   updated externally during open-hand poses.
    """
    lm = extract_landmarks(hand_landmarks)
    across, forward, normal = compute_palm_frame(lm)

    angles = {}

    # Thumb
    bend, rota1, rota2 = compute_thumb(lm, across, forward, normal)
    angles["thumb_bend_joint"]  = clip(bend,  *JOINT_LIMITS["thumb_bend_joint"])
    angles["thumb_rota_joint1"] = clip(rota1, *JOINT_LIMITS["thumb_rota_joint1"])
    angles["thumb_rota_joint2"] = clip(rota2, *JOINT_LIMITS["thumb_rota_joint2"])

    # Spread for index and middle
    raw_spread = compute_spread(lm, across, normal)

    # Fingers
    for finger in FINGER_NAMES:
        j1_i, j2_i, j3_i, tip_i = FINGER_INDICES[finger]
        mcp_flex, pip_flex, dip_flex = finger_flex_measures(
            forward, across, lm[j1_i], lm[j2_i], lm[j3_i], lm[tip_i]
        )

        j1_raw, j2_raw = apply_flex_blend(finger, mcp_flex, pip_flex, dip_flex)

        prefix = "mid" if finger == "middle" else finger
        j1_name = f"{prefix}_joint1"
        j2_name = f"{prefix}_joint2"

        angles[j1_name] = clip(
            remap_clip(j1_raw, *FLEX_REMAP[finger]["j1"], *JOINT_LIMITS[j1_name]),
            *JOINT_LIMITS[j1_name]
        )
        angles[j2_name] = clip(
            remap_clip(j2_raw, *FLEX_REMAP[finger]["j2"], *JOINT_LIMITS[j2_name]),
            *JOINT_LIMITS[j2_name]
        )

        # Spread for index only
        if finger == "index":
            spread_val = correct_spread(finger, raw_spread.get(finger, 0.0), neutral_spread)
            cfg = SPREAD_REMAP["index"]
            angles["index_bend_joint"] = clip(
                remap_clip(spread_val, cfg["in_lo"], cfg["in_hi"], cfg["out_lo"], cfg["out_hi"]),
                *JOINT_LIMITS["index_bend_joint"]
            )

    return [angles[name] for name in JOINT_NAMES]


# ---------------------------------------------------------------------------
# Neutral spread calibration
# ---------------------------------------------------------------------------
class NeutralSpreadCalibrator:
    """
    Tracks the neutral spread during open-hand poses using EMA.
    Call update() every frame; it self-calibrates when the hand is open.
    """
    def __init__(self, alpha=0.05):
        self.neutral = {"index": 0.0}
        self.filters = {"index": EMASmoother(alpha=alpha, num_joints=1)}

    def update(self, raw_spread, is_open):
        if not is_open:
            return
        for finger, value in raw_spread.items():
            if finger in self.filters:
                self.neutral[finger] = float(self.filters[finger].update(np.array([value]))[0])


# ---------------------------------------------------------------------------
# ROS publisher
# ---------------------------------------------------------------------------
class XHandPublisher:
    def __init__(self, host='localhost', port=9090, hand_id=0,
                 kp=10.0, ki=0.0, kd=0.5, effort_limit=10.0, mode=1):
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()

        self.publisher = roslibpy.Topic(
            self.client,
            '/xhand_control/xhand_command',
            'xhand_control_ros/XHandCommand'
        )

        self.smoother    = EMASmoother(alpha=0.2, num_joints=12)
        self.calibrator  = NeutralSpreadCalibrator(alpha=0.05)
        self.hand_id     = hand_id
        n = len(JOINT_NAMES)
        self.kp           = [kp]           * n
        self.ki           = [ki]           * n
        self.kd           = [kd]           * n
        self.effort_limit = [effort_limit] * n
        self.mode         = [mode]         * n

    def send(self, hand_landmarks):
        if not self.client.is_connected:
            return

        lm = extract_landmarks(hand_landmarks)
        across, forward, normal = compute_palm_frame(lm)
        raw_spread = compute_spread(lm, across, normal)

        # Estimate open hand: index finger fairly straight
        j1_i, j2_i, j3_i, tip_i = FINGER_INDICES["index"]
        mcp, pip, dip = finger_flex_measures(forward, across, lm[j1_i], lm[j2_i], lm[j3_i], lm[tip_i])
        is_open = (mcp + pip + dip) < 0.6

        self.calibrator.update(raw_spread, is_open)

        raw     = landmarks_to_angles(hand_landmarks, self.calibrator.neutral)
        smoothed = self.smoother.update(raw)

        msg = {
            'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'hand_id':      self.hand_id,
            'name':         JOINT_NAMES,
            'position':     list(smoothed),
            'kp':           self.kp,
            'ki':           self.ki,
            'kd':           self.kd,
            'effort_limit': self.effort_limit,
            'mode':         self.mode,
        }
        self.publisher.publish(roslibpy.Message(msg))

    def shutdown(self):
        self.publisher.unadvertise()
        self.client.terminate()