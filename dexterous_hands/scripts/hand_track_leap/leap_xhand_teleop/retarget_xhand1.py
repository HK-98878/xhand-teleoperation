"""
XHAND1 retargeter
=================

Maps 21 hand keypoints to the 12 actuated DOFs of the RobotEra XHAND1.

THIS MODULE IS THE MOST OPINIONATED PART OF THE PIPELINE. If the robot's
fingers behave wrong, this is almost certainly where to look.

XHAND1 DOF layout used here (12 total)
--------------------------------------
We adopt the following joint ordering and signs. Override `JOINT_LIMITS_RAD`
and `JOINT_DIRECTIONS` if your URDF uses different conventions.

    Index   Name                 Range (rad)        Description
    -----   --------             -------------      ---------------------------
    0       thumb_pinch_flex     [-0.20,  1.50]     Thumb curl toward a pinch
    1       thumb_cross_palm     [-0.20,  1.40]     Thumbpad rotation: facing-out -> facing-palm
    2       thumb_mid_bend       [ 0.00,  1.40]     Thumb IP curl (distal vs proximal)
    3       index_abd            [-0.30,  0.30]     Index splay (+ toward thumb)
    4       index_mcp_flex       [ 0.00,  1.60]     Index MCP curl
    5       index_pip_flex       [ 0.00,  1.70]     Index PIP curl (couples to DIP physically)
    6       middle_mcp_flex      [ 0.00,  1.60]
    7       middle_pip_flex      [ 0.00,  1.70]
    8       ring_mcp_flex        [ 0.00,  1.60]
    9       ring_pip_flex        [ 0.00,  1.70]
    10      pinky_mcp_flex       [ 0.00,  1.60]
    11      pinky_pip_flex       [ 0.00,  1.70]

NOTE on coupled DIP: XHAND1 (like most production hands) couples the DIP to
the PIP through a tendon, so we don't expose a DIP DOF -- the PIP angle
already implies the DIP angle. We compute pip_flex from the human's combined
PIP+DIP curl so the visible pose matches.

Conventions
-----------
Input keypoints are in METERS in Leap world frame. We first rotate them into
the wrist-local frame defined in `keypoints.py:HandKeypoints.wrist_local()`:
    +x : palm forward (wrist -> middle MCP)
    +y : palm normal (out of palm)
    +z : x cross y (toward thumb side, for right hand)

In this frame:
    - "MCP flex" = angle the proximal phalanx makes below the palm plane (-y)
    - "PIP flex" = angle of intermediate phalanx relative to proximal
    - "DIP flex" = angle of distal phalanx relative to intermediate
      (we add PIP+DIP and clip into the PIP DOF range; tendon coupling will
       distribute it on the robot)
    - "abduction" = angle the proximal phalanx makes in the palm plane (z direction)
    - thumb is treated separately (see _retarget_thumb), with three DOFs:
        pinch_flex (toward a pinch), cross_palm (thumbpad twist), mid_bend (IP curl).

Tuning
------
1. If a joint moves the wrong direction, flip its sign in `JOINT_DIRECTIONS`.
2. If a joint is the right direction but too weak/strong, adjust
   `HUMAN_TO_ROBOT_GAIN[i]` for that index.
3. Thumb-specific tuning: `_THUMB_PINCH_FLEX_GAIN`, `_THUMB_CROSS_PALM_GAIN`,
   `_THUMB_MID_BEND_GAIN`. The pinch_flex gain especially -- a relaxed human
   thumb has a narrower swing than the robot's full pinch range.
4. If `thumb_pinch_flex` reads non-zero with your hand flat and thumb extended
   to the side, recalibrate `_THUMB_REST_AXIS_RIGHT/LEFT`.

This module is deliberately pure-numpy and stateless so you can:
  - swap it for a learned retargeter (e.g. dex-retargeting / pinocchio IK) later
  - unit-test it on synthetic poses
  - run it on a separate thread without locking concerns
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .keypoints import KP, HandKeypoints


XHAND1_JOINT_NAMES = [
    "thumb_pinch_flex",   # 0  curl toward a pinch with index/middle
    "thumb_cross_palm",   # 1  thumbpad rotation: facing-out -> facing-palm
    "thumb_mid_bend",     # 2  IP joint curl (distal vs proximal)
    "index_abd",          # 3  Index splay (+ toward thumb)
    "index_mcp_flex",     # 4
    "index_pip_flex",     # 5  PIP+DIP combined (tendon coupling)
    "middle_mcp_flex",    # 6
    "middle_pip_flex",    # 7
    "ring_mcp_flex",      # 8
    "ring_pip_flex",      # 9
    "pinky_mcp_flex",     # 10
    "pinky_pip_flex",     # 11
]

# Conservative defaults. CHECK YOUR URDF and override these.
JOINT_LIMITS_RAD = np.array([
    [-0.50, 1.80],   # 0  thumb_pinch_flex   - generous range; tighten later if URDF requires
    [-0.20, 1.40],   # 1  thumb_cross_palm   - thumbpad twist
    [ 0.00, 1.40],   # 2  thumb_mid_bend     - IP curl
    [-0.30, 0.30],   # 3  index_abd
    [ 0.00, 1.60],   # 4  index_mcp_flex
    [ 0.00, 1.70],   # 5  index_pip_flex
    [ 0.00, 1.60],   # 6
    [ 0.00, 1.70],   # 7
    [ 0.00, 1.60],   # 8
    [ 0.00, 1.70],   # 9
    [ 0.00, 1.60],   # 10
    [ 0.00, 1.70],   # 11
], dtype=np.float32)

# +1 means "human flexion increases robot joint value", -1 inverts.
JOINT_DIRECTIONS = np.ones(12, dtype=np.float32)

# Per-joint gain applied to the raw human angle before clipping. Lets you
# match the geometric range of the human's motion to the robot's. 1.0 = no scaling.
HUMAN_TO_ROBOT_GAIN = np.ones(12, dtype=np.float32)

# Thumb tuning constants -- isolated because the thumb is hardest to get right.
# Apply these gains BEFORE clipping. Bump up if the robot under-reaches at
# full human range; cut down if it saturates the limits too easily.
_THUMB_PINCH_FLEX_GAIN = 1.0   # multiplies thumb_pinch_flex (joint 0)
_THUMB_CROSS_PALM_GAIN = 1.0   # multiplies thumb_cross_palm (joint 1)
_THUMB_MID_BEND_GAIN   = 1.0   # multiplies thumb_mid_bend   (joint 2)

# Anatomical rest axis for the thumb proximal phalanx in WRIST-LOCAL frame,
# right hand. When the hand is flat and relaxed with the thumb extended out
# to the side, the proximal phalanx (THUMB_MCP -> THUMB_IP) points roughly
# forward and outward to the thumb side -- approximately 45 deg between +x
# (palm forward) and +z (thumb side).
#
# pinch_flex is measured AS THE ROTATION FROM THIS REST AXIS TOWARD +x, so
# a relaxed thumb reads ~0 and a thumb folded into a pinch reads ~+1.0+ rad.
#
# To CALIBRATE for a particular user: hold the hand flat with thumb fully
# extended out to the side (NOT pinching, NOT crossing palm), look at the
# diagnostics panel, and tweak this vector until thumb_pinch_flex reads ~0.
_THUMB_REST_AXIS_RIGHT = np.array([0.7071, 0.0, 0.7071], dtype=np.float32)
_THUMB_REST_AXIS_LEFT  = np.array([0.7071, 0.0, -0.7071], dtype=np.float32)


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in radians between two vectors."""
    ua, ub = _safe_unit(a), _safe_unit(b)
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


def _signed_angle_in_plane(v_from: np.ndarray, v_to: np.ndarray, plane_normal: np.ndarray) -> float:
    """Signed angle from v_from to v_to, with sign determined by plane_normal.

    Right-hand rule: positive when rotation is counterclockwise viewed from +plane_normal.
    Vectors are projected onto the plane first.
    """
    n = _safe_unit(plane_normal)
    a = v_from - np.dot(v_from, n) * n
    b = v_to - np.dot(v_to, n) * n
    a, b = _safe_unit(a), _safe_unit(b)
    cross = np.cross(a, b)
    sin_ = float(np.dot(cross, n))
    cos_ = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arctan2(sin_, cos_))


@dataclass
class RetargetDebug:
    """Optional intermediate values exposed for the viewer / logger."""
    raw_angles: np.ndarray          # before gain/clip, length 12
    clipped_angles: np.ndarray      # after gain/clip
    finger_mcp_angles: dict         # per-finger MCP raw angle for diagnostics
    finger_pip_angles: dict
    finger_dip_angles: dict
    thumb_opposition: float
    thumb_pitch: float
    thumb_flex: float


class XHand1Retargeter:
    """Stateless retargeter from HandKeypoints to 12-DOF XHAND1 commands."""

    def __init__(
        self,
        joint_limits: np.ndarray = None,
        directions: np.ndarray = None,
        gains: np.ndarray = None,
    ):
        self.joint_limits = joint_limits if joint_limits is not None else JOINT_LIMITS_RAD.copy()
        self.directions = directions if directions is not None else JOINT_DIRECTIONS.copy()
        self.gains = gains if gains is not None else HUMAN_TO_ROBOT_GAIN.copy()

    def __call__(self, kp: HandKeypoints) -> tuple[np.ndarray, RetargetDebug]:
        # Work in wrist-local frame so the math is independent of where the
        # hand is in the camera's view.
        local = kp.wrist_local()  # (21, 3)

        x_axis = np.array([1.0, 0.0, 0.0])  # palm forward
        y_axis = np.array([0.0, 1.0, 0.0])  # palm normal (out of palm)
        z_axis = np.array([0.0, 0.0, 1.0])  # toward thumb side

        # ---- Per-finger MCP/PIP/DIP angles ----
        finger_kps = {
            "index":  (KP.INDEX_MCP, KP.INDEX_PIP, KP.INDEX_DIP, KP.INDEX_TIP),
            "middle": (KP.MIDDLE_MCP, KP.MIDDLE_PIP, KP.MIDDLE_DIP, KP.MIDDLE_TIP),
            "ring":   (KP.RING_MCP, KP.RING_PIP, KP.RING_DIP, KP.RING_TIP),
            "pinky":  (KP.PINKY_MCP, KP.PINKY_PIP, KP.PINKY_DIP, KP.PINKY_TIP),
        }
        mcp_angles, pip_angles, dip_angles, abd_angles = {}, {}, {}, {}
        for name, (mcp_i, pip_i, dip_i, tip_i) in finger_kps.items():
            mcp = local[mcp_i]
            pip = local[pip_i]
            dip = local[dip_i]
            tip = local[tip_i]

            v_prox = pip - mcp     # proximal phalanx
            v_inter = dip - pip    # intermediate phalanx
            v_dist = tip - dip     # distal phalanx

            # MCP flex: angle proximal phalanx makes below palm plane.
            # In wrist-local frame, palm plane is y=0; "below" means -y.
            # We measure the signed angle from +x to v_prox in the (x, -y) plane.
            mcp_flex = _signed_angle_in_plane(x_axis, v_prox, plane_normal=z_axis)
            # In our convention, curling DOWN (toward -y) is positive flex; the
            # signed_angle_in_plane around +z gives that sign correctly for a right hand.
            # (Sanity-check on the robot; if inverted, flip JOINT_DIRECTIONS for that joint.)
            mcp_angles[name] = mcp_flex

            # MCP abduction: signed angle in palm plane (xz). Used only for index.
            abd = _signed_angle_in_plane(x_axis, v_prox, plane_normal=y_axis)
            abd_angles[name] = abd

            # PIP and DIP: angle BETWEEN successive phalanges (always positive curl).
            pip_flex = _angle_between(v_prox, v_inter)
            dip_flex = _angle_between(v_inter, v_dist)
            pip_angles[name] = pip_flex
            dip_angles[name] = dip_flex

        # ---- Thumb (special case) ----
        thumb_pinch, thumb_cross, thumb_mid = self._retarget_thumb(local, kp.is_left)

        # ---- Pack into 12-DOF vector ----
        raw = np.zeros(12, dtype=np.float32)
        raw[0] = thumb_pinch * _THUMB_PINCH_FLEX_GAIN
        raw[1] = thumb_cross * _THUMB_CROSS_PALM_GAIN
        raw[2] = thumb_mid   * _THUMB_MID_BEND_GAIN
        raw[3] = abd_angles["index"]
        raw[4] = mcp_angles["index"]
        raw[5] = pip_angles["index"] + dip_angles["index"]   # tendon coupling
        raw[6] = mcp_angles["middle"]
        raw[7] = pip_angles["middle"] + dip_angles["middle"]
        raw[8] = mcp_angles["ring"]
        raw[9] = pip_angles["ring"] + dip_angles["ring"]
        raw[10] = mcp_angles["pinky"]
        raw[11] = pip_angles["pinky"] + dip_angles["pinky"]

        # Apply direction & gain, then clip to robot limits.
        scaled = raw * self.directions * self.gains
        clipped = np.clip(scaled, self.joint_limits[:, 0], self.joint_limits[:, 1])

        debug = RetargetDebug(
            raw_angles=raw,
            clipped_angles=clipped,
            finger_mcp_angles=mcp_angles,
            finger_pip_angles=pip_angles,
            finger_dip_angles=dip_angles,
            thumb_opposition=thumb_pinch,   # field kept for API stability
            thumb_pitch=thumb_cross,        # field kept for API stability
            thumb_flex=thumb_mid,           # field kept for API stability
        )
        return clipped, debug

    @staticmethod
    def _retarget_thumb(local: np.ndarray, is_left: bool) -> tuple[float, float, float]:
        """Decompose thumb pose into (pinch_flex, cross_palm, mid_bend).

        XHAND1 thumb DOF semantics (corrected based on robot's actual joint behavior):

          Joint 0  pinch_flex   - thumb curls forward TOWARD a pinch with the
                                  index/middle fingertips. Goes from "thumb
                                  stuck out to the side" (extended/abducted)
                                  to "thumb pointing forward over palm".

          Joint 1  cross_palm   - rotation of the thumb across the palm such
                                  that the thumbpad turns from facing OUT
                                  (away from the palm, palm-up rest pose) to
                                  facing INTO the palm (opposition pose).
                                  This is rotation about the thumb's own long
                                  axis -- the thumbpad's twist.

          Joint 2  mid_bend     - flexion at the IP joint: distal phalanx
                                  curl relative to proximal. Always positive.

        How we extract them from the 21-keypoint Leap data
        --------------------------------------------------
        Working in wrist-local coords:
            +x = palm forward (toward MCP row)
            +y = palm normal  (out of palm, away from palm-down)
            +z = thumb side   (toward the thumb for a right hand)

        v_prox = THUMB_IP - THUMB_MCP   (proximal phalanx; first real segment)
        v_dist = THUMB_TIP - THUMB_IP   (distal phalanx)

        - pinch_flex is the swing of v_prox in the (+x, +z) palm plane,
          measured FROM the rest axis (~45 deg toward thumb side) TOWARD +x.
          When the thumb extends out to the side, v_prox is along the rest
          axis -> pinch_flex ~ 0. When the thumb pinches forward, v_prox
          aligns with +x -> pinch_flex grows. Note this is NOT the same as
          the old "opposition" angle: opposition rotates inward across the
          palm (toward -z), pinch flex rotates inward toward +x.

        - cross_palm is the rotation of the distal phalanx around the proximal
          axis. We project v_dist onto the plane perpendicular to v_prox, and
          measure how far around that circle it has rotated relative to the
          "thumbpad facing out" reference. When the thumbpad faces out (rest),
          the IP curls the distal toward -y (downward). When the thumbpad
          faces into the palm (opposed), the same IP curl bends the distal
          toward -z (toward the pinky). The angle between those two curl
          directions IS the cross-palm rotation.

        - mid_bend is just angle_between(v_prox, v_dist).
        """
        mcp = local[KP.THUMB_MCP]
        ip = local[KP.THUMB_IP]
        tip = local[KP.THUMB_TIP]

        v_prox = ip - mcp
        v_dist = tip - ip

        # ---- Joint 0: pinch flex (toward/away from a pinch) ----
        # Rest axis: roughly 45 deg between +x and +z for a right hand. As the
        # thumb folds toward a pinch, v_prox planar component rotates from the
        # rest axis toward +x. We measure that signed angle around +y.
        # Sign convention: + pinch_flex = rotated TOWARD +x (forward), i.e.
        # toward a pinch with the index/middle fingertips.
        rest_axis = _THUMB_REST_AXIS_LEFT if is_left else _THUMB_REST_AXIS_RIGHT
        v_prox_planar = np.array([v_prox[0], 0.0, v_prox[2]], dtype=np.float32)

        if np.linalg.norm(v_prox_planar) > 1e-6:
            # signed angle from rest -> v_prox_planar around +y
            raw = _signed_angle_in_plane(
                rest_axis, v_prox_planar,
                plane_normal=np.array([0.0, 1.0, 0.0]),
            )
            # For a RIGHT hand, rotating the planar projection from rest
            # (+x,+z at 45 deg) toward +x is a positive rotation around +y
            # (verified by cross product: cross([.7,0,.7], [.99,0,.17])_y > 0).
            # That's already the sign we want: + pinch_flex = toward pinch.
            # For a LEFT hand the rest axis mirrors, flip sign.
            pinch_flex = raw if not is_left else -raw
        else:
            pinch_flex = 0.0

        # ---- Joint 1: cross-palm rotation (thumbpad twist) ----
        # The thumbpad twist is encoded in WHICH DIRECTION the distal phalanx
        # bends relative to the proximal. With thumbpad facing OUT (palm-up
        # rest), an IP curl bends distal toward -y (down). With thumbpad
        # facing INTO the palm, the same IP curl bends distal toward -z
        # (toward pinky for a right hand).
        #
        # IMPORTANT: this signal only makes sense when the IP joint is
        # meaningfully curled. With distal collinear with proximal, the
        # perpendicular component of v_dist is tiny and noisy, and atan2 of
        # tiny numbers can read anywhere from -pi to +pi. We gate on
        # mid_bend > MIN_CURL_FOR_TWIST and otherwise hold cross_palm at 0.
        u = v_prox / (np.linalg.norm(v_prox) + 1e-9)

        # Reference direction: project -y onto plane perpendicular to u.
        # This is the direction distal points under IP curl when the thumbpad
        # faces OUT (palm-up rest pose).
        ref = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        w_ref = ref - np.dot(ref, u) * u
        n_w = np.linalg.norm(w_ref)
        if n_w < 1e-6:
            # Proximal aligned with -y -- degenerate. Use +x as ref.
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            w_ref = ref - np.dot(ref, u) * u
            n_w = np.linalg.norm(w_ref) + 1e-9
        w_ref /= n_w
        v_perp = np.cross(u, w_ref)

        v_dist_perp = v_dist - np.dot(v_dist, u) * u
        # Compare magnitude of perpendicular distal component to the proximal
        # length -- gives us "how much is the IP curled?" in dimensionless terms.
        proximal_len = np.linalg.norm(v_prox) + 1e-9
        twist_signal_strength = np.linalg.norm(v_dist_perp) / proximal_len

        # Only trust cross_palm when distal is curled at least ~15 deg off
        # the proximal axis (sin(15 deg) ~ 0.26, scaled by phalanx-length ratio).
        MIN_TWIST_SIGNAL = 0.15
        if twist_signal_strength < MIN_TWIST_SIGNAL:
            cross_palm = 0.0
        else:
            cw = float(np.dot(v_dist_perp, w_ref))
            cv = float(np.dot(v_dist_perp, v_perp))
            # Physical thumb twist is bounded to ~+/-90 deg. Using full atan2
            # gives us [-pi, pi], which means values just past +pi/2 wrap to -pi
            # and we get bang-bang behavior in joint 1.
            #
            # Instead we measure the angle in the HALF plane where cw >= 0
            # (i.e. the "thumbpad still mostly pointing toward reference" half).
            # If cw goes negative, the geometry has flipped past 90 deg, which
            # is unphysical for the human thumb -- we mirror the reading by
            # using |cw| instead of cw, which keeps the angle in [-pi/2, pi/2].
            #
            # Equivalently: cross_palm = atan2(cv, |cw|) is guaranteed in
            # [-pi/2, pi/2] and is continuous near the +/-90 deg boundary.
            cross_palm = float(np.arctan2(cv, abs(cw)))
            if is_left:
                cross_palm = -cross_palm

            # Additional safety: low-pass-filter-friendly clip. If the resulting
            # raw angle exceeds the joint's physical limit, cap it so the
            # downstream OneEuro filter sees a smooth signal.
            cross_palm = float(np.clip(cross_palm, -np.pi / 2 + 0.05, np.pi / 2 - 0.05))

        # ---- Joint 2: mid bend (IP curl) ----
        mid_bend = _angle_between(v_prox, v_dist)

        return float(pinch_flex), float(cross_palm), float(mid_bend)