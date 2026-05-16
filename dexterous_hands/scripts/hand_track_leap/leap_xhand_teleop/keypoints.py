"""
21-keypoint hand model definition.

We adopt the MediaPipe-style 21-point indexing because it's the de-facto standard
for hand retargeting work and most of the public retargeting code (dex-retargeting,
manopth, etc.) uses it. Leap Motion provides bone start/end positions which we
remap into this convention in `leap_source.py`.

Index layout:

       8   12   16   20      <- finger tips (DIP+1 / TIP)
       |    |   |    |
       7   11   15   19      <- DIP joints
       |    |   |    |
       6   10   14   18      <- PIP joints
       |    |   |    |
   4   5    9   13   17      <- MCP joints (4 = thumb tip)
   |   \    |   /   /
   3    \   |  /   /
   |     \  | /   /
   2      \ |/   /
   |       \|/
   1 ------ 0                <- 0 = wrist, 1=CMC, 2=MCP(thumb), 3=IP, 4=TIP

Indexing:
  0  : WRIST
  1  : THUMB_CMC      (carpometacarpal, base of thumb metacarpal)
  2  : THUMB_MCP      (metacarpophalangeal, base of proximal phalanx)
  3  : THUMB_IP       (interphalangeal)
  4  : THUMB_TIP
  5  : INDEX_MCP
  6  : INDEX_PIP
  7  : INDEX_DIP
  8  : INDEX_TIP
  9  : MIDDLE_MCP
  10 : MIDDLE_PIP
  11 : MIDDLE_DIP
  12 : MIDDLE_TIP
  13 : RING_MCP
  14 : RING_PIP
  15 : RING_DIP
  16 : RING_TIP
  17 : PINKY_MCP
  18 : PINKY_PIP
  19 : PINKY_DIP
  20 : PINKY_TIP

All coordinates are in METERS, in the Leap device frame (right-handed, Y up,
+Z toward the user). The retargeter converts to a wrist-local frame.
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class KP(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Kinematic chains - useful for visualization and for the retargeter.
FINGER_CHAINS = {
    "thumb":  [KP.WRIST, KP.THUMB_CMC, KP.THUMB_MCP, KP.THUMB_IP, KP.THUMB_TIP],
    "index":  [KP.WRIST, KP.INDEX_MCP, KP.INDEX_PIP, KP.INDEX_DIP, KP.INDEX_TIP],
    "middle": [KP.WRIST, KP.MIDDLE_MCP, KP.MIDDLE_PIP, KP.MIDDLE_DIP, KP.MIDDLE_TIP],
    "ring":   [KP.WRIST, KP.RING_MCP, KP.RING_PIP, KP.RING_DIP, KP.RING_TIP],
    "pinky":  [KP.WRIST, KP.PINKY_MCP, KP.PINKY_PIP, KP.PINKY_DIP, KP.PINKY_TIP],
}

# Edge list for wireframe rendering.
HAND_EDGES = []
for chain in FINGER_CHAINS.values():
    for a, b in zip(chain[:-1], chain[1:]):
        HAND_EDGES.append((int(a), int(b)))
# Palm edges across MCPs for a more rigid-looking wireframe
HAND_EDGES += [
    (int(KP.INDEX_MCP), int(KP.MIDDLE_MCP)),
    (int(KP.MIDDLE_MCP), int(KP.RING_MCP)),
    (int(KP.RING_MCP), int(KP.PINKY_MCP)),
    (int(KP.THUMB_CMC), int(KP.INDEX_MCP)),
]


@dataclass
class HandKeypoints:
    """A single frame of hand tracking data.

    Attributes:
        points: (21, 3) float32 array, meters, in Leap device frame.
        is_left: True if left hand, False if right.
        timestamp: monotonic seconds (float).
        confidence: 0..1 from Leap's pinch_strength fallback or 1.0 if not available.
        palm_normal: (3,) unit vector, Leap's palm normal.
        palm_direction: (3,) unit vector, Leap's palm forward direction (toward fingers).
    """
    points: np.ndarray
    is_left: bool
    timestamp: float
    confidence: float = 1.0
    palm_normal: np.ndarray = None
    palm_direction: np.ndarray = None

    def __post_init__(self):
        assert self.points.shape == (21, 3), f"expected (21,3), got {self.points.shape}"
        self.points = self.points.astype(np.float32)

    def wrist_local(self) -> np.ndarray:
        """Returns keypoints expressed in the wrist-local frame.

        x-axis: from wrist toward middle MCP (palm forward)
        y-axis: palm normal (out of palm; for right hand, points away from palm-down)
        z-axis: x cross y (toward thumb side for right hand)

        This frame is what the retargeter consumes. Defining it explicitly here
        (rather than relying on Leap's basis vectors) makes the kinematics
        reproducible if you ever swap the input source.
        """
        wrist = self.points[KP.WRIST]
        mid_mcp = self.points[KP.MIDDLE_MCP]
        idx_mcp = self.points[KP.INDEX_MCP]
        pky_mcp = self.points[KP.PINKY_MCP]

        x = mid_mcp - wrist
        x /= (np.linalg.norm(x) + 1e-9)

        # Palm plane via two MCP edges; normal is consistent for the given handedness.
        v1 = idx_mcp - wrist
        v2 = pky_mcp - wrist
        n = np.cross(v1, v2)
        # For a right hand, np.cross(idx-wrist, pky-wrist) points OUT of the palm
        # (away from palm-down). For a left hand it points the other way.
        if self.is_left:
            n = -n
        y = n / (np.linalg.norm(n) + 1e-9)

        # Re-orthogonalize
        z = np.cross(x, y)
        z /= (np.linalg.norm(z) + 1e-9)
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=1)  # 3x3, columns are basis vectors
        local = (self.points - wrist) @ R  # apply R^T via right-multiply
        return local.astype(np.float32)
