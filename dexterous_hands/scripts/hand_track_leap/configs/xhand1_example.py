"""
Example: how to override retargeting parameters.

Copy this file, modify, and pass the configured retargeter into run_teleop.py
(or import it directly).
"""
import numpy as np
from leap_xhand_teleop import XHand1Retargeter
from leap_xhand_teleop.retarget_xhand1 import JOINT_LIMITS_RAD


# Example: clamp index abduction more tightly to avoid the robot finger
# colliding with the middle finger.
limits = JOINT_LIMITS_RAD.copy()
limits[3] = [-0.15, 0.15]   # index_abd

# Example: invert thumb opposition direction if your URDF defines it backwards.
directions = np.ones(12, dtype=np.float32)
directions[0] = -1.0   # thumb_cmc_yaw

# Example: scale all PIP commands by 1.2 to compensate for human under-curl.
gains = np.ones(12, dtype=np.float32)
gains[[5, 7, 9, 11]] = 1.2  # all four PIP joints

retargeter = XHand1Retargeter(
    joint_limits=limits,
    directions=directions,
    gains=gains,
)
