import numpy as np


# MANO joint indices per finger
MANO_FINGERS = {
    "thumb":  [1, 2, 3, 4],   # CMC, MCP, IP, tip
    "index":  [5, 6, 7, 8],   # MCP, PIP, DIP, tip
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

GAIN_PER_FINGER = {"index": 0.85, "middle": 0.88, "ring": 0.82, "pinky": 0.85}

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in radians, in [0, pi]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def compute_hand_axes(joints: np.ndarray) -> dict:
    """Compute stable hand-frame axes from the four MCP positions
    and the wrist. Robust to finger pose because it only uses the
    palm landmarks.
    
    Returns:
      across_palm: unit vector from index-MCP-side toward pinky-MCP-side
      finger_extend: unit vector from wrist toward middle-MCP (the
                     direction fingers extend at rest)
      palm_normal: unit vector perpendicular to both, palm-out direction
    """
    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    ring_mcp = joints[13]
    pinky_mcp = joints[17]
    
    # Across-palm axis: from index-MCP toward pinky-MCP
    across_palm = pinky_mcp - index_mcp
    across_palm /= np.linalg.norm(across_palm)

    # index_across_palm = middle_mcp-index_mcp
    # index_across_palm /= np.linalg.norm(index_across_palm)
    # middle_across_palm = ring_mcp - middle_mcp
    # middle_across_palm /= np.linalg.norm(middle_across_palm)
    # ring_across_palm = pinky_mcp - ring_mcp
    # ring_across_palm /= np.linalg.norm(ring_across_palm)
    # pinky_across_palm = -ring_across_palm
    # pinky_across_palm /= np.linalg.norm(pinky_across_palm)
    # across_palm_fingers = {"index": index_across_palm,
    #                        "middle": middle_across_palm,
    #                        "ring": ring_across_palm,
    #                        "pinky": pinky_across_palm}
    
    # Finger-extend axis: from wrist toward middle MCP (most central)
    finger_extend = middle_mcp - wrist
    finger_extend /= np.linalg.norm(finger_extend)
    
    # Palm normal: perpendicular to both. Direction depends on
    # cross product order; we want palm-out.
    palm_normal = np.cross(finger_extend, across_palm)
    palm_normal /= np.linalg.norm(palm_normal)
    
    return {
        # "across_palm": across_palm_fingers,
        "across_palm": across_palm,
        "finger_extend": finger_extend,
        "palm_normal": palm_normal,
    }

def signed_angle_around_axis(v1: np.ndarray, v2: np.ndarray,
                              axis: np.ndarray) -> float:
    """Signed angle from v1 to v2, rotating around axis. Positive
    angle is right-hand-rule rotation around the axis."""
    # Project both onto the plane perpendicular to axis
    axis = axis / np.linalg.norm(axis)
    v1p = v1 - np.dot(v1, axis) * axis
    v2p = v2 - np.dot(v2, axis) * axis
    
    n1 = np.linalg.norm(v1p)
    n2 = np.linalg.norm(v2p)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    v1p /= n1
    v2p /= n2
    
    cross = np.cross(v1p, v2p)
    sin_a = np.dot(cross, axis)
    cos_a = np.dot(v1p, v2p)
    return float(np.arctan2(sin_a, cos_a))

def retarget_curl_finger(joints: np.ndarray, idxs: list,
                          ref_axis: np.ndarray, curl_axis : np.ndarray) -> tuple:
    """Compute (j1, j2) for a flex finger, using the hand-frame
    across_palm axis as the curl reference. This is robust to
    finger abduction (squeezing fingers together) because it doesn't
    rely on each finger's instantaneous curl plane.

    ref_axes: the finger-extend axis projected onto the curl plane
    """
    mcp = joints[idxs[0]]
    pip = joints[idxs[1]]
    dip = joints[idxs[2]]
    tip = joints[idxs[3]]
    
    proximal = pip - mcp
    middle = dip - pip
    end = tip - dip
    
    # j1: rotation from "extended" to proximal segment around the
    # curl axis. Negative means curl-toward-palm, positive means
    # away. Or vice versa - depends on cross product convention.
    j1 = signed_angle_around_axis(ref_axis, proximal, curl_axis)
    
    # j2: rotation from proximal to middle segment around the same axis
    j2 = signed_angle_around_axis(proximal, middle, curl_axis)

    j3 = signed_angle_around_axis(middle, end, curl_axis)
    
    # Clamp hyperextension and reverse rotations
    j1 = max(j1, 0.0)
    j2 = max(j2, 0.0)
    
    return j1 + j2*0.7, j2*0.3 + j3

"""Original primitive thumb retarget"""
# def retarget_thumb(joints: np.ndarray) -> tuple:
#     """Return (bend, rota1, rota2) for the XHand thumb's 3 DOF.
    
#     Thumb topology between MANO and XHand is genuinely different,
#     so this is a simplified mapping that captures gross opposition
#     and curl.
#     """
#     wrist = joints[0]
#     cmc = joints[1]   # thumb CMC (carpometacarpal)
#     mcp = joints[2]   # thumb MCP
#     ip = joints[3]    # thumb IP
#     tip = joints[4]   # thumb tip
    
#     # Crude approach: use the same logic as other fingers for
#     # rota1/rota2 (curl), and use a simple thumb-to-index distance
#     # for the bend (opposition) joint.
#     extended_axis = cmc - wrist
#     proximal = mcp - cmc
#     middle = ip - mcp
    
#     rota1 = angle_between(extended_axis, proximal)
#     rota2 = angle_between(proximal, middle)
    
#     # Bend: how "opposed" is the thumb? Distance from thumb tip to
#     # index MCP, normalised to a 0..1 range.
#     index_mcp = joints[5]
#     thumb_to_index = np.linalg.norm(tip - index_mcp)
#     # Rough heuristic: 0.10m apart = no opposition, 0.04m = full opposition
#     bend = np.clip((0.10 - thumb_to_index) / 0.06, 0.0, 1.0)
#     bend_rad = bend * 1.5  # scale to ~85 deg of bend at full opposition
    
#     return bend_rad, rota1, rota2

# def retarget_thumb(joints: np.ndarray) -> tuple:
#     wrist = joints[0]
#     cmc = joints[1]   # thumb CMC (carpometacarpal)
#     mcp = joints[2]   # thumb MCP
#     ip = joints[3]    # thumb IP
#     tip = joints[4]   # thumb tip

#     index_base = joints[5] - wrist

#     palm_normal = np.cross(joints[17] - wrist, index_base) # Index and pinky base joints
#     palm_normal /= np.linalg.norm(palm_normal)

#     # Bend/abduction joint

#     # Out of plane motion of thumb base - moving toward the camera
#     thumb_wrist_base = mcp - wrist
#     cos_a = np.dot(palm_normal, thumb_wrist_base) / np.linalg.norm(thumb_wrist_base)
#     bend = (0.8*np.pi/2) - np.arccos(np.clip(cos_a,-1.0,1.0))
#     # Cross-palm abduction of thumb base
#     thumb_base = mcp - cmc
#     thumb_base_proj = thumb_base - np.dot(thumb_base, palm_normal) * palm_normal
#     index_base_proj = index_base - np.dot(index_base, palm_normal) * palm_normal

#     cos_a = np.dot(thumb_base_proj, index_base_proj) / (np.linalg.norm(thumb_base_proj)*np.linalg.norm(index_base_proj))
#     bend_abduct = np.arccos(np.clip(cos_a,-1.0,1.0))
#     if bend_abduct > np.pi / 2: bend_abduct = np.pi - bend_abduct

#     # Rotation joints

#     # Crude approach: use the same underlying logic as other fingers for
#     # rota1/rota2 (curl). No need for blending a third joint as the 
#     # human thumb does not have one 
#     extended_axis = cmc - wrist
#     proximal = ip - mcp
#     middle = tip - ip

#     rota1 = angle_between(extended_axis, proximal) + 0.1
#     rota2 = angle_between(proximal, middle)

#     return 4*bend + 1*bend_abduct,1*rota1,1.2* rota2

# def retarget_thumb(joints: np.ndarray) -> tuple:
#     wrist = joints[0]
#     cmc = joints[1]   # thumb CMC (carpometacarpal)
#     mcp = joints[2]   # thumb MCP
#     ip = joints[3]    # thumb IP
#     tip = joints[4]   # thumb tip

#     index_base = joints[5] - wrist

#     palm_normal = np.cross(joints[17] - wrist, index_base) # Index and pinky base joints
#     palm_normal /= np.linalg.norm(palm_normal)

#     # Bend/abduction joint

#     # plane through thumb
#     proximal = ip - mcp
#     upper = tip - ip
#     proximal /= np.linalg.norm(proximal)
#     upper /= np.linalg.norm(upper)
#     thumb_norm = np.cross(proximal, upper)
#     # Probably need a colinearity check between proximal and upper

#     index_middle = index_base - joints[9] # Middle to index vector
#     index_middle /= np.linalg.norm(index_middle)
#     project_plane = np.cross(index_middle, palm_normal)
    
#     thumb_norm_projected = thumb_norm - np.dot(thumb_norm, project_plane) * project_plane
#     thumb_norm_projected /= np.linalg.norm(thumb_norm_projected)
#     track_abduct = np.arccos(np.dot(thumb_norm_projected,palm_normal))
#     thumb_abduct = (track_abduct - 2.4) * 2.3

#     # Rotation joints

#     # Crude approach: use the same underlying logic as other fingers for
#     # rota1/rota2 (curl). No need for blending a third joint as the 
#     # human thumb does not have one 
#     extended_axis = cmc - wrist
#     proximal = ip - mcp
#     middle = tip - ip

#     rota1 = angle_between(extended_axis, proximal) + 0.1
#     rota2 = angle_between(proximal, middle)

#     return thumb_abduct,1*rota1,0.9* rota2


def retarget_thumb(joints: np.ndarray) -> tuple:
    wrist = joints[0]
    cmc = joints[1]   # thumb CMC (carpometacarpal)
    mcp = joints[2]   # thumb MCP
    ip = joints[3]    # thumb IP
    tip = joints[4]   # thumb tip

    index_base = joints[5] - wrist

    palm_normal = np.cross(joints[17] - wrist, index_base) # Index and pinky base joints
    palm_normal /= np.linalg.norm(palm_normal)

    # Bend/abduction joint

    # index->ring vector (plane normal)
    plane_normal = joints[5] - joints[13]
    plane_normal /= np.linalg.norm(plane_normal)

    thumb_base = cmc - wrist
    extended_axis = mcp - cmc
    # Project both
    thumb_base -= np.dot(thumb_base,plane_normal) * plane_normal
    extended_axis -= np.dot(extended_axis,plane_normal) * plane_normal
    thumb_base /= np.linalg.norm(thumb_base)
    extended_axis /= np.linalg.norm(extended_axis)

    thumb_abduct = signed_angle_around_axis(thumb_base, extended_axis, plane_normal)

    # Rotation joints

    # Crude approach: use the same underlying logic as other fingers for
    # rota1/rota2 (curl). No need for blending a third joint as the 
    # human thumb does not have one 
    extended_axis = mcp - cmc
    proximal = ip - mcp
    middle = tip - ip

    rota1 = angle_between(extended_axis, proximal) + 0.1
    rota2 = angle_between(proximal, middle)

    return 1.57 - (thumb_abduct + 0.1) * 2.5,1*rota1,0.9* rota2



def retarget_xhand(joints_mano: np.ndarray) -> np.ndarray:
    """Compute 12-DOF XHand qpos from MANO 21-keypoint hand pose.
    Output order matches your YAML's target_joint_names:
      0: thumb_bend
      1: thumb_rota1
      2: thumb_rota2
      3: index_bend
      4: index_j1
      5: index_j2
      6: mid_j1
      7: mid_j2
      8: ring_j1
      9: ring_j2
      10: pinky_j1
      11: pinky_j2
    """
    qpos = np.zeros(12, dtype=np.float64)
    axes = compute_hand_axes(joints_mano)
    
    # Thumb (3 DOF)
    qpos[0], qpos[1], qpos[2] = retarget_thumb(joints_mano)
    
    # Index bend: leave at zero (or estimate from index-middle separation later)
    qpos[3] = 0.0
    
    # Index, middle, ring, pinky: each has 2 DOF
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        idxs = MANO_FINGERS[name]
        j1, j2 = retarget_curl_finger(joints_mano, idxs, axes["finger_extend"], axes["across_palm"]) # [name]
        j1 *= GAIN_PER_FINGER[name]
        j2 *= GAIN_PER_FINGER[name]
        # Output position depends on finger
        if name == "index":
            qpos[4], qpos[5] = j1, j2
        elif name == "middle":
            qpos[6], qpos[7] = j1, j2
        elif name == "ring":
            qpos[8], qpos[9] = j1, j2
        elif name == "pinky":
            qpos[10], qpos[11] = j1, j2
    
    return qpos