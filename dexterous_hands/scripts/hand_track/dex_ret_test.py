"""
Standalone validation that dex-retargeting installs and runs correctly.
Uses the shipped LEAP hand teleop config to validate the install -
output values aren't meaningful for XHand, this is only an install/runtime
check before writing the XHand-specific config.
"""

import os
import time
import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig


# --- adjust these two paths for your machine -------------------------------
# Adjust these paths
XHAND_CONFIG = "./urdf/xhand_left_vector.yml"
XHAND_URDF_DIR = "./urdf"

R_MANO_TO_XHAND = np.array([
    [ 0,  1,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
], dtype=np.float64)


def to_xhand_frame(joints_mano: np.ndarray) -> np.ndarray:
    """Rotate MANO-frame joints into XHand URDF frame."""
    return joints_mano @ R_MANO_TO_XHAND.T

def main():
    RetargetingConfig.set_default_urdf_dir(XHAND_URDF_DIR)
    config = RetargetingConfig.load_from_file(XHAND_CONFIG)
    retargeting = config.build()

    joint_names = retargeting.optimizer.target_joint_names
    print(f"[init] retargeting built. {len(joint_names)} target joints:")
    for n in joint_names:
        print(f"        {n}")

    # Synthetic flat-hand pose, MANO-topology, in metres.
    joints_3d = make_flat_hand_pose()
    print(f"\n[input] hand pose shape: {joints_3d.shape}")
    print(f"[input] wrist:     {joints_3d[0]}")
    print(f"[input] thumb tip: {joints_3d[4]}")
    print(f"[input] index tip: {joints_3d[8]}")

    # Cold call (includes optimiser init).
    indices = retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0]
    task_indices = indices[1]

    # Cold call
    joints_in_xhand = to_xhand_frame(joints_3d)
    target_vectors = (joints_in_xhand[task_indices]
                      - joints_in_xhand[origin_indices])
    print(f"\n[input] target_vectors shape: {target_vectors.shape}")
    print(f"[input] target_vectors:\n{target_vectors}")

    t0 = time.perf_counter()
    qpos = retargeting.retarget(target_vectors)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warm calls
    warm_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        qpos = retargeting.retarget(target_vectors)
        warm_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"\n[timing] cold call: {cold_ms:.2f} ms")
    print(f"[timing] warm call: median {np.median(warm_times):.2f} ms, "
          f"p95 {np.percentile(warm_times, 95):.2f} ms, "
          f"max {np.max(warm_times):.2f} ms")

    print(f"\n[output] qpos shape: {qpos.shape}")
    print(f"[output] qpos (rad): {np.array2string(qpos, precision=3)}")
    print(f"[output] qpos (deg): "
          f"{np.array2string(np.rad2deg(qpos), precision=1)}")

    # Smoothness check: small input perturbations should produce small
    # output deltas, confirming warm-start works.
    print(f"\n[smoothness] perturbed inputs:")
    rng = np.random.default_rng(0)
    prev_qpos = qpos.copy()
    for i in range(5):
        perturbed = joints_3d + rng.standard_normal(joints_3d.shape) * 0.005
        pose_xhand = to_xhand_frame(perturbed)
        target_vectors = (pose_xhand[task_indices] - pose_xhand[origin_indices])
        qpos = retargeting.retarget(target_vectors)
        delta = np.abs(qpos - prev_qpos).max()
        print(f"  step {i}: max joint delta = {np.rad2deg(delta):.2f} deg")
        prev_qpos = qpos.copy()


def make_flat_hand_pose() -> np.ndarray:
    """Synthetic flat-hand MANO-topology pose, ~18cm from wrist to middle
    fingertip, fingers along +x, palm normal +z."""
    j = np.zeros((21, 3), dtype=np.float64)
    mcp_y = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
    mcp_x = np.array([0.04, 0.08, 0.085, 0.08, 0.075])
    finger_lengths = np.array([0.05, 0.07, 0.075, 0.07, 0.06])
    indices = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }
    for fi, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        for k, idx in enumerate(indices[name]):
            j[idx, 0] = mcp_x[fi] + (k / 3.0) * finger_lengths[fi]
            j[idx, 1] = mcp_y[fi]
            j[idx, 2] = 0.0
    return j


if __name__ == "__main__":
    main()