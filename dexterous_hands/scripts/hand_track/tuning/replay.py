"""
Offline tool for inspecting and tuning XHand retargeting against
recorded WiLoR sessions.

Loads a pickle of WiLoR outputs, runs each through the retargeter, and
shows both the human MANO skeleton and the resulting XHand robot pose
in a single sapien window.

Controls (focus must be on sapien window):
  right arrow   - next frame
  left arrow    - previous frame
  space         - play/pause
  + / -         - increase/decrease scaling factor
  s             - print current state to console
  r             - reset to frame 0
  q / esc       - quit
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import sapien
from sapien.utils.viewer import Viewer
from dex_retargeting.retargeting_config import RetargetingConfig
from yourdfpy import URDF

from custom_retarget import retarget_xhand


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLOURS_RGB = {
    "thumb":  (0.89, 0.10, 0.11),
    "index":  (0.22, 0.49, 0.72),
    "middle": (0.30, 0.69, 0.29),
    "ring":   (0.60, 0.31, 0.64),
    "pinky":  (1.00, 0.50, 0.00),
}

R_MANO_TO_XHAND = np.array([
    [ -1, 0,  0],
    [ 0,  0,  1],
    [ 0,  1,  0],
], dtype=np.float64)


def bone_colour(a: int, b: int):
    j = max(a, b)
    if j <= 4:  return FINGER_COLOURS_RGB["thumb"]
    if j <= 8:  return FINGER_COLOURS_RGB["index"]
    if j <= 12: return FINGER_COLOURS_RGB["middle"]
    if j <= 16: return FINGER_COLOURS_RGB["ring"]
    return FINGER_COLOURS_RGB["pinky"]


# --- skeleton rendering inside sapien -------------------------------------

class HumanSkeletonActors:
    """Renders the MANO skeleton as a set of small sapien primitives.
    
    Each bone is a thin cylinder, each joint is a small sphere. We
    create them once and just update poses each frame, which is much
    cheaper than recreating geometry."""

    def __init__(self, scene: sapien.Scene, offset: np.ndarray | None = None):
        self.scene = scene
        self.offset = offset if offset is not None else np.array([0.0, 0.3, 0.0])
        self.joint_actors = []
        self.bone_actors = []

        # 21 small spheres for joints
        for i in range(21):
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.005,
                                      material=sapien.render.RenderMaterial([0.1, 0.1, 0.1, 1.0]))
            actor = builder.build_kinematic(name=f"joint_{i}")
            self.joint_actors.append(actor)

        # 20 thin cylinders for bones (one per HAND_BONES entry)
        for i, (a, b) in enumerate(HAND_BONES):
            builder = scene.create_actor_builder()
            colour = sapien.render.RenderMaterial(list(bone_colour(a, b)) + [1.0])
            # We'll size + orient these per-frame; default unit length
            builder.add_capsule_visual(radius=0.003, half_length=0.02,
                                        material=colour)
            actor = builder.build_kinematic(name=f"bone_{i}")
            self.bone_actors.append(actor)

    def update(self, joints_3d: np.ndarray):
        """Update positions of all joint and bone actors from a (21, 3)
        keypoint array. Coordinates are taken as-is (in MANO frame),
        with self.offset added so the human skeleton displays beside
        the robot rather than on top."""
        for i in range(21):
            p = R_MANO_TO_XHAND @ (joints_3d[i] + self.offset)
            self.joint_actors[i].set_pose(sapien.Pose(p=p.tolist()))

        for i, (a, b) in enumerate(HAND_BONES):
            pa = R_MANO_TO_XHAND @ (joints_3d[a] + self.offset)
            pb = R_MANO_TO_XHAND @ (joints_3d[b] + self.offset)
            mid = (pa + pb) / 2
            direction = pb - pa
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction = direction / length
            # Capsule's local x-axis = bone direction. Quaternion that
            # rotates [1,0,0] to direction.
            x_axis = np.array([1.0, 0.0, 0.0])
            v = np.cross(x_axis, direction)
            s = np.linalg.norm(v)
            c = np.dot(x_axis, direction)
            if s < 1e-6:
                # parallel or antiparallel
                if c > 0:
                    quat = [1.0, 0.0, 0.0, 0.0]
                else:
                    quat = [0.0, 0.0, 0.0, 1.0]
            else:
                axis = v / s
                angle = np.arctan2(s, c)
                half = angle / 2
                quat = [np.cos(half),
                        axis[0] * np.sin(half),
                        axis[1] * np.sin(half),
                        axis[2] * np.sin(half)]
            # Note: capsule half_length was set at build time.  We can't
            # easily resize live, so we accept slightly inaccurate bone
            # lengths in favor of speed. The orientation is correct.
            self.bone_actors[i].set_pose(
                sapien.Pose(p=mid.tolist(), q=quat))

# --- main replay loop ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=str)
    parser.add_argument("--config", type=str, default="urdf/xhand_left_vector.yml")
    parser.add_argument("--urdf-dir", type=str,
                        default="./urdf")
    parser.add_argument("--start-scaling", type=float, default=1.5)
    parser.add_argument("--play-hz", type=float, default=15.0,
                        help="Playback rate when playing")
    args = parser.parse_args()

    with open(args.session, "rb") as f:
        frames: List[Dict] = pickle.load(f)
    print(f"[load] {len(frames)} frames from {args.session}")

    RetargetingConfig.set_default_urdf_dir(args.urdf_dir)
    config = RetargetingConfig.load_from_file(args.config)
    retargeting = config.build()
    print(f"adaptor: {retargeting.optimizer.adaptor}")
    indices = retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0]
    task_indices = indices[1]
    joint_limits = retargeting.joint_limits
    joint_names = retargeting.optimizer.target_joint_names
    print(f"[retarget] {len(joint_names)} joints")

    urdf_abs = str(Path(args.urdf_dir) / config.urdf_path)
    print(f"[urdf] {urdf_abs}")

    # --- sapien scene
    scene = sapien.Scene()
    scene.set_timestep(1.0 / 60.0)
    scene.add_ground(altitude=-0.5)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0, 1, -1], [0.7, 0.7, 0.7])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_abs)
    robot.set_root_pose(sapien.Pose(p=[0, 0, 0]))
    sapien_joint_names = [j.name for j in robot.active_joints]

    # Cached mapping from retargeting joint order to sapien joint order
    sapien_idx = np.array([sapien_joint_names.index(n) for n in joint_names])

    # Human skeleton actors offset in y so they appear next to the robot
    skeleton = HumanSkeletonActors(scene, offset=np.array([0.3, 0, -0.075]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(0, -0.3, -0.05)
    viewer.set_camera_rpy(0, -0.4, -1.57)

    # --- state
    frame_idx = 0
    playing = False
    scaling = args.start_scaling
    last_play_advance = time.perf_counter()
    play_period = 1.0 / args.play_hz

    # def compute_qpos(frame, scl):
    #     joints_mano = frame["joints_3d"]
    #     joints_xhand = joints_mano @ R_MANO_TO_XHAND.T
    #     target_vectors = (joints_xhand[task_indices]
    #                       - joints_xhand[origin_indices])
    #     retargeting.optimizer.scaling = scl

    #     # Reset warm-start to mid-range pose every call
    #     midpoint = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    #     retargeting.last_qpos = midpoint.astype(np.float32)

    #     qpos = retargeting.retarget(target_vectors)
    #     return np.clip(qpos, joint_limits[:, 0], joint_limits[:, 1])
    
    def compute_qpos(frame, _):  # scl unused, kept for compat
        joints_mano = frame["joints_3d"]
        qpos = retarget_xhand(joints_mano)
        return np.clip(qpos, joint_limits[:, 0], joint_limits[:, 1])
    
    def update():
        frame = frames[frame_idx]
        qpos = compute_qpos(frame, scaling)
        full_qpos = np.zeros(len(sapien_joint_names))
        full_qpos[sapien_idx] = qpos
        robot.set_qpos(full_qpos)
        skeleton.update(frame["joints_3d"])
        return qpos

    qpos_current = update()

    # --- input handling
    # sapien's viewer exposes window.key_press(key) returning bool,
    # but the API varies between sapien versions. We poll each frame.
    print("\n[controls]")
    print("  right/left: step frames")
    print("  space: play/pause")
    print("  +/-: scaling factor")
    print("  s: snapshot to console")
    print("  r: reset to frame 0")
    print("  q: quit\n")

    # Track keys so we only fire on press, not on hold
    prev_keys = {}

    def edge(key: str) -> bool:
        """True on the frame the key transitions from up to down."""
        try:
            now = viewer.window.key_down(key) # type: ignore
        except Exception:
            now = False
        was = prev_keys.get(key, False)
        prev_keys[key] = now
        return now and not was

    while not viewer.closed:
        # Handle input
        changed = False
        if edge("right"):
            frame_idx = min(frame_idx + 1, len(frames) - 1)
            changed = True
        if edge("left"):
            frame_idx = max(frame_idx - 1, 0)
            changed = True
        if edge("space"):
            playing = not playing
            print(f"[play] {'started' if playing else 'paused'}")
        if edge("="):  # + key on most keyboards
            scaling = min(scaling + 0.05, 3.0)
            print(f"[scaling] {scaling:.2f}")
            changed = True
        if edge("-"):
            scaling = max(scaling - 0.05, 0.3)
            print(f"[scaling] {scaling:.2f}")
            changed = True
        if edge("r"):
            frame_idx = 0
            changed = True
        if edge("g"):
            frame = frames[frame_idx]
            joints_mano = frame["joints_3d"]
            joints_xhand = joints_mano @ R_MANO_TO_XHAND.T
            target_vectors = (joints_xhand[task_indices]
                              - joints_xhand[origin_indices])
            target_vectors_scaled = target_vectors * scaling
            
            # Use yourdfpy for FK at zero pose
            yrdf = URDF.load(urdf_abs)
            yrdf.update_cfg(np.zeros(yrdf.num_actuated_joints))
            
            wrist_pos = yrdf.get_transform("left_hand_link")[:3, 3]
            
            robot_fingertip_links = [
                "left_hand_thumb_rota_tip",
                "left_hand_index_rota_tip",
                "left_hand_mid_tip",
                "left_hand_ring_tip",
                "left_hand_pinky_tip",
            ]
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            
            # Pre-compute robot fingertip positions
            robot_tip_positions = []
            for link_name in robot_fingertip_links:
                pos = yrdf.get_transform(link_name)[:3, 3] - wrist_pos
                robot_tip_positions.append(pos)
            
            print(f"\n[geometric] frame {frame_idx}, scaling {scaling:.2f}")
            print(f"For each human target, distance to each robot fingertip at rest:")
            for i, name in enumerate(finger_names):
                target = target_vectors_scaled[i]
                print(f"\n  Human {name:7s} target: [{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]")
                dists = [np.linalg.norm(target - rt) for rt in robot_tip_positions]
                closest_idx = int(np.argmin(dists))
                for j in range(len(robot_fingertip_links)):
                    rt = robot_tip_positions[j]
                    marker = " <-- closest" if j == closest_idx else ""
                    print(f"    robot {finger_names[j]:7s} tip at [{rt[0]:+.3f}, {rt[1]:+.3f}, {rt[2]:+.3f}]  "
                          f"dist={dists[j]:.3f}{marker}")
        if edge("f"):
            # Force a fresh retargeter call for the current frame.
            # Bypasses warm-start state to show what the optimiser produces
            # cold against this exact input.
            print(f"[fresh] rebuilding retargeter and recomputing frame {frame_idx}...")
            t0 = time.perf_counter()
            fresh = RetargetingConfig.load_from_file(args.config).build()
            fresh.optimizer.scaling_factor = scaling # type: ignore

            frame = frames[frame_idx]
            joints_mano = frame["joints_3d"]
            joints_xhand = joints_mano @ R_MANO_TO_XHAND.T
            target_vectors = (joints_xhand[task_indices]
                              - joints_xhand[origin_indices])
            qpos_fresh = fresh.retarget(target_vectors)
            qpos_fresh = np.clip(qpos_fresh, joint_limits[:, 0], joint_limits[:, 1])

            # Apply to the visible robot
            full_qpos = np.zeros(len(sapien_joint_names))
            full_qpos[sapien_idx] = qpos_fresh
            robot.set_qpos(full_qpos)

            # Update the cached qpos so the human skeleton stays in sync
            qpos_current = qpos_fresh

            elapsed = time.perf_counter() - t0
            print(f"[fresh] done in {elapsed*1000:.0f} ms")
            print(f"[fresh] qpos: "
                  f"{np.array2string(np.rad2deg(qpos_fresh), precision=3)}")
        if edge("i"):
            # Inspect joint 0 and the supposed fingertips
            frame = frames[frame_idx]
            j = frame["joints_3d"]
            print(f"\n[inspect] frame {frame_idx}, all 21 joints relative to wrist:")
            rel = j - j[0]  # relative to wrist
            for i in range(21):
                v = rel[i]
                dist = np.linalg.norm(v)
                print(f"  joint {i:2d}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]  dist={dist:.3f}")
            
            print(f"\n  Standard MANO assumption: thumb_tip=4, index_tip=8, middle_tip=12, ring_tip=16, pinky_tip=20")
            print(f"  These should be the joints farthest from the wrist (tips).")
            print(f"  Distances:")
            for tip_idx in [4, 8, 12, 16, 20]:
                print(f"    joint {tip_idx}: {np.linalg.norm(rel[tip_idx]):.3f} m")
        if edge("q") or edge("escape"):
            break

        # Playback advance
        now = time.perf_counter()
        if playing and (now - last_play_advance) > play_period:
            if frame_idx < len(frames) - 2:
                frame_idx += 2
                changed = True
                last_play_advance = now
            else:
                playing = False
                print("[play] reached end")

        if changed:
            qpos_current = update()

        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()