# Leap Motion 2 → XHAND1 Teleoperation

Camera-based teleop pipeline mapping a human hand seen by the Leap Motion 2 to
the 12 DOFs of the RobotEra XHAND1, designed for ~30 Hz operation with optional
real-time visualisation and ROS publishing.

## Pipeline

```
LeapMotion2  ─►  LeapSource  ─►  KeypointFilter  ─►  XHand1Retargeter  ─►  RosBridgePublisher
                  (21 kp,         (one-Euro,         (12-DOF rad cmd)        (roslibpy →
                   meters)         ~30 Hz tuned)                              rosbridge_server)
                       │
                       └─►  HandWireframeViewer  (Open3D 3D + matplotlib diagnostics)
                       └─►  CSV logger           (optional, for offline analysis)
```

Every stage can be toggled. Common bring-up workflow:

| Stage              | Software-only | Hardware bring-up   | Robot bring-up   | Production     |
| ------------------ | ------------- | ------------------- | ---------------- | -------------- |
| Source             | `--mock`      | Leap                | Leap             | Leap           |
| Viewer             | ON            | ON                  | ON               | `--no-viewer`  |
| ROS publisher      | `--no-ros`    | `--no-ros`          | ON               | ON             |
| CSV log            | optional      | `--log debug.csv`   | optional         | optional       |

## Setup (Ubuntu 20.04 / 22.04)

```bash
# 1. System: Gemini Hand Tracking Software (>= 5.17)
#    Download .deb from https://leap2.ultraleap.com/downloads/
sudo apt install ./ultraleap-hand-tracking-service-*.deb
sudo systemctl enable --now ultraleap-hand-tracking-service

# 2. Python venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Ultraleap Python bindings (NOT on PyPI)
git clone https://github.com/ultraleap/leapc-python-bindings
cd leapc-python-bindings
pip install -r requirements.txt
pip install -e leapc-python-api
cd ..

# 4. Sanity check (no hardware needed)
python -m examples.test_retargeter

# 5. Software-only run (synthetic hand)
python -m examples.run_teleop --mock --no-ros

# 6. Live, viewer-only
python -m examples.run_teleop --no-ros

# 7. Full pipeline (assuming rosbridge running on robot at 192.168.1.10)
python -m examples.run_teleop --ros-host 192.168.1.10
```

The ROS side just needs `rosbridge_server`:
```bash
# ROS 1
rosrun rosbridge_server rosbridge_websocket
# ROS 2
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```
We publish `std_msgs/Float64MultiArray` on `/xhand1/target_joint_positions` by
default. Pass `--ros-jointstate` for `sensor_msgs/JointState` instead.

## CLI

```
--mock                  use synthetic hand (no Leap hardware)
--hand {left,right,any} pick which hand to track if multiple visible
--hmd                   set Leap to HMD tracking mode
--mincutoff 1.5         One-Euro min cutoff Hz (higher = less smoothing)
--beta 0.05             One-Euro beta (higher = more responsive at speed)
--rate 30               target loop rate Hz
--no-ros                disable ROS publishing
--ros-host localhost    rosbridge host
--ros-port 9090         rosbridge port
--ros-topic /xhand1/target_joint_positions
--ros-jointstate        publish JointState instead of Float64MultiArray
--no-viewer             headless (no GUI at all)
--no-3d                 disable Open3D 3D wireframe only
--no-diag               disable matplotlib diagnostics only
--log path.csv          dump every frame to CSV
```

## Diagnostics panel

When the viewer is on, you get two windows:

**Open3D wireframe** — 21 keypoints (color-coded per finger) connected by
bones, plus a small coordinate frame at the wrist showing the wrist-local
basis the retargeter uses. If this gizmo doesn't sit naturally on your wrist
with `+x` going through your middle MCP, the retargeter input is wrong.

**Matplotlib diagnostics** — three panels:
1. *Text*: live FPS, frame age (latency since last received frame), tracking
   confidence, dropped-frame counts.
2. *Per-keypoint jitter*: rolling std-dev (mm) for each of the 21 keypoints.
   With a still hand and good lighting you should see < 1 mm everywhere
   post-filter; > 3-4 mm typically indicates poor lighting or partial occlusion.
3. *Joint command bar chart*: current 12-DOF command in rad with the joint
   range shown as a green band. Bars turn red when within 0.05 rad of a limit
   so you can spot saturation immediately.

## Retargeter

`leap_xhand_teleop/retarget_xhand1.py` is the most opinionated module — it's
where you'll need to tune things if the robot doesn't behave right. The
top-of-file docstring lays out:

* The 12-DOF assumed layout and joint ordering
* Joint range (`JOINT_LIMITS_RAD`) - **verify against your URDF**
* Sign conventions per joint (`JOINT_DIRECTIONS`)
* Per-joint gains (`HUMAN_TO_ROBOT_GAIN`)
* Thumb rest axis (`_THUMB_REST_AXIS_RIGHT/LEFT`) - calibrate to user

### Tuning workflow

1. Run `python -m examples.test_retargeter` — confirms basic kinematic
   conventions are correct for synthetic poses.
2. Run `python -m examples.run_teleop --mock --no-ros` — confirms the full
   pipeline works end-to-end without hardware.
3. Run `python -m examples.run_teleop --no-ros` with your hand — watch the
   joint bar chart. For each finger, slowly curl/extend it and check:
   * The corresponding bar moves in the *correct direction* (curl → positive)
   * The bar's *range* roughly matches the human's range of motion
   * Other fingers' bars don't move when they shouldn't
4. If a joint moves backward, flip its sign in `JOINT_DIRECTIONS`.
5. If the range doesn't match, scale `HUMAN_TO_ROBOT_GAIN` for that joint.
6. If saturation is happening too easily, widen `JOINT_LIMITS_RAD` (within
   the URDF's true limits).
7. For the thumb specifically: hold your hand flat with the thumb fully
   relaxed and abducted (NOT pinched, NOT opposed). All three thumb bars
   should read close to 0. If `thumb_cmc_yaw` is non-zero in this pose,
   adjust `_THUMB_REST_AXIS_RIGHT` so the planar component of your relaxed
   metacarpal direction matches it.

See `configs/xhand1_example.py` for how to override retargeting parameters
without touching the source.

## Architecture notes

* The pipeline is intentionally modular — `keypoints.py` defines the contract
  (21-point ordering, wrist-local basis), and every other module consumes
  that contract. You can drop in a different hand-tracking source (MediaPipe,
  vision-based ManoNet, etc.) by emitting `HandKeypoints` objects.
* Loop pacing is done in `run_teleop.py` with a fixed-period sleep; if the
  Leap delivers slower than `--rate`, the loop blocks on `src.get(timeout=...)`
  and frame drops are counted in the diagnostics panel.
* `LeapSource` uses a bounded queue (size 2) so it always serves the newest
  frame to the consumer — appropriate for teleop, where stale data is worse
  than dropped data.
* The thumb is treated separately from the other four fingers, because
  measuring thumb opposition with the same MCP/PIP/DIP scheme as fingers
  produces nonsense (the thumb's intermediate phalanx doesn't really exist).

## Files

```
leap_xhand_teleop/
├── leap_xhand_teleop/
│   ├── __init__.py
│   ├── keypoints.py              # 21-point convention + HandKeypoints dataclass
│   ├── leap_source.py            # Leap Gemini SDK wrapper + mock mode
│   ├── filters.py                # One-Euro filter
│   ├── retarget_xhand1.py        # 21-kp -> 12-DOF mapping (the opinionated module)
│   ├── ros_publisher.py          # roslibpy bridge (toggleable)
│   └── viewer.py                 # Open3D + matplotlib diagnostics
├── examples/
│   ├── run_teleop.py             # main entry point
│   └── test_retargeter.py        # synthetic-pose unit tests
├── configs/
│   └── xhand1_example.py         # example parameter overrides
└── requirements.txt
```
