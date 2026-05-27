# XHand Scripts

Control, teleoperation, and deployment scripts for the XHand dexterous robot hand.

## Folder Structure

```
scripts/
├── hand_track/          # Webcam hand tracking pipeline (see hand_track/README.md)
├── hand_recording/      # Data capture for training recordings
├── exports/             # Exported session data
├── raw_recordings/      # Raw recording files
├── haptic_comparison/   # Haptic sensor analysis images
├── teleop.py            # Live teleoperation via webcam hand tracking
├── deploy_vla.py        # Deploy a trained SmolVLA policy on hardware
├── keyframes.py         # Execute named hand pose sequences
├── calib.py             # Monitor tactile sensor contact location
├── capture.py           # One-shot: print current joint positions
├── reset.py             # Reset hand to home and disengage
├── joint_tests.py       # Sweep all joints through their range of motion
├── sensor_reads.py      # Live tactile force visualizer (flat + 3D)
└── plot_joint_log.py    # Plot joint state/target from a deploy_vla CSV log
```

---

## Main Scripts

### `teleop.py`

Live teleoperation: streams webcam frames through WiLoR-mini hand pose estimation, applies one-euro filtering and dex-retargeting, and publishes joint commands to the XHand at 50 Hz via ROSBridge. This is the production teleoperation script that pulls from `hand_track/` as a library.

**Dependencies:** `wilor_mini`, `dex_retargeting`, `opencv-python`, `torch`, `roslibpy`

**Requires:** ROSBridge at `localhost:9090`, `hand_track/urdf/xhand_left_vector.yml`

**Run:**
```bash
# With robot connected
python teleop.py

# Visualization only (no robot commands)
python teleop.py --no-robot
```

**What it does:**
- Three threads: **camera** → **inference** (WiLoR, GPU if available) → **command** (50 Hz filter + retarget + publish)
- One-euro filter parameters: `min_cutoff=0.5 Hz`, `beta=0.05`
- PD controller: `kp=100`, `ki=0`, `kd=1200`
- WiLoR is warmed up on 5 frames before the main loop
- Press `q` to quit

---

### `deploy_vla.py`

Deploys a trained SmolVLA vision-language-action policy on the XHand. Reads observations from two cameras and the haptic tactile sensors, runs the policy inference loop, and sends joint commands at a fixed control rate.

**Dependencies:** `lerobot` (SmolVLA), `roslibpy`, `opencv-python`, `torch`

**Requires:**
- ROSBridge at `localhost:9090`
- UGREEN main camera and C920 wrist camera (identified by `/dev/v4l/by-id/` paths)
- A trained SmolVLA checkpoint directory

**Run:**
```bash
# Activate the lerobot venv first
source /home/hsken/Documents/smolvla-testing/lerobot/.venv/bin/activate

python deploy_vla.py --checkpoint <path/to/checkpoint> --task "pick up the block"
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint PATH` | required | Path to LeRobot checkpoint directory |
| `--task TEXT` | required | Language task description passed to the model |
| `--ros-host HOST` | `localhost` | ROSBridge host |
| `--ros-port PORT` | `9090` | ROSBridge port |
| `--device DEVICE` | `cuda` | Torch inference device |
| `--no-haptic` | off | Use a neutral haptic image instead of live sensor data |
| `--control-hz HZ` | `30.0` | Command loop frequency |
| `--home RAD×12` | flat hand | Home joint positions for reset |
| `--log-file PATH` | auto-named | CSV output for joint state/target logging |

**Observation inputs:**

| Key | Source | Shape |
|---|---|---|
| `observation.state` | ROSBridge joint state | `(12,)` float32 |
| `observation.images.top` | UGREEN main camera | `1920×1080` |
| `observation.images.c920_side` | C920 wrist camera | `1920×1080` |
| `observation.images.haptic` | Tactile sensors → image | `120×580` |

**Architecture:** five threads — two camera captures, one ROS state subscriber, one inference thread (refills an action deque), one command thread (dequeues actions at fixed rate).

**Keyboard controls (no Enter needed):**

| Key | Action |
|---|---|
| `p` | Pause / resume policy |
| `r` | Reset hand to home position |
| `z` | Zero (re-baseline) haptic sensors |
| `q` | Quit and reset to home |

Starts **paused** — press `p` to begin. Joint state and commanded targets are logged to a CSV file every control tick.

---

## Utility Scripts

### `keyframes.py`

Executes a named sequence of hand poses with linear interpolation and per-finger force cutoff. Moves through each keyframe step, halting individual fingers when contact force exceeds the threshold.

**Run:**
```bash
python keyframes.py <keyframe> [--force THRESHOLD]
```

**Available keyframes:** `wide_grasp`, `grasp`, `pinch`, `tight_pinch`

**What it does:**
- Resets the hand to home, prompts for sensor zeroing (`[ENTER]`), then prompts to start (`[ENTER]`)
- Linearly interpolates position to each keyframe target over its duration (seconds)
- Monitors `force_calc` per finger; freezes individual finger joints if the normal force exceeds `--force` (default 15)
- Prompts `[ENTER]` to reset back to home at the end

---

### `calib.py`

Monitors the raw tactile sensor output on a single finger and prints the peak contact location (row, col) whenever normal force exceeds a threshold. Useful for verifying sensor calibration and locating contact during manual tests.

**Run:**
```bash
python calib.py
# Edit FINGER at the top to select which finger to monitor (0=thumb ... 4=pinky)
```

---

### `capture.py`

One-shot script: subscribes to the hand state topic, prints the current 12-DOF joint position array, then exits. Use this to read and save the current hand configuration.

**Run:**
```bash
python capture.py
```

---

### `reset.py`

Moves the hand to the home position with soft gains, then publishes a mode-0 (disengage) command so the fingers can be moved freely by hand. Run this whenever you need to safely release the hand.

**Run:**
```bash
python reset.py
```

Home position: `[0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]` rad

---

### `joint_tests.py`

Sweeps each finger to its joint limits and back in sequence, then does the same for the thumb. Used to verify that all joints are mechanically and electrically functional after assembly or firmware changes.

**Run:**
```bash
python joint_tests.py
```

Sequence: index → middle → ring → pinky (each to max then min), then thumb joints.

---

### `sensor_reads.py`

Live tactile force visualizer. Subscribes to the hand state and renders a matplotlib figure with two rows per finger: a flat parallelogram unwrap (heatmap of normal force + quiver of tangential forces) and a 3D elliptical cylinder view. Updates at ~10 Hz.

**Run:**
```bash
python sensor_reads.py
# Ctrl-C to stop
```

Geometry is calibrated to the physical sensor layout: 12 rows × 10 cols, 4.25 mm axial pitch, 3.53 mm circumferential pitch, with a 0.25-row helix correction.

---

### `plot_joint_log.py`

Plots joint state vs. commanded target from a CSV log produced by `deploy_vla.py`. Shows one subplot per finger (index, middle, ring, pinky) with solid lines for state and dashed lines for target.

**Run:**
```bash
python plot_joint_log.py joint_log_20260514_120000.csv
python plot_joint_log.py joint_log_20260514_120000.csv --save out.png
python plot_joint_log.py joint_log_20260514_120000.csv --title "grasp task run 3"
```

Thumb joints (0–2) are excluded from the plot.
