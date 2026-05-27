# Webcam-Based Hand Tracking

Webcam-driven hand tracking and retargeting pipeline for the XHand dexterous robot hand. Scripts range from lightweight MediaPipe landmark tracking to full 3D mesh recovery with dex-retargeting.

## Folder Structure

```
hand_track/
├── _DATA/                          # Model weights and data files (not tracked in git)
│   ├── data/
│   │   ├── mano/                   # MANO hand model (MANO_RIGHT.pkl)
│   │   └── mano_mean_params.npz
│   ├── hamer_ckpts/                # HaMeR checkpoint
│   │   ├── checkpoints/hamer.ckpt
│   │   ├── dataset_config.yaml
│   │   └── model_config.yaml
│   └── vitpose_ckpts/
│       └── vitpose+_huge/          # ViTPose wholebody checkpoint
├── hamer/                          # HaMeR submodule - only necessary if using track-hamer.py
├── msgs/
│   └── XHandCommand.msg            # ROS message definition for hand commands
├── tuning/                         # Retargeting tuning tools
│   ├── recordings/                 # Recorded sessions (session_001–006.pkl)
│   ├── custom_retarget.py
│   └── replay.py
├── urdf/                           # XHand robot description
│   ├── meshes/                     # STL meshes for left/right hand links
│   ├── xhand_left.urdf
│   ├── xhand_left_dexpilot.yml
│   └── xhand_left_vector.yml
├── track.py                        # Hand tracking scripts (iterations 1–4)
├── track_hamer.py
├── track_mano.py
├── track_wilor.py
├── track_wilor_retarget.py              # Tracking with retargeting
├── joint_mapping.py                # Joint mapping tests for track.py
├── joint_mapping_2.py
├── utils.py
├── dex_ret_test.py                 # Dexterous retargeting test
├── hand_landmarker.task            # MediaPipe hand landmarker model
└── README.md
```

## Tracking Scripts

All scripts must be run from the `hand_track/` directory. Press `q` or close the window to quit.

---

### `track.py`

The simplest end-to-end script. Uses MediaPipe hand landmarker in live-stream mode to detect 21 landmarks per frame, maps them to 12 XHand joint angles, and publishes commands to the robot over ROSBridge.

**Dependencies:** `mediapipe`, `opencv-python`, `roslibpy`, ROS Noetic (`/opt/ros/noetic`)

**Requires:** `hand_landmarker.task`, ROSBridge running at `localhost:9090`

**Run:**
```bash
python track.py
```

**What it does:**
- Opens webcam (device 0) at the system default resolution
- Runs MediaPipe hand detection asynchronously on every frame
- On each detection, calls `XHandPublisher.send()` which computes joint angles from landmark geometry and publishes a `JointState` command via ROSBridge
- Draws the 21-landmark skeleton (green connections, red dots) on the preview window
- PD controller parameters: `kp=100`, `kd=1200`, `effort_limit=100`, `mode=3`

---

### `track_hamer.py`

Adds HaMeR 3D hand mesh recovery on top of MediaPipe detection. MediaPipe provides the detection bounding box; HaMeR estimates the full MANO mesh from the cropped hand region. Didn't develop to the point of robot publishing — visualization only.

**Dependencies:** `mediapipe`, `opencv-python`, `torch`, `hamer` submodule (see `hamer/setup.py`)

**Requires:** `hand_landmarker.task`, `_DATA/hamer_ckpts/checkpoints/hamer.ckpt`

The MANO model can be downloaded from the [MANO website](https://mano.is.tue.mpg.de/). Use `fetch_demo_data.sh` after fetching the submodule to get the task. If you run into errors when using gdown, use the wget route in the same script instead.


**Run:**
```bash
python track_hamer.py
```

**What it does:**
- MediaPipe runs on every frame for real-time landmark overlay
- Every `HAMER_EVERY_N_FRAMES` frames (default 5), the frame crop is passed to HaMeR in a background thread
- HaMeR runs on CPU (GPU disabled at the top of the script via `CUDA_VISIBLE_DEVICES=''`)
- Display layout: webcam feed on the left, HaMeR front-view and side-view mesh renders stacked on the right
- Handedness is inferred from MediaPipe's `handedness` field (note: MediaPipe reports mirrored labels from a webcam, so `"Left"` label = right hand physically)
- Adjust `RESCALE_FACTOR` if the hand crop is being cut off at the edges

---

### `track_mano.py`

Uses MediaPipe landmarks as 2D targets and fits MANO pose parameters to them via gradient descent (Adam optimizer) on every inference tick. Produces a 3D skeleton that respects the MANO kinematic model rather than raw 2D projections.

**Dependencies:** `mediapipe`, `opencv-python`, `torch`, `smplx`

**Requires:** `hand_landmarker.task`, `_DATA/data/mano/MANO_RIGHT.pkl`

These can be fetched in the same way as for `track_hamer.py`

**Run:**
```bash
python track_mano.py
```

**What it does:**
- MediaPipe provides 21 normalized 3D landmarks per frame
- Every `HAMER_EVERY_N_FRAMES` frames (default 3), landmarks are converted to MANO joint order and passed to the fitting thread
- `MANOFitter` runs `MANO_ITERATIONS=15` Adam steps warm-started from the previous frame's solution, minimizing 2D reprojection error with a small pose regularization term
- Display layout: webcam feed on the left, MANO front and side orthographic skeleton projections on the right (dark panels, green bones)
- Shape parameters (`betas`) are fixed at zero — only pose is optimized per frame
- Tune `MANO_ITERATIONS` and `MANO_LR` to trade off latency vs accuracy

---

### `track_wilor.py`

Uses [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for direct 3D keypoint estimation. More accurate than MANO fitting for fast motion. Includes session recording. Does not have a command/retargeting thread — use `track_wilor_retarget.py` for robot control.

**Dependencies:** `wilor_mini`, `dex_retargeting`, `opencv-python`, `torch`

**Requires:** WiLoR-mini model weights (downloaded automatically on first run)

**Run:**
```bash
# Basic visualization
python track_wilor.py

# Record a session to pickle
python track_wilor.py --record tuning/recordings/session_007.pkl --max-frames 600
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--record PATH` | None | Save recorded frames to a pickle file |
| `--max-frames N` | 600 | Stop recording after N frames (~30s at 20 Hz) |

**What it does:**
- Three threads: **camera** (captures 640×480), **inference** (runs WiLoR, GPU if available), **main** (display)
- WiLoR is warmed up on 5 frames before the main loop to avoid first-frame latency spikes
- Inference runs on every new frame; the main thread always displays the latest filtered pose
- Overlay: per-finger colored skeleton bones (thumb=red, index=orange, middle=green, ring=purple, pinky=blue) with a HUD showing pose age and inference latency
- Press `s` to print a timing stats snapshot; press `q` to quit and print final stats
- Each recorded frame stores `joints_3d (21,3)`, `kpts_2d (21,2)`, `is_right`, `capture_ts`

---

### `track_wilor_retarget.py`

Full tracking and retargeting pipeline. WiLoR inference + one-euro filtering + dex-retargeting onto the XHand left hand, running at 50 Hz in a dedicated command thread. Robot publishing is stubbed.

**Dependencies:** `wilor_mini`, `dex_retargeting`, `opencv-python`, `torch`

**Requires:** WiLoR-mini model weights, `urdf/xhand_left_vector.yml`, `urdf/xhand_left.urdf`

**Run:**
```bash
python track_wilor_retarget.py
```

**What it does:**
- Four threads: **camera** → **inference** → **command** → **main**
- The **command thread** ticks at 50 Hz: applies one-euro filtering to the 21×3 MANO keypoints, rotates from MANO frame to XHand URDF frame, then calls `dex_retargeting` to solve 12 target joint angles
- Joint angles are clipped to hardware limits; `!! at limit` appears in the HUD if any joint is saturating
- Retargeting frame rotation: MANO (fingers along +x, palm +z) → XHand URDF (fingers along -z, palm +y)
- One-euro filter parameters: `min_cutoff=0.5 Hz`, `beta=0.05` — increase `beta` to reduce lag during fast motion at the cost of more jitter
- Per-stage timing stats are printed every 2 seconds and dumped on quit
- Press `s` for a midrun stats snapshot; press `q` to quit

**XHand joint order (12 DOF):**

| Index | Joint |
|---|---|
| 0 | `thumb_bend_joint` |
| 1 | `thumb_rota_joint1` |
| 2 | `thumb_rota_joint2` |
| 3 | `index_bend_joint` |
| 4 | `index_joint1` |
| 5 | `index_joint2` |
| 6 | `mid_joint1` |
| 7 | `mid_joint2` |
| 8 | `ring_joint1` |
| 9 | `ring_joint2` |
| 10 | `pinky_joint1` |
| 11 | `pinky_joint2` |

---

## Supporting Files

### `joint_mapping.py` / `joint_mapping_2.py`

Joint angle computation and ROS publishing for `track.py`. `joint_mapping_2.py` is the current version — it computes finger bend angles from MediaPipe landmark geometry, applies joint limits, and publishes via `roslibpy` to the ROSBridge websocket. `joint_mapping.py` is an earlier iteration kept for reference.

### `utils.py`

Shared utilities: one-euro filter implementation (`OneEuroScalar`, `OneEuroVector`) and the `XHandPublisher` base class used by the tracking scripts.

### `dex_ret_test.py`

Standalone install validation for `dex_retargeting`. Builds the XHand retargeter from `urdf/xhand_left_vector.yml` and runs it against a synthetic flat-hand pose. Use this to confirm the retargeting stack is correctly installed before running the full pipeline.

```bash
python dex_ret_test.py
```

## Retargeting tuning

Tools for replaying and adjusting recorded WiLoR data sessions offline. `replay.py` loads a pickle session and visualizes it. `custom_retarget.py` allows tuning the retargeting parameters against recorded data without needing live hardware.

**Dependencies:** `sapien`