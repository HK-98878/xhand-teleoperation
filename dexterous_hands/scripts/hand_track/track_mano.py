import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import time
import threading
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import smplx

import traceback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
model_path     = "hand_landmarker.task"
MANO_MODEL_DIR = "./_DATA/data/"   # adjust if needed

# MANO fitting iterations — tune this for speed vs accuracy tradeoff
MANO_ITERATIONS = 15
MANO_LR         = 0.05  # lower LR since we're refining not solving from scratch
HAMER_EVERY_N_FRAMES = 3   # how often to run MANO fitting

# MediaPipe landmark indices for the 21 joints, in MANO order
# MANO joints: wrist(0), index(1-4), middle(5-8), pinky(9-12), ring(13-16), thumb(17-20)
# MediaPipe:   wrist(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
MP_TO_MANO_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 13, 14, 15, 16, 1, 2, 3, 4]

# Panel layout
PANEL_SKELETON_BG = (20, 20, 20)   # dark background for 3D views

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

latest_result = None
result_lock   = threading.Lock()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# MANO joint connections for skeleton drawing (in MANO joint order)
MANO_CONNECTIONS = [
    (0,1),(1,2),(2,3),          # index
    (0,4),(4,5),(5,6),          # middle
    (0,7),(7,8),(8,9),          # pinky
    (0,10),(10,11),(11,12),     # ring
    (0,13),(13,14),(14,15),     # thumb
    (1,4),(4,7),(7,10),(10,13), # palm
]

def draw_landmarks_on_frame(frame, result):
    h, w, _ = frame.shape
    if not result.hand_landmarks:
        return frame
    for hand_landmarks in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame

# ---------------------------------------------------------------------------
# MANO model + fitting
# ---------------------------------------------------------------------------
device = torch.device('cpu')

print("Loading MANO model...")
mano_layer = smplx.create(
    MANO_MODEL_DIR,
    model_type='mano',
    use_pca=False,          # use full 45-dim pose instead of PCA components
    is_rhand=True,
    flat_hand_mean=True,
).to(device)
mano_layer.eval()
print("MANO ready.")

# Freeze shape params — we only optimise pose per frame
MANO_SHAPE = torch.zeros(1, 10, device=device)

class SharedState:
    def __init__(self):
        self.mano_frame_data   = None
        self.mano_frame_lock   = threading.Lock()
        self.latest_mano_joints = None
        self.mano_joints_lock  = threading.Lock()
        self.latest_result     = None
        self.result_lock       = threading.Lock()

state = SharedState()

# MANO gives 16 joints (no fingertips):
# 0=wrist, 1-4=index, 5-8=middle, 9-12=pinky, 13-15=ring, 16=thumb_base
# Remap target to 16 joints by dropping fingertips (indices 4,8,12,16,20 in MANO order)
MANO_16_IDX = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

def mp_landmarks_to_3d(hand_landmarks, img_w, img_h):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    pts_mano = pts[MP_TO_MANO_IDX]   # reorder to MANO 21-joint order
    pts_16   = pts_mano[MANO_16_IDX] # drop fingertips → 16 joints
    return pts_16

class MANOFitter:
    def __init__(self, n_iter=10, lr=0.1):
        self.n_iter = n_iter
        self.lr     = lr
        # Warm-start state — persists between frames
        self.pose       = torch.zeros(1, 45, device=device)
        self.global_rot = torch.zeros(1, 3,  device=device)
        self.transl     = torch.zeros(1, 3,  device=device)

    def fit(self, target_joints_np):
        target_2d = torch.tensor(
            target_joints_np[:, :2], dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Initialise from previous frame's solution
        pose       = nn.Parameter(self.pose.clone())
        global_rot = nn.Parameter(self.global_rot.clone())
        transl     = nn.Parameter(self.transl.clone())

        optimiser = torch.optim.Adam([pose, global_rot, transl], lr=self.lr)

        for _ in range(self.n_iter):
            optimiser.zero_grad()

            out = mano_layer(
                betas=MANO_SHAPE,
                global_orient=global_rot,
                hand_pose=pose,
                transl=transl,
                return_verts=True,
            )

            pred_joints = out.joints
            if pred_joints is None:
                return None

            pred_joints = pred_joints - pred_joints[:, 0:1, :]
            scale = pred_joints[:, 4:5, :].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pred_joints = pred_joints / scale

            pred_2d = pred_joints[:, :, :2]
            loss = torch.nn.functional.mse_loss(pred_2d, target_2d)
            loss = loss + 0.01 * pose.pow(2).mean()

            loss.backward()
            optimiser.step()

        # Save solution for next frame warm-start
        self.pose       = pose.detach().clone()
        self.global_rot = global_rot.detach().clone()
        self.transl     = transl.detach().clone()

        with torch.no_grad():
            final_out = mano_layer(
                betas=MANO_SHAPE,
                global_orient=self.global_rot,
                hand_pose=self.pose,
                transl=self.transl,
                return_verts=True,
            )
            if final_out.joints is None:
                return None
            fitted = final_out.joints[0].cpu().numpy()

        fitted -= fitted[0]
        scale = np.linalg.norm(fitted[4])
        if scale > 1e-6:
            fitted /= scale

        return fitted


fitter = MANOFitter(n_iter=MANO_ITERATIONS, lr=MANO_LR)

# ---------------------------------------------------------------------------
# 3D skeleton panel rendering (orthographic projection)
# ---------------------------------------------------------------------------
def project_orthographic(joints_3d, view='front', panel_size=300, margin=30):
    if view == 'front':
        coords = joints_3d[:, [0, 1]]
    else:
        coords = joints_3d[:, [2, 1]]

    coords = coords.copy()
    coords[:, 1] = -coords[:, 1]

    mn, mx = coords.min(), coords.max()
    rng = max(mx - mn, 1e-6)
    scale  = (panel_size - 2 * margin) / rng
    coords = (coords - mn) * scale + margin

    # Ensure plain Python int tuples for cv2
    return [(int(round(x)), int(round(y))) for x, y in coords]


def draw_skeleton_panel(joints_3d, view, panel_h, panel_w, label):
    """Draw a 3D skeleton as an orthographic projection on a dark panel."""
    panel = np.full((panel_h, panel_w, 3), PANEL_SKELETON_BG, dtype=np.uint8)

    if joints_3d is None or len(joints_3d) < 16:
        cv2.putText(panel, f"MANO {label} (waiting...)", (10, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return panel

    pts = project_orthographic(joints_3d, view=view,
                                panel_size=min(panel_h, panel_w), margin=25)
    for a, b in MANO_CONNECTIONS:
      if a >= len(pts) or b >= len(pts):
          print(f"[draw] bad connection ({a},{b}) — pts len={len(pts)}")
          continue
      cv2.line(panel, pts[a], pts[b], (0, 200, 100), 1)

    # Draw connections
    for a, b in MANO_CONNECTIONS:
        cv2.line(panel, pts[a], pts[b], (0, 200, 100), 1)

    # Draw joints — colour thumb differently for easy debugging
    for i, (px, py) in enumerate(pts):
        color = (50, 150, 255) if i >= 17 else (255, 200, 50)
        cv2.circle(panel, (px, py), 4, color, -1)

    cv2.putText(panel, f"MANO {label}", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# MANO fitting thread
# ---------------------------------------------------------------------------
latest_mano_joints = None
mano_lock          = threading.Lock()
mano_frame_data    = None   # hand_landmarks to process
mano_data_lock     = threading.Lock()
mano_running       = True

def store_result(result, output_image, timestamp_ms):
    with state.result_lock:
        state.latest_result = result

def mano_worker():
    global mano_running
    while mano_running:
        with state.mano_frame_lock:
            data = state.mano_frame_data
            state.mano_frame_data = None

        if data is None:
            time.sleep(0.01)
            continue

        hand_landmarks, img_w, img_h = data
        if hand_landmarks is None or len(hand_landmarks) < 21:
            continue
        
        try:
            t0 = time.time()
            target = mp_landmarks_to_3d(hand_landmarks, img_w, img_h)

            fitted = fitter.fit(target)
            if fitted is not None:
                with state.mano_joints_lock:
                    state.latest_mano_joints = fitted

            dt = (time.time() - t0) * 1000
            print(f"[MANO] fit {dt:.1f}ms")
            with state.mano_joints_lock:
                state.latest_mano_joints = fitted
        except Exception as e:
            print(f"[MANO] Error: {e}")
            traceback.print_exc()  # prints the full stack trace with line numbers


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def make_placeholder(h, w, text):
    panel = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.putText(panel, text, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return panel


def build_display(webcam_frame, mano_joints, panel_h, panel_w):
    """
    Layout:
      [ webcam (full height) | MANO front (top half) ]
      [                      | MANO side  (bot half) ]
    """
    front = draw_skeleton_panel(mano_joints, 'front', panel_h, panel_w, "Front")
    side  = draw_skeleton_panel(mano_joints, 'side',  panel_h, panel_w, "Side")

    right_col = np.vstack([front, side])
    # Ensure webcam matches panel height
    cam_resized = cv2.resize(webcam_frame, (webcam_frame.shape[1], panel_h * 2))
    return np.hstack([cam_resized, right_col])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.9,
    result_callback=store_result,
)

mano_thread = threading.Thread(target=mano_worker, daemon=True)
mano_thread.start()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        ret, frame = cap.read()
        cam_h, cam_w = frame.shape[:2]
        panel_h = cam_h // 2
        panel_w = cam_w // 2

        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result_to_draw = latest_result

            if result_to_draw:
                frame = draw_landmarks_on_frame(frame, result_to_draw)

            # Push to MANO worker every N frames
            frame_count += 1
            with state.result_lock:
                result_to_draw = state.latest_result

            if result_to_draw:
                frame = draw_landmarks_on_frame(frame, result_to_draw)

            if (frame_count % HAMER_EVERY_N_FRAMES == 0
                    and result_to_draw
                    and result_to_draw.hand_landmarks
                    and len(result_to_draw.hand_landmarks[0]) == 21):
                with state.mano_frame_lock:
                    state.mano_frame_data = (
                        result_to_draw.hand_landmarks[0],
                        cam_w, cam_h,
                    )

            with state.mano_joints_lock:
                mano_joints = state.latest_mano_joints

            display = build_display(frame, mano_joints, panel_h, panel_w)
            cv2.imshow("Hand Tracking", display)

            try:
                if (cv2.waitKey(1) & 0xFF == ord('q') or
                        cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1):
                    break
            except cv2.error:
                break

        mano_running = False
        cap.release()
        cv2.destroyAllWindows()