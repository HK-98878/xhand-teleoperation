import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import time
import threading
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import sys

# ---------------------------------------------------------------------------
# HaMeR imports — no detectron2 or ViTPose needed
# ---------------------------------------------------------------------------
HAMER_DIR = Path(__file__).parent / "hamer"
sys.path.insert(0, str(HAMER_DIR))

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE    = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_MEAN  = np.array([123.675, 116.280, 103.530], dtype=np.float32)
DEFAULT_STD   = np.array([58.395,  57.120,  57.375],  dtype=np.float32)
RESCALE_FACTOR = 2.0   # padding around the hand bbox — increase if hand is cut off

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
model_path = "hand_landmarker.task"

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

def store_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

# ---------------------------------------------------------------------------
# HaMeR setup
# ---------------------------------------------------------------------------
print("Loading HaMeR model...")
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu")
hamer_model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
hamer_model = hamer_model.to(device)
hamer_model.eval()
renderer = Renderer(model_cfg, faces=hamer_model.mano.faces)
IMG_SIZE = model_cfg.MODEL.IMAGE_SIZE   # typically 256
print(f"HaMeR ready on {device}. Image size: {IMG_SIZE}")

# ---------------------------------------------------------------------------
# Crop preprocessing — replicates ViTDetDataset without detectron2
# ---------------------------------------------------------------------------

def landmarks_to_bbox(hand_landmarks, img_w, img_h):
    """Get pixel bbox [x1,y1,x2,y2] from MediaPipe landmarks."""
    xs = [lm.x * img_w for lm in hand_landmarks]
    ys = [lm.y * img_h for lm in hand_landmarks]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def crop_hand(frame_rgb, bbox, rescale=RESCALE_FACTOR, img_size=IMG_SIZE):
    """
    Crop and preprocess a hand region for HaMeR, replicating ViTDetDataset.
    Returns:
        img_tensor  : (3, img_size, img_size) float32 normalised tensor
        box_center  : (2,) float32
        box_size    : float32 scalar (side length of the square crop)
        img_size_wh : (2,) int  [W, H] of original frame
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox

    # Square crop centred on bbox, padded by rescale_factor
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    box_size = max(x2 - x1, y2 - y1) * rescale

    # Clamp to image
    x1c = max(0, int(cx - box_size / 2))
    y1c = max(0, int(cy - box_size / 2))
    x2c = min(w, int(cx + box_size / 2))
    y2c = min(h, int(cy + box_size / 2))

    crop = frame_rgb[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, None, None, None

    # Resize to model input size
    crop_resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalise: (pixel - mean) / std
    crop_f = crop_resized.astype(np.float32)
    crop_f = (crop_f - DEFAULT_MEAN) / DEFAULT_STD

    # (H,W,3) → (3,H,W)
    img_tensor = torch.from_numpy(crop_f).permute(2, 0, 1)

    box_center = np.array([cx, cy], dtype=np.float32)
    return img_tensor, box_center, float(box_size), np.array([w, h], dtype=np.float32)


def build_hamer_batch(img_tensor, box_center, box_size, img_size_wh, is_right):
    """Package crop into the dict format HaMeR expects."""
    batch = {
        'img':        img_tensor.unsqueeze(0),           # (1,3,H,W)
        'box_center': torch.tensor(box_center).unsqueeze(0),   # (1,2)
        'box_size':   torch.tensor([box_size]),                 # (1,)
        'img_size':   torch.tensor(img_size_wh).unsqueeze(0),  # (1,2)
        'right':      torch.tensor([float(is_right)]),          # (1,)
        'personid':   torch.tensor([0]),
    }
    return batch


# ---------------------------------------------------------------------------
# HaMeR inference thread
# ---------------------------------------------------------------------------
latest_hamer_front = None
latest_hamer_side  = None
hamer_lock         = threading.Lock()
hamer_frame_data   = None   # (frame_rgb, hand_landmarks, is_right)
hamer_data_lock    = threading.Lock()
hamer_running      = True

HAMER_EVERY_N_FRAMES = 5   # increase if GPU can't keep up


def run_hamer(frame_rgb, hand_landmarks, is_right):
    """Run HaMeR on a single hand crop. Returns (front_panel, side_panel) float32 [0,1] RGB."""
    h, w = frame_rgb.shape[:2]
    bbox = landmarks_to_bbox(hand_landmarks, w, h)

    img_tensor, box_center, box_size, img_size_wh = crop_hand(frame_rgb, bbox)
    if img_tensor is None:
        return None, None

    # Mirror x coords for left hand (HaMeR trained on right hands)
    if not is_right:
        img_tensor = torch.flip(img_tensor, dims=[2])

    batch = build_hamer_batch(img_tensor, box_center, box_size, img_size_wh, is_right)
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = hamer_model(batch)

    # Camera translation
    multiplier = (2 * batch['right'] - 1)
    pred_cam   = out['pred_cam']
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]

    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE
        * batch['img_size'].float().max()
    )
    pred_cam_t = cam_crop_to_full(
        pred_cam,
        batch['box_center'].float(),
        batch['box_size'].float(),
        batch['img_size'].float(),
        scaled_focal_length,
    ).detach().cpu().numpy()

    verts   = out['pred_vertices'][0].detach().cpu().numpy()
    cam_t   = pred_cam_t[0]

    # Flip x for left hand
    if not is_right:
        verts[:, 0] = -verts[:, 0]

    # --- Front view: mesh on white background ---
    white_img_tensor = (
        torch.ones(3, IMG_SIZE, IMG_SIZE)
        - torch.tensor(DEFAULT_MEAN / 255.0)[:, None, None]
    ) / torch.tensor(DEFAULT_STD / 255.0)[:, None, None]

    front_raw = renderer(
        verts, cam_t,
        white_img_tensor,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
    )
    front_panel = front_raw  # float32 RGB [0,1]

    # --- Side view ---
    side_raw = renderer(
        verts, cam_t,
        white_img_tensor,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        side_view=True,
    )
    side_panel = side_raw

    return front_panel, side_panel


def hamer_worker():
    global latest_hamer_front, latest_hamer_side, hamer_running

    while hamer_running:
        with hamer_data_lock:
            data = hamer_frame_data

        if data is None:
            time.sleep(0.01)
            continue

        # Clear slot
        with hamer_data_lock:
            globals()['hamer_frame_data'] = None

        frame_rgb, hand_landmarks, is_right = data
        try:
            front, side = run_hamer(frame_rgb, hand_landmarks, is_right)
            if front is not None:
                with hamer_lock:
                    globals()['latest_hamer_front'] = front
                    globals()['latest_hamer_side']  = side
        except Exception as e:
            print(f"[HaMeR] {e}")

        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def make_placeholder(h, w, text):
    panel = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.putText(panel, text, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return panel


def float_panel_to_bgr(panel_float, h, w):
    """Convert float32 RGB [0,1] → uint8 BGR, resized to (w,h)."""
    bgr = (panel_float[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(bgr, (w, h))


def build_display(webcam_frame, front_panel, side_panel, panel_h, panel_w):
    """
    Layout:
      [ webcam (full height W) | front_panel (top half) ]
      [                        | side_panel  (bot half) ]
    """
    front_bgr = (float_panel_to_bgr(front_panel, panel_h, panel_w)
                 if front_panel is not None
                 else make_placeholder(panel_h, panel_w, "HaMeR front (waiting...)"))
    side_bgr  = (float_panel_to_bgr(side_panel, panel_h, panel_w)
                 if side_panel is not None
                 else make_placeholder(panel_h, panel_w, "HaMeR side  (waiting...)"))

    # Add labels
    for img, label in [(front_bgr, "Front"), (side_bgr, "Side")]:
        cv2.putText(img, f"HaMeR {label}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

    right_col = np.vstack([front_bgr, side_bgr])
    return np.hstack([webcam_frame, right_col])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.9,
    result_callback=store_result,
)

hamer_thread = threading.Thread(target=hamer_worker, daemon=True)
hamer_thread.start()

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

            # Push to HaMeR worker every N frames
            frame_count += 1
            if (frame_count % HAMER_EVERY_N_FRAMES == 0
                    and result_to_draw
                    and result_to_draw.hand_landmarks):

                hand_lm = result_to_draw.hand_landmarks[0]

                # Determine handedness: MediaPipe reports mirrored, so flip
                is_right = True
                if result_to_draw.handedness:
                    label = result_to_draw.handedness[0][0].category_name
                    is_right = (label == "Left")  # mirrored camera: Left label = right hand

                with hamer_data_lock:
                    globals()['hamer_frame_data'] = (frame_rgb.copy(), hand_lm, is_right)

            with hamer_lock:
                front = latest_hamer_front
                side  = latest_hamer_side

            display = build_display(frame, front, side, panel_h, panel_w)
            cv2.imshow("Hand Tracking", display)

            try:
                if (cv2.waitKey(1) & 0xFF == ord('q') or
                        cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1):
                    break
            except cv2.error:
                break

        hamer_running = False
        cap.release()
        cv2.destroyAllWindows()