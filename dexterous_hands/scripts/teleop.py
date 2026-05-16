from typing import Dict
import threading
import cv2
import numpy as np
import time
import torch
import argparse

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

from hand_track.track4 import LatestFrame, LatestPose, StageStats, R_MANO_TO_XHAND, camera_thread, inference_thread, suppress_output, draw_overlay
from hand_track.tuning.custom_retarget import retarget_xhand
from hand_track.utils import XHandPublisher, OneEuroVector

def command_thread(latest_pose: LatestPose,
                   filtered_pose : LatestPose,
                   stop_event : threading.Event,
                   target_hz : float = 50.0,
                   filter_params : dict | None = None,
                   disable_ros : bool = False,
                   hand_params: dict | None = None):
    """Reads latest_pose at fixed rate; filters, rotates, retargets;
    writes filtered_pose with both filtered keypoints (for visualisation)
    and retargeted qpos (for robot)."""
    period = 1.0 / target_hz
    filter_params = filter_params or dict(min_cutoff=0.5, beta=0.05)
    kpts3d_filter = OneEuroVector(**filter_params)
    kpts2d_filter = OneEuroVector(**filter_params)

    next_tick = time.perf_counter()

    if not disable_ros:
        if hand_params is None: hand_params = dict()
        pub = XHandPublisher(**hand_params)
    else:
        pub = None

    while not stop_event.is_set():
        now = time.perf_counter()
        dt = next_tick - now
        if dt > 0:
            time.sleep(dt)
        next_tick += period
        if next_tick < time.perf_counter() - period:
            next_tick = time.perf_counter() + period

        with latest_pose.lock:
            if latest_pose.kpts_3d is None or latest_pose.kpts_2d is None:
                continue
            kpts_3d = latest_pose.kpts_3d
            kpts_2d = latest_pose.kpts_2d
            capture_ts = latest_pose.capture_ts
            inference_ts = latest_pose.inference_ts
        
        t = time.perf_counter()
        filtered_kpts_3d = kpts3d_filter(kpts_3d, t)
        filtered_kpts_2d = kpts2d_filter(kpts_2d, t)
        qpos = retarget_xhand(filtered_kpts_3d)
        qpos_clipped = np.clip(qpos,XHandPublisher.JOINT_LIMITS[:,0],XHandPublisher.JOINT_LIMITS[:,1])

        with filtered_pose.lock:
            filtered_pose.kpts_3d = filtered_kpts_3d
            filtered_pose.kpts_2d = filtered_kpts_2d
            filtered_pose.qpos = qpos_clipped
            filtered_pose.capture_ts = capture_ts
            filtered_pose.inference_ts = inference_ts

        if pub is not None:
            pub.send(qpos_clipped.tolist())

    if pub is not None:
        pub.shutdown()

def main():
    parser = argparse.ArgumentParser(description="XHand keyframe controller")
    parser.add_argument(
        "--no-robot",
        action="store_true",
        default=False,
        help="Force threshold for movement cutoff"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[init] device={device}, dtype={dtype}")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    stages: Dict[str, StageStats] = {
        name: StageStats(name) for name in [
            "camera",       # camera thread per-frame
            "inference",    # inference thread per-prediction
            "infer_idle",   # inference thread waiting for new frames
            "command",      # command thread per-tick (filter+retarget+write)
            "retarget",     # subset of command: retargeting alone
            "viz_2d",       # main thread overlay drawing
            "loop_total",   # main thread per-iteration
        ]
    }

    latest_frame = LatestFrame()
    latest_pose = LatestPose()
    filtered_pose = LatestPose()
    stop_event = threading.Event()

    # Warmup the WiLoR model so first-frame jitter doesn't pollute stats.
    print("[warmup] priming WiLoR...")
    cap_warm = cv2.VideoCapture(0)
    for _ in range(5):
        ok, f = cap_warm.read()
        if ok:
            with suppress_output():
                _ = pipe.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap_warm.release()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[warmup] done.")

    filter_params = dict(min_cutoff=0.5, beta=0.05)
    hand_params = dict(kp=100, ki=0, kd=1200)

    threads = [
        threading.Thread(target=camera_thread, name="camera", daemon=True,
                          args=(latest_frame, stop_event, stages["camera"])),
        threading.Thread(target=inference_thread, name="inference", daemon=True,
                          args=(pipe, latest_frame, latest_pose, stop_event,
                                stages["inference"], stages["infer_idle"])),
        threading.Thread(target=command_thread, name="command", daemon=True,
                          args=(latest_pose, filtered_pose, stop_event),
                          kwargs=dict(target_hz=50.0,
                                      disable_ros = args.no_robot,
                                      filter_params=filter_params,
                                      hand_params=hand_params)),
    ]
    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            with latest_frame.lock:
                bgr = (latest_frame.bgr.copy()
                        if latest_frame.bgr is not None else None)
            with filtered_pose.lock:
                fp_kpts_2d = (filtered_pose.kpts_2d.copy()
                              if filtered_pose.kpts_2d is not None else None)
                fp_capture = filtered_pose.capture_ts
                fp_inf_ts = filtered_pose.inference_ts

            if bgr is not None:
                hud = []
                if fp_capture:
                    age_ms = (time.perf_counter() - fp_capture) * 1000.0
                    hud.append(f"pose age {age_ms:5.1f} ms")
                if fp_capture and fp_inf_ts:
                    inf_lat_ms = (fp_inf_ts - fp_capture) * 1000.0
                    hud.append(f"infer lat {inf_lat_ms:5.1f} ms")

                if fp_kpts_2d is not None:
                    bgr = draw_overlay(bgr, fp_kpts_2d, hud_lines=hud)

                cv2.imshow("WiLoR teleop", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()