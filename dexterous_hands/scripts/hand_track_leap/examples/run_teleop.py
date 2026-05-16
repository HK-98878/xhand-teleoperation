#!/usr/bin/env python3
"""
Leap Motion 2 -> XHAND1 teleoperation main script.

Examples:
    # Software-only test, no hardware, no ROS, viewer only:
    python -m examples.run_teleop --mock --no-ros

    # Real Leap, viewer only (still no ROS) -- great for first hardware bring-up:
    python -m examples.run_teleop --no-ros

    # Full pipeline:
    python -m examples.run_teleop --ros-host 192.168.1.10 --log out.csv

    # Headless (e.g. on the robot machine), no GUI, just publish:
    python -m examples.run_teleop --no-viewer
"""
import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np

from leap_xhand_teleop import (
    LeapSource,
    KeypointFilter,
    XHand1Retargeter,
    XHAND1_JOINT_NAMES,
    RosBridgePublisher,
    HandWireframeViewer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Leap Motion 2 -> XHAND1 teleop")
    # Source
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic mock hand data (no Leap hardware needed).")
    p.add_argument("--hand", choices=["left", "right", "any"], default="any",
                   help="Which hand to track if multiple are visible. Default 'any'.")
    p.add_argument("--hmd", action="store_true",
                   help="Set Leap to HMD tracking mode instead of desktop.")
    p.add_argument("--verbose", action="store_true",
                   help="Print Leap event diagnostics (helpful when no data shows up).")
    # Filtering
    p.add_argument("--mincutoff", type=float, default=1.5,
                   help="One-Euro filter min cutoff (Hz). Higher = less smoothing.")
    p.add_argument("--beta", type=float, default=0.05,
                   help="One-Euro filter beta. Higher = more responsive to fast moves.")
    # Output rate
    p.add_argument("--rate", type=float, default=30.0,
                   help="Target loop rate (Hz). Default 30.")
    p.add_argument("--viewer-rate", type=float, default=20.0,
                   help="Max viewer redraw rate (Hz). Default 20. Lower if "
                   "the viewer feels laggy or steals time from the control loop.")
    # ROS
    p.add_argument("--no-ros", action="store_true", help="Disable ROS publishing.")
    p.add_argument("--ros-host", default="localhost")
    p.add_argument("--ros-port", type=int, default=9090)
    p.add_argument("--ros-topic", default="/xhand1/target_joint_positions")
    p.add_argument("--ros-jointstate", action="store_true",
                   help="Publish sensor_msgs/JointState instead of Float64MultiArray.")
    # Viewer
    p.add_argument("--no-viewer", action="store_true", help="Disable all GUI panels.")
    p.add_argument("--no-3d", action="store_true", help="Disable 3D wireframe.")
    p.add_argument("--no-diag", action="store_true", help="Disable diagnostics panel.")
    # Logging
    p.add_argument("--log", type=str, default=None,
                   help="Path to CSV file for logging keypoints + joint commands.")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------ Source -----------------------------
    src = LeapSource(prefer_hand=args.hand, mock=args.mock, hmd_mode=args.hmd,
                     verbose=args.verbose)
    src.start()

    # ------------------------------ Filter -----------------------------
    filt = KeypointFilter(mincutoff=args.mincutoff, beta=args.beta, freq=args.rate)

    # --------------------------- Retargeter ----------------------------
    retarget = XHand1Retargeter()

    # ------------------------------ ROS --------------------------------
    ros = RosBridgePublisher(
        host=args.ros_host, port=args.ros_port, topic=args.ros_topic,
        enabled=not args.no_ros, use_jointstate=args.ros_jointstate,
    )
    ros.start()

    # --------------------------- Viewer --------------------------------
    if args.no_viewer:
        viewer = None
    else:
        viewer = HandWireframeViewer(
            show_3d=not args.no_3d,
            show_diag=not args.no_diag,
        )

    # --------------------------- Logger --------------------------------
    csv_writer = None
    csv_file = None
    if args.log:
        csv_file = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        kp_cols = []
        for i in range(21):
            kp_cols += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]
        csv_writer.writerow(["t", "is_left", "confidence"] + kp_cols + list(XHAND1_JOINT_NAMES))

    # --------------------------- Signal handling ------------------------
    stop_flag = {"v": False}
    def _sigint(*_):
        print("\n[main] caught SIGINT, shutting down...")
        stop_flag["v"] = True
    signal.signal(signal.SIGINT, _sigint)

    # ----------------------------- Main loop ---------------------------
    period = 1.0 / args.rate
    viewer_period = 1.0 / max(args.viewer_rate, 1.0)
    t_next = time.monotonic()
    last_viewer_draw = 0.0
    n_frames = 0
    last_status = time.monotonic()

    print(f"[main] running. mode={'mock' if args.mock else 'live'}, "
          f"ros={'OFF' if args.no_ros else 'ON'}, "
          f"viewer={'OFF' if args.no_viewer else 'ON'}, "
          f"target_rate={args.rate}Hz, viewer_rate={args.viewer_rate}Hz")
    print("[main] press Ctrl-C to stop.")

    if viewer is not None:
        viewer.open_now()
        print("[main] viewer windows opened. Waiting for tracking data...")

    # Cache the most recent frame for the viewer-only redraw path.
    last_smoothed = None
    last_kp = None
    last_joint_cmd = None

    try:
        while not stop_flag["v"]:
            # Always pump GUI events first - keeps windows responsive even
            # when no data is arriving.
            if viewer is not None:
                if not viewer.poll():
                    print("[main] viewer closed, exiting.")
                    break

            kp = src.get(timeout=period)
            now = time.monotonic()
            if kp is None:
                if viewer is not None:
                    viewer.note_drop()
                # Don't sleep more - we already waited `period` seconds in get()
                continue

            # Filter -> retarget -> publish (every frame, full rate)
            smoothed = filt.filter(kp.points, kp.timestamp)
            kp.points = smoothed
            joint_cmd, debug = retarget(kp)
            ros.publish(joint_cmd)

            last_smoothed, last_kp, last_joint_cmd = smoothed, kp, joint_cmd

            # Viewer redraw is rate-limited - it's the most expensive thing
            # in the loop and starves the rest if we update every frame.
            if viewer is not None and (now - last_viewer_draw) >= viewer_period:
                viewer.update(smoothed, kp, joint_cmd)
                last_viewer_draw = now

            if csv_writer is not None:
                row = [kp.timestamp, int(kp.is_left), kp.confidence]
                row += smoothed.flatten().tolist()
                row += joint_cmd.tolist()
                csv_writer.writerow(row)

            n_frames += 1
            if now - last_status > 2.0:
                eff_fps = n_frames / (now - last_status)
                hand_str = "L" if kp.is_left else "R"
                print(f"[main] eff_fps={eff_fps:5.1f} hand={hand_str}  "
                      f"|cmd|_inf={np.abs(joint_cmd).max():.3f} rad  "
                      f"thumb_opp={debug.thumb_opposition:+.2f} rad")
                n_frames = 0
                last_status = now

            # Pace the loop. We don't sleep if we're already behind.
            t_next += period
            if t_next < now: t_next = now + period
            sleep_for = t_next - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        print("[main] cleaning up...")
        src.stop()
        ros.stop()
        if viewer is not None:
            viewer.close()
        if csv_file is not None:
            csv_file.close()
            print(f"[main] log saved to {args.log}")


if __name__ == "__main__":
    main()