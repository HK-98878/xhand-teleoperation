#!/usr/bin/env python3
"""
Calibrate the thumb rest axis to your hand.

Hold your hand FLAT and RELAXED (palm down, fingers extended, thumb extended
out to the side -- NOT pinching, NOT crossing the palm) over the Leap camera.
The script averages the proximal-phalanx direction over a few seconds and
prints the values you should paste into retarget_xhand1.py as
_THUMB_REST_AXIS_RIGHT (or LEFT).

Also prints the diagnostic ranges of `pinch_flex` and `cross_palm` while you
move your thumb through its full range, so you can verify both signals are
smooth and have sensible magnitudes.

Run:
    python -m examples.calibrate_thumb_rest --duration 3
    # then move your thumb through pinch / cross-palm / extension for ~10s
"""
import argparse
import time
import numpy as np

from leap_xhand_teleop import LeapSource, KeypointFilter, XHand1Retargeter
from leap_xhand_teleop.keypoints import KP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", choices=["left", "right", "any"], default="any")
    p.add_argument("--duration", type=float, default=3.0,
                   help="Seconds to capture the rest pose.")
    p.add_argument("--explore", type=float, default=10.0,
                   help="Seconds to capture full range of motion afterward.")
    args = p.parse_args()

    src = LeapSource(prefer_hand=args.hand, verbose=False)
    src.start()
    filt = KeypointFilter(mincutoff=2.0, beta=0.05, freq=30.0)

    # ===== Phase 1: capture rest pose =====
    print()
    print("=" * 70)
    print("PHASE 1: Hold your hand FLAT and RELAXED with the thumb")
    print("extended out to the side (NOT pinching, NOT opposed).")
    print(f"Capturing for {args.duration:.1f} seconds. Don't move...")
    print("=" * 70)
    print("Press Enter when ready...")
    input()

    samples = []
    is_left_votes = []
    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end:
        kp = src.get(timeout=0.05)
        if kp is None:
            continue
        smoothed = filt.filter(kp.points, kp.timestamp)
        kp.points = smoothed
        local = kp.wrist_local()

        # We want the proximal phalanx direction in wrist-local frame,
        # projected onto the palm plane (xz).
        v_prox = local[KP.THUMB_IP] - local[KP.THUMB_MCP]
        planar = np.array([v_prox[0], 0.0, v_prox[2]])
        n = np.linalg.norm(planar)
        if n > 1e-6:
            samples.append(planar / n)
            is_left_votes.append(int(kp.is_left))

    if not samples:
        print("ERROR: no thumb samples captured. Was your hand visible?")
        src.stop()
        return

    avg = np.mean(samples, axis=0)
    avg /= np.linalg.norm(avg) + 1e-9
    is_left = np.mean(is_left_votes) > 0.5

    print()
    print(f"Captured {len(samples)} frames.")
    print(f"Hand chirality detected: {'LEFT' if is_left else 'RIGHT'}")
    print(f"Mean proximal-phalanx direction (palm-plane, wrist-local):")
    print(f"    [{avg[0]:.4f}, 0.0, {avg[2]:.4f}]")
    print()
    print("Paste this into leap_xhand_teleop/retarget_xhand1.py:")
    if is_left:
        print(f"_THUMB_REST_AXIS_LEFT  = np.array([{avg[0]:.4f}, 0.0, {avg[2]:.4f}], dtype=np.float32)")
    else:
        print(f"_THUMB_REST_AXIS_RIGHT = np.array([{avg[0]:.4f}, 0.0, {avg[2]:.4f}], dtype=np.float32)")

    # ===== Phase 2: range of motion sweep =====
    print()
    print("=" * 70)
    print("PHASE 2: Now move your thumb through its full range:")
    print("  - Fully extended out to the side")
    print("  - Folded forward into a pinch with index/middle")
    print("  - Crossed over the palm (thumbpad facing palm)")
    print("  - Fully bent at the IP joint")
    print(f"Recording for {args.explore:.1f} seconds...")
    print("=" * 70)
    print("Press Enter when ready...")
    input()

    # Apply the captured rest axis to the retargeter for this phase.
    rt = XHand1Retargeter()
    if is_left:
        from leap_xhand_teleop.retarget_xhand1 import _THUMB_REST_AXIS_LEFT
        # Override (not supported via constructor, monkey-patch the module)
        import leap_xhand_teleop.retarget_xhand1 as rmod
        rmod._THUMB_REST_AXIS_LEFT[:] = avg.astype(np.float32)
    else:
        import leap_xhand_teleop.retarget_xhand1 as rmod
        rmod._THUMB_REST_AXIS_RIGHT[:] = avg.astype(np.float32)

    pinch_history = []
    cross_history = []
    mid_history = []
    t_end = time.monotonic() + args.explore
    while time.monotonic() < t_end:
        kp = src.get(timeout=0.05)
        if kp is None:
            continue
        smoothed = filt.filter(kp.points, kp.timestamp)
        kp.points = smoothed
        j, dbg = rt(kp)
        pinch_history.append(dbg.thumb_opposition)  # actually pinch_flex
        cross_history.append(dbg.thumb_pitch)        # actually cross_palm
        mid_history.append(dbg.thumb_flex)            # actually mid_bend

    if pinch_history:
        pa = np.array(pinch_history)
        ca = np.array(cross_history)
        ma = np.array(mid_history)
        print()
        print("Range of motion summary (raw signals, before gain/clip):")
        print(f"  pinch_flex : min={pa.min():+.3f}  max={pa.max():+.3f}  range={pa.max()-pa.min():.3f} rad")
        print(f"  cross_palm : min={ca.min():+.3f}  max={ca.max():+.3f}  range={ca.max()-ca.min():.3f} rad")
        print(f"  mid_bend   : min={ma.min():+.3f}  max={ma.max():+.3f}  range={ma.max()-ma.min():.3f} rad")
        print()
        # Detect rapid jumps that indicate atan2 wrap-around or bad reference.
        for name, arr in [("pinch_flex", pa), ("cross_palm", ca), ("mid_bend", ma)]:
            if len(arr) < 2:
                continue
            diffs = np.abs(np.diff(arr))
            big_jumps = (diffs > 1.0).sum()
            if big_jumps > 0:
                print(f"  WARN: {name} had {big_jumps} jumps > 1 rad/frame "
                      f"(max jump {diffs.max():.2f}). Indicates wrap-around or unstable reference.")

    src.stop()


if __name__ == "__main__":
    main()