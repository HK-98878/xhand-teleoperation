#!/usr/bin/env python3
"""
Sanity-check the XHAND1 retargeter on synthetic poses.

Each test constructs a known hand pose (in WORLD frame, where wrist=origin,
palm-forward=+x, palm-normal=+y) and asserts that the retargeter produces
the expected joint values. This is the fastest way to catch sign-convention
or axis-confusion bugs without hardware.

Run:    python -m examples.test_retargeter
"""
import numpy as np
from leap_xhand_teleop.keypoints import HandKeypoints, KP
from leap_xhand_teleop.retarget_xhand1 import XHand1Retargeter, XHAND1_JOINT_NAMES


def _flat_open_hand() -> np.ndarray:
    """Right hand, palm flat, fingers extended along +x, thumb extended out
    to the +z (thumb) side at ~45 deg.

    Middle MCP is exactly on +x so wrist_local() resolves to identity. With
    thumb at the rest axis, the new retargeter should output ~0 everywhere.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[KP.WRIST] = [0.0, 0.0, 0.0]

    # Thumb proximal phalanx at ~45 deg (the rest axis), distal extended
    # along the same direction (no IP curl, no thumbpad twist).
    # CMC and MCP are co-located (Leap pads metacarpal as zero-length).
    pts[KP.THUMB_CMC] = [0.025, 0.0, 0.025]
    pts[KP.THUMB_MCP] = [0.025, 0.0, 0.025]
    # proximal phalanx along (+x, +z) at 45 deg, length ~30 mm
    pts[KP.THUMB_IP]  = [0.025 + 0.0212, 0.0, 0.025 + 0.0212]
    # distal continues along same direction (no IP curl)
    pts[KP.THUMB_TIP] = [0.025 + 0.0354, 0.0, 0.025 + 0.0354]

    fingers = [
        ("index",  KP.INDEX_MCP,  KP.INDEX_PIP,  KP.INDEX_DIP,  KP.INDEX_TIP,   0.020),
        ("middle", KP.MIDDLE_MCP, KP.MIDDLE_PIP, KP.MIDDLE_DIP, KP.MIDDLE_TIP,  0.000),
        ("ring",   KP.RING_MCP,   KP.RING_PIP,   KP.RING_DIP,   KP.RING_TIP,   -0.020),
        ("pinky",  KP.PINKY_MCP,  KP.PINKY_PIP,  KP.PINKY_DIP,  KP.PINKY_TIP,  -0.040),
    ]
    for name, mcp, pip, dip, tip, z in fingers:
        pts[mcp] = [0.085, 0.0, z]
        pts[pip] = [0.115, 0.0, z]
        pts[dip] = [0.140, 0.0, z]
        pts[tip] = [0.160, 0.0, z]
    return pts


def _curled_index() -> np.ndarray:
    pts = _flat_open_hand()
    pts[KP.INDEX_PIP] = [0.115, 0.0, 0.020]
    pts[KP.INDEX_DIP] = [0.115, -0.025, 0.020]
    pts[KP.INDEX_TIP] = [0.100, -0.040, 0.020]
    return pts


def _pinching_thumb() -> np.ndarray:
    """Thumb folded forward toward a pinch with index/middle fingers."""
    pts = _flat_open_hand()
    pts[KP.THUMB_IP]  = [0.025 + 0.030, 0.0, 0.025 + 0.005]
    pts[KP.THUMB_TIP] = [0.025 + 0.050, 0.0, 0.025 + 0.008]
    return pts


def _crossed_thumb() -> np.ndarray:
    """Thumb still extended along rest axis BUT thumbpad rotated to face palm.

    Proximal phalanx stays along the rest axis so pinch_flex stays ~0, but the
    distal phalanx bends across the palm (toward -z) instead of down (-y),
    indicating the thumb has rotated about its own length axis.
    """
    pts = _flat_open_hand()
    pts[KP.THUMB_IP] = [0.025 + 0.0212, 0.0, 0.025 + 0.0212]
    ip = pts[KP.THUMB_IP]
    pts[KP.THUMB_TIP] = [ip[0] + 0.005, 0.0, ip[2] - 0.025]
    return pts


def _mid_bent_thumb() -> np.ndarray:
    """Thumb proximal at rest, IP joint curled ~90 deg (distal bends down)."""
    pts = _flat_open_hand()
    pts[KP.THUMB_IP]  = [0.025 + 0.0212, 0.0, 0.025 + 0.0212]
    ip = pts[KP.THUMB_IP]
    pts[KP.THUMB_TIP] = [ip[0], ip[1] - 0.025, ip[2]]
    return pts


def _make_kp(points: np.ndarray) -> HandKeypoints:
    return HandKeypoints(
        points=points, is_left=False, timestamp=0.0, confidence=1.0,
        palm_normal=np.array([0, 1, 0], dtype=np.float32),
        palm_direction=np.array([1, 0, 0], dtype=np.float32),
    )


def _pretty(joints):
    return ", ".join(f"{n[:14]:>14s}={v:+.3f}" for n, v in zip(XHAND1_JOINT_NAMES, joints))


def run_tests():
    rt = XHand1Retargeter()
    print("=" * 80)

    # Test 1: flat hand should produce ~zero everywhere
    kp = _make_kp(_flat_open_hand())
    j, dbg = rt(kp)
    print("\n[Test 1] flat open hand, thumb at rest (expect all near zero)")
    print("  joints:", _pretty(j))
    assert np.abs(j).max() < 0.30, f"flat hand should be near-zero, got max={np.abs(j).max()}"
    print(f"  PASS (max abs joint = {float(np.abs(j).max()):.4f} rad)")

    # Test 2: curled index
    kp = _make_kp(_curled_index())
    j, dbg = rt(kp)
    print("\n[Test 2] curled index (PIP ~90 deg, DIP ~45 deg)")
    print("  joints:", _pretty(j))
    assert j[5] > 1.0, f"expected significant index PIP curl, got {j[5]}"
    assert abs(j[7]) < 0.3 and abs(j[9]) < 0.3 and abs(j[11]) < 0.3, \
        f"non-index PIPs should remain ~0: {j[[7, 9, 11]]}"
    print("  PASS")

    # Test 3: thumb pinch flex
    kp = _make_kp(_pinching_thumb())
    j, dbg = rt(kp)
    print("\n[Test 3] thumb folded toward pinch (expect thumb_pinch_flex > 0)")
    print(f"  thumb_pinch_flex (J0): {j[0]:+.3f}")
    print(f"  thumb_cross_palm (J1): {j[1]:+.3f}  (expect ~0)")
    print(f"  thumb_mid_bend   (J2): {j[2]:+.3f}  (expect ~0)")
    assert j[0] > 0.3, f"expected positive pinch flex, got {j[0]}"
    assert abs(j[2]) < 0.3, f"mid_bend should be near zero, got {j[2]}"
    print("  PASS")

    # Test 4: thumb cross-palm
    kp = _make_kp(_crossed_thumb())
    j, dbg = rt(kp)
    print("\n[Test 4] thumbpad rotated to face palm (expect thumb_cross_palm > 0)")
    print(f"  thumb_pinch_flex (J0): {j[0]:+.3f}  (expect ~0)")
    print(f"  thumb_cross_palm (J1): {j[1]:+.3f}")
    print(f"  thumb_mid_bend   (J2): {j[2]:+.3f}  (some IP bend is OK here)")
    assert j[1] > 0.5, f"expected positive cross_palm, got {j[1]}"
    print("  PASS")

    # Test 5: thumb IP mid-bend
    kp = _make_kp(_mid_bent_thumb())
    j, dbg = rt(kp)
    print("\n[Test 5] thumb IP curled ~90 deg (expect thumb_mid_bend > 0)")
    print(f"  thumb_pinch_flex (J0): {j[0]:+.3f}  (expect ~0)")
    print(f"  thumb_cross_palm (J1): {j[1]:+.3f}  (expect ~0; this IS the reference curl direction)")
    print(f"  thumb_mid_bend   (J2): {j[2]:+.3f}")
    assert j[2] > 0.5, f"expected significant mid_bend, got {j[2]}"
    print("  PASS")

    print("\n" + "=" * 80)
    print("All retargeter sanity tests passed.")


if __name__ == "__main__":
    run_tests()