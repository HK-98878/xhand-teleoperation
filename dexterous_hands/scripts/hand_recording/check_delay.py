import json, numpy as np
from pathlib import Path

SESSION = Path("raw_recordings/session_20260514_174548")  # edit me

# 1. Rosbridge queue lag
recs = [json.loads(l) for l in open(SESSION / "robot_state.jsonl")]
deltas_ms = [(r["robot_timestamp_ns"] - r["sensor_timestamp_ns"]) / 1e6
             for r in recs
             if r.get("sensor_timestamp_ns", 0) > 0]
if deltas_ms:
    print(f"Rosbridge delay (sensor→host receipt):")
    print(f"  n={len(deltas_ms)}")
    print(f"  median: {np.median(deltas_ms):7.1f}ms")
    print(f"  p95:    {np.percentile(deltas_ms, 95):7.1f}ms")
    print(f"  max:    {max(deltas_ms):7.1f}ms")
    print(f"  std:    {np.std(deltas_ms):7.1f}ms")
else:
    print("No sensor_timestamp_ns available — can't compute rosbridge delay")

# 2. Camera frame interval stats (per camera)
print()
for cam_dir in sorted((SESSION / "cameras").iterdir()):
    if not cam_dir.is_dir():
        continue
    frames_path = cam_dir / "frames.jsonl"
    if not frames_path.exists():
        continue
    frames = [json.loads(l) for l in open(frames_path)]
    if len(frames) < 2:
        continue
    ts = np.array([f["host_timestamp_ns"] for f in frames])
    intervals_ms = np.diff(ts) / 1e6
    print(f"{cam_dir.name}:")
    print(f"  n_frames: {len(frames)}")
    print(f"  duration: {(ts[-1] - ts[0]) / 1e9:.1f}s")
    print(f"  measured fps: {(len(frames) - 1) / ((ts[-1] - ts[0]) / 1e9):.2f}")
    print(f"  interval median: {np.median(intervals_ms):.1f}ms (expected ~33.3)")
    print(f"  interval p95:    {np.percentile(intervals_ms, 95):.1f}ms")
    print(f"  interval max:    {max(intervals_ms):.1f}ms")
    print(f"  gaps > 100ms:    {int((intervals_ms > 100).sum())}")

# 3. Event timestamps vs camera coverage
print()
events = [json.loads(l) for l in open(SESSION / "events.jsonl")]
print(f"Events: {len(events)}")
for e in events:
    ts = e["robot_timestamp_ns"]
    line = f"  {e['event']} @ {ts}"
    # For each camera, find frame closest to this event ts
    for cam_dir in sorted((SESSION / "cameras").iterdir()):
        if not cam_dir.is_dir():
            continue
        frames = [json.loads(l) for l in open(cam_dir / "frames.jsonl")]
        if not frames:
            continue
        frame_ts = np.array([f["host_timestamp_ns"] for f in frames])
        idx = int(np.argmin(np.abs(frame_ts - ts)))
        offset_ms = (frame_ts[idx] - ts) / 1e6
        line += f"  | {cam_dir.name}: frame {idx} ({offset_ms:+.0f}ms)"
    print(line)