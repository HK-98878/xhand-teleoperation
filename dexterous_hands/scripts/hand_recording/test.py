import json, numpy as np
from pathlib import Path

SESSION = Path("exports/session_20260514_185215")  # edit me

records = [json.loads(l) for l in open(SESSION / "haptic_vector.jsonl")]
print(f"Loaded {len(records)} samples")

# Filter to records with actual data, gather per-axis values across all taxels.
xs, ys, zs = [], [], []
for r in records:
    rf = r.get("raw_force")
    if rf is None: continue
    a = np.array(rf, dtype=np.float32).reshape(-1, 3)  # (5*120, 3)
    # Skip "zero-everywhere" samples — they dominate idle periods and skew stats.
    if np.abs(a).max() < 0.5: continue
    xs.append(a[:, 0])
    ys.append(a[:, 1])
    zs.append(a[:, 2])

xs = np.concatenate(xs) if xs else np.array([])
ys = np.concatenate(ys) if ys else np.array([])
zs = np.concatenate(zs) if zs else np.array([])

def stats(name, vals):
    if len(vals) == 0:
        print(f"  {name}: no data"); return
    pcts = np.percentile(vals, [1, 50, 90, 95, 99, 99.9])
    print(f"  {name}: n={len(vals)}, "
          f"min={vals.min():+7.1f}, max={vals.max():+7.1f}, "
          f"|.50|={pcts[1]:+6.1f}, |.90|={pcts[2]:+6.1f}, "
          f"|.95|={pcts[3]:+6.1f}, |.99|={pcts[4]:+6.1f}, |.999|={pcts[5]:+6.1f}")

print("\nForce distributions (after filtering out idle samples):")
stats("x (shear)", xs)
stats("y (shear)", ys)
stats("z (normal)", zs)

# Also report how often each axis would saturate at various bound choices
print("\nSaturation rates per bound choice:")
for axis, vals in [("x", xs), ("y", ys), ("z (positive)", zs[zs > 0] if len(zs) else zs), ("z (negative)", -zs[zs < 0] if len(zs) else zs)]:
    if len(vals) == 0: continue
    for bound in [20, 30, 50, 70, 100]:
        sat = (np.abs(vals) > bound).mean() * 100
        print(f"  {axis} bound=±{bound}: {sat:.2f}% saturated")
    print()