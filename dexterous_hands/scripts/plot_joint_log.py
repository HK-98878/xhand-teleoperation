#!/usr/bin/env python3
"""
plot_joint_log.py — plot joint state and target from a deploy_vla.py CSV log.

Shows one subplot per finger (index, middle, ring, pinky) with all joints of
that finger overlaid. Solid lines = state, dashed lines = target.

Usage:
    python plot_joint_log.py joint_log_20240517_120000.csv
    python plot_joint_log.py joint_log_20240517_120000.csv --save out.png
    python plot_joint_log.py joint_log_20240517_120000.csv --title "grasp task"
"""

import argparse
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

# Finger → (joint indices, joint short-names).
# Thumb (0-2) is excluded. Index has an extra abduction joint at index 3.
FINGERS = {
    "index":  ([3, 4, 5], ["bend", "joint1", "joint2"]),
    "middle": ([6, 7],    ["joint1", "joint2"]),
    "ring":   ([8, 9],    ["joint1", "joint2"]),
    "pinky":  ([10, 11],  ["joint1", "joint2"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-finger plot of joint state vs target from a deploy_vla.py CSV log."
    )
    parser.add_argument("log_file", help="CSV log file produced by deploy_vla.py")
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save figure to this path instead of opening an interactive window",
    )
    parser.add_argument("--title", default=None, help="Optional figure title override")
    args = parser.parse_args()

    df = pd.read_csv(args.log_file)
    if "timestamp" not in df.columns:
        sys.exit("[ERROR] CSV has no 'timestamp' column — is this a deploy_vla joint log?")

    t = df["timestamp"].values - df["timestamp"].values[0]

    fig, axes = plt.subplots(2, 2, figsize=(24, 14), squeeze=False)
    fig.suptitle(args.title or args.log_file, fontsize=10)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (finger, (joint_indices, joint_names)) in zip(axes.flat, FINGERS.items()):
        for i, (idx, name) in enumerate(zip(joint_indices, joint_names)):
            state_col = f"state_{idx}"
            target_col = f"target_{idx}"
            if state_col not in df.columns or target_col not in df.columns:
                sys.exit(f"[ERROR] missing columns for joint {idx} in {args.log_file}")
            color = colors[i % len(colors)]
            ax.plot(t, df[state_col].values,  color=color, linewidth=1.2, label=name)
            ax.plot(t, df[target_col].values, color=color, linewidth=1.2, linestyle="--")

        ax.set_title(finger, fontsize=10)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("position (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", title="joint", title_fontsize=7)
        ax.grid(True, linewidth=0.4)

    # Figure-level linestyle legend
    style_handles = [
        mlines.Line2D([], [], color="k", linewidth=1.2,                  label="state"),
        mlines.Line2D([], [], color="k", linewidth=1.2, linestyle="--",  label="target"),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[plot] saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
