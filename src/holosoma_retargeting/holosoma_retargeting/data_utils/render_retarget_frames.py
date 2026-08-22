#!/usr/bin/env python3
"""Render a retargeted robot trajectory as a stick-figure contact sheet.

Draws the robot's kinematic tree from MuJoCo forward kinematics using matplotlib, so it
needs no GL context and works on a headless box whose GPU is busy. Overlays the source
human keypoints stored alongside the solution, which makes tracking error, ground contact
and limb interpenetration directly visible.

    python render_retarget_frames.py --npz /path/clip.npz --robot r1 --out /tmp/clip.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def body_tree(model) -> list[tuple[int, int]]:
    """Parent/child body index pairs, skipping the world body."""
    return [(model.body_parentid[b], b) for b in range(1, model.nbody) if model.body_parentid[b] != 0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--robot", default="r1")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--highlight", nargs="*", default=["wrist_roll", "hip_roll"],
                   help="Substrings of body names to mark, for spotting interpenetration")
    args = p.parse_args()

    pkg_root = Path(__file__).resolve().parent.parent
    xml = pkg_root / "models" / args.robot / f"{args.robot}_26dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    d = np.load(args.npz, allow_pickle=True)
    q = d["qpos"]
    human = d["human_joints"] if "human_joints" in d.files else None

    idx = np.linspace(0, q.shape[0] - 1, args.frames).astype(int)
    tree = body_tree(model)
    hl_ids = [
        b for b in range(model.nbody)
        if any(h in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for h in args.highlight)
    ]

    # Precompute all frames so the axes can share limits.
    xyz_all, hum_all = [], []
    for f in idx:
        data.qpos[:] = q[f, : model.nq]
        mujoco.mj_forward(model, data)
        xyz_all.append(data.xpos.copy())
        if human is not None:
            hum_all.append(human[f])

    pts = np.concatenate([x[1:] for x in xyz_all], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.25

    views = [("X (forward)", "Z (up)", 0, 2), ("Y (left)", "Z (up)", 1, 2)]
    fig, axes = plt.subplots(len(views), len(idx), figsize=(2.9 * len(idx), 3.0 * len(views)), squeeze=False)

    for r, (xl, yl, a, b) in enumerate(views):
        for c, f in enumerate(idx):
            ax = axes[r][c]
            xyz = xyz_all[c]
            for par, ch in tree:
                ax.plot([xyz[par, a], xyz[ch, a]], [xyz[par, b], xyz[ch, b]], "-", color="#1f77b4", lw=1.4)
            ax.scatter(xyz[1:, a], xyz[1:, b], s=3, color="#1f77b4")
            if hl_ids:
                ax.scatter(xyz[hl_ids, a], xyz[hl_ids, b], s=34, facecolors="none", edgecolors="#d62728", lw=1.3)
            if hum_all:
                ax.scatter(hum_all[c][:, a], hum_all[c][:, b], s=9, color="#2ca02c", alpha=0.65, marker="x")
            ax.axhline(0.0, color="#888888", ls="--", lw=0.8)
            ax.set_xlim(lo[a] - pad, hi[a] + pad)
            ax.set_ylim(min(lo[b], 0) - 0.1, hi[b] + pad)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"frame {f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{yl} vs {xl}", fontsize=7)

    fig.suptitle(
        f"{Path(args.npz).stem}   blue = R1 links, red rings = wrist/hip, green x = source human keypoints",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=95)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
