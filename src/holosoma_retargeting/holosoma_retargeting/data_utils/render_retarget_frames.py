#!/usr/bin/env python3
"""Render a retargeted robot trajectory for visual inspection.

Draws the robot's kinematic tree from MuJoCo forward kinematics with matplotlib, so it
needs no GL context and works on a headless box whose GPU is busy. Overlays the source
human keypoints stored alongside the solution, which makes tracking error, ground contact
and limb interpenetration directly visible.

Three things make this useful for diagnosing a *bad* solve rather than just viewing a good
one:

* **Pinned joints are drawn in red.** A link whose parent joint is sitting within
  ``SAT_MARGIN`` of a range limit on that frame is highlighted, so a leg locked against its
  stop is obvious at a glance rather than inferred from a metric.
* **``--compare``** renders a second run beside the first with shared axes and synchronised
  frames, for before/after on the same clip.
* **``--gif``** writes an animation. A contact sheet cannot show that a pose is *frozen*,
  which is the signature of the degenerate solves.

    # contact sheet
    python render_retarget_frames.py --npz clip.npz --out /tmp/clip.png
    # animation, before vs after
    python render_retarget_frames.py --npz base/clip.npz --compare fixed/clip.npz \
        --gif --out /tmp/clip.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manim  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# Fraction of a joint's range within which it counts as pinned, matching
# analyze_retarget_quality.SAT_MARGIN_FRAC so the picture and the numbers agree.
SAT_MARGIN_FRAC = 0.02

OK_COLOR = "#1f77b4"
PIN_COLOR = "#d62728"
HUMAN_COLOR = "#2ca02c"


def body_tree(model) -> list[tuple[int, int]]:
    """Parent/child body index pairs, skipping the world body."""
    return [(model.body_parentid[b], b) for b in range(1, model.nbody) if model.body_parentid[b] != 0]


def joint_of_body(model) -> dict[int, int]:
    """Map body id -> its first limited joint id, for pinned-joint highlighting."""
    out: dict[int, int] = {}
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        out.setdefault(int(model.jnt_bodyid[j]), j)
    return out


class Traj:
    """One retargeting result: FK'd link positions plus per-frame pinned-joint flags."""

    def __init__(self, path: Path, model, data, label: str):
        d = np.load(path, allow_pickle=True)
        self.label = label
        self.q = d["qpos"]
        self.human = d["human_joints"] if "human_joints" in d.files else None
        self.n = self.q.shape[0]

        b2j = joint_of_body(model)
        self.xyz = np.zeros((self.n, model.nbody, 3))
        self.pinned = np.zeros((self.n, model.nbody), dtype=bool)

        lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
        for i in range(self.n):
            data.qpos[:] = self.q[i, : model.nq]
            mujoco.mj_forward(model, data)
            self.xyz[i] = data.xpos
            for body_id, j in b2j.items():
                v = data.qpos[model.jnt_qposadr[j]]
                margin = SAT_MARGIN_FRAC * (hi[j] - lo[j])
                self.pinned[i, body_id] = bool(v <= lo[j] + margin or v >= hi[j] - margin)


def draw(ax, tr: Traj, frame: int, tree, a: int, b: int, lims, show_human: bool, center: bool = False) -> None:
    ax.clear()
    pts = tr.xyz[frame]
    # Follow the root horizontally. Without this a clip that travels several metres is
    # squashed to a few pixels by the shared equal-aspect axes and the pose -- the whole
    # point of the render -- becomes unreadable.
    off = np.zeros(3)
    if center:
        off = pts[1].copy()
        off[2] = 0.0  # keep the true ground plane; only follow horizontally
    pts = pts - off
    for par, ch in tree:
        color = PIN_COLOR if tr.pinned[frame, ch] else OK_COLOR
        lw = 2.2 if tr.pinned[frame, ch] else 1.3
        ax.plot([pts[par, a], pts[ch, a]], [pts[par, b], pts[ch, b]], "-", color=color, lw=lw)
    ax.scatter(pts[1:, a], pts[1:, b], s=3, color=OK_COLOR)
    if show_human and tr.human is not None and frame < tr.human.shape[0]:
        h = tr.human[frame] - off
        ax.scatter(h[:, a], h[:, b], s=14, color=HUMAN_COLOR, alpha=0.7, marker="x")
    ax.axhline(0.0, color="#888888", ls="--", lw=0.8)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    n_pin = int(tr.pinned[frame].sum())
    ax.set_title(f"{tr.label}  f{frame}" + (f"  [{n_pin} joints pinned]" if n_pin else ""),
                 fontsize=8, color=PIN_COLOR if n_pin else "black")


def shared_limits(trajs: list[Traj], a: int, b: int, pad: float = 0.25, center: bool = False):
    if center:
        pts = np.concatenate([(t.xyz[:, 1:, :] - t.xyz[:, 1:2, :] * np.array([1, 1, 0])).reshape(-1, 3) for t in trajs], axis=0)
    else:
        pts = np.concatenate([t.xyz[:, 1:, :].reshape(-1, 3) for t in trajs], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return ((lo[a] - pad, hi[a] + pad), (min(lo[b], 0) - 0.1, hi[b] + pad))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--compare", default=None, help="Second run of the SAME clip, drawn beside the first")
    p.add_argument("--labels", nargs=2, default=None, help="Titles for the two panels")
    p.add_argument("--robot", default="r1")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=6, help="Columns in the contact sheet")
    p.add_argument("--gif", action="store_true", help="Write an animation instead of a contact sheet")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=2, help="Frame stride for the animation")
    p.add_argument("--view", default="side", choices=["side", "front"],
                   help="side = X(forward) vs Z(up); front = Y(left) vs Z(up)")
    p.add_argument("--no-human", action="store_true")
    p.add_argument("--center", action="store_true", help="Follow the root horizontally so travelling clips stay readable")
    args = p.parse_args()

    pkg_root = Path(__file__).resolve().parent.parent
    xml = pkg_root / "models" / args.robot / f"{args.robot}_26dof.xml"
    if not xml.exists():
        cands = sorted((pkg_root / "models" / args.robot).glob("*.xml"))
        if not cands:
            raise SystemExit(f"no MuJoCo xml for robot {args.robot}")
        xml = cands[0]
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    labels = args.labels or (["before", "after"] if args.compare else [Path(args.npz).stem])
    trajs = [Traj(Path(args.npz), model, data, labels[0])]
    if args.compare:
        trajs.append(Traj(Path(args.compare), model, data, labels[1]))

    tree = body_tree(model)
    a, b = (0, 2) if args.view == "side" else (1, 2)
    axis_label = "X (forward)" if args.view == "side" else "Y (left)"
    lims = shared_limits(trajs, a, b, center=args.center)
    n = min(t.n for t in trajs)
    show_human = not args.no_human

    if args.gif:
        fig, axes = plt.subplots(1, len(trajs), figsize=(4.2 * len(trajs), 4.6), squeeze=False)
        idx = list(range(0, n, max(1, args.stride)))

        def update(fi):
            for k, tr in enumerate(trajs):
                draw(axes[0][k], tr, fi, tree, a, b, lims, show_human, args.center)
                axes[0][k].set_xlabel(axis_label, fontsize=7)
            return []

        anim = manim.FuncAnimation(fig, update, frames=idx, interval=1000 / args.fps, blit=False)
        anim.save(args.out, writer=manim.PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"wrote {args.out}  ({len(idx)} frames, red = joint at its limit)")
        return

    cols = np.linspace(0, n - 1, args.frames).astype(int)
    fig, axes = plt.subplots(len(trajs), len(cols), figsize=(2.7 * len(cols), 3.1 * len(trajs)), squeeze=False)
    for r, tr in enumerate(trajs):
        for c, f in enumerate(cols):
            draw(axes[r][c], tr, f, tree, a, b, lims, show_human, args.center)
            if c == 0:
                axes[r][c].set_ylabel(f"Z (up) vs {axis_label}", fontsize=7)
    fig.suptitle(
        f"{Path(args.npz).stem}   blue = R1 links, RED = joint at its limit, green x = source human",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=95)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
