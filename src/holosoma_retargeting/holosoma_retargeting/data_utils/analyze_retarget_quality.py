#!/usr/bin/env python3
"""Measure retargeting quality directly from saved qpos trajectories.

Complements ``evaluation/eval_retargeting.py``: this reads a directory of
``<clip>.npz`` retargeting outputs and reports the four artifacts that matter for a
lower-DoF humanoid, per clip and in aggregate. It is deliberately independent of the
input motion format so it can be pointed at any run, and it is fast enough to sweep.

    foot_slide_*        Horizontal toe speed while that toe is in contact. The direct
                        measure of foot sliding, and the thing
                        --retargeter.foot-sticking-tolerance trades against tracking.
    ground_pen_*        How far the lowest contact sphere goes below z=0.
    joint_sat_*         Fraction of (frame, joint) samples within a small margin of a
                        joint limit. High saturation means the motion is asking for more
                        range than the embodiment has -- the signature of an infeasible
                        clip rather than a bad solve.
    self_collide_*      Closest distance between non-adjacent robot bodies, via MuJoCo
                        collision detection.

Usage:
    python analyze_retarget_quality.py --results-dir /path/to/rt_out --robot r1
    python analyze_retarget_quality.py --results-dir /path/to/rt_out --robot r1 --json out.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# A toe is treated as in contact when it is within this height of the ground.
CONTACT_HEIGHT_M = 0.03
# Joint-limit saturation margin, as a fraction of each joint's total range.
SAT_MARGIN_FRAC = 0.02
# Toe speed above this during contact counts as sliding (m/s), matching
# RetargetingEvaluator.sliding_threshold.
SLIDING_THRESHOLD_MPS = 0.01


@dataclass
class ClipMetrics:
    clip_id: str
    frames: int
    cost: float
    foot_slide_frac: float
    foot_slide_mean_mps: float
    foot_slide_p95_mps: float
    ground_pen_max_m: float
    ground_pen_frac: float
    joint_sat_frac: float
    joint_sat_worst: str
    self_collide_min_m: float
    self_collide_pair: str
    base_z_min: float
    base_z_max: float


def _link_ids(model, names: list[str]) -> list[int]:
    out = []
    for n in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid < 0:
            raise ValueError(f"body {n!r} not found in model")
        out.append(bid)
    return out


def analyze_clip(path: Path, model, data, toe_bodies: list[str], contact_bodies: list[str], fps_default: float) -> ClipMetrics | None:
    try:
        d = np.load(path, allow_pickle=True)
        q = d["qpos"]
    except (OSError, KeyError, ValueError):
        return None
    if q.ndim != 2 or q.shape[0] < 2:
        return None

    fps = float(d["fps"]) if "fps" in d.files else fps_default
    dt = 1.0 / fps
    cost = float(d["cost"]) if "cost" in d.files else float("nan")

    toe_ids = _link_ids(model, toe_bodies)
    contact_ids = _link_ids(model, contact_bodies)

    n = q.shape[0]
    toe_xyz = np.zeros((n, len(toe_ids), 3))
    contact_z = np.zeros((n, len(contact_ids)))
    qpos_dof = np.zeros((n, model.nq))
    min_pair_dist = np.inf
    worst_pair = "-"

    for i in range(n):
        data.qpos[:] = q[i, : model.nq]
        mujoco.mj_forward(model, data)
        qpos_dof[i] = data.qpos
        for k, bid in enumerate(toe_ids):
            toe_xyz[i, k] = data.xpos[bid]
        for k, bid in enumerate(contact_ids):
            contact_z[i, k] = data.xpos[bid][2]
        # ncon reflects actual detected contacts; a negative dist is interpenetration.
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if b1 == b2:
                continue
            # Body 0 is `world`. Foot-vs-ground contacts are the *intended* behaviour, and
            # counting them here made every clip look like it had ~1 mm of self-collision
            # (the solver's contact tolerance) and buried the real interpenetrations.
            if b1 == 0 or b2 == 0:
                continue
            # Ignore geoms whose bodies are directly connected by a joint.
            if model.body_parentid[b1] == b2 or model.body_parentid[b2] == b1:
                continue
            dist = float(con.dist)
            if dist < min_pair_dist:
                min_pair_dist = dist
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1)
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2)
                worst_pair = f"{n1}<->{n2}"

    # --- foot sliding -------------------------------------------------------
    toe_xy_speed = np.linalg.norm(np.diff(toe_xyz[:, :, :2], axis=0), axis=-1) / dt
    toe_xy_speed = np.vstack([np.zeros((1, toe_xy_speed.shape[1])), toe_xy_speed])
    in_contact = toe_xyz[:, :, 2] < CONTACT_HEIGHT_M
    sliding = in_contact & (toe_xy_speed > SLIDING_THRESHOLD_MPS)
    n_contact = int(in_contact.sum())
    slide_speeds = toe_xy_speed[sliding]
    foot_slide_frac = float(sliding.sum() / n_contact) if n_contact else 0.0
    foot_slide_mean = float(slide_speeds.mean()) if slide_speeds.size else 0.0
    foot_slide_p95 = float(np.percentile(slide_speeds, 95)) if slide_speeds.size else 0.0

    # --- ground penetration -------------------------------------------------
    lowest = contact_z.min(axis=1)
    ground_pen_max = float(max(0.0, -lowest.min()))
    ground_pen_frac = float(np.mean(lowest < -0.005))

    # --- joint-limit saturation --------------------------------------------
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    limited = model.jnt_limited.astype(bool)
    sat_counts = np.zeros(model.njnt)
    total = 0
    for j in range(model.njnt):
        if not limited[j]:
            continue
        adr = model.jnt_qposadr[j]
        rng = hi[j] - lo[j]
        if rng <= 0:
            continue
        margin = SAT_MARGIN_FRAC * rng
        vals = qpos_dof[:, adr]
        sat_counts[j] = float(np.sum((vals <= lo[j] + margin) | (vals >= hi[j] - margin)))
        total += n
    joint_sat_frac = float(sat_counts.sum() / total) if total else 0.0
    worst_j = int(np.argmax(sat_counts))
    worst_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, worst_j) or "-"
    if sat_counts[worst_j] == 0:
        worst_name = "-"

    return ClipMetrics(
        clip_id=path.stem,
        frames=n,
        cost=cost,
        foot_slide_frac=foot_slide_frac,
        foot_slide_mean_mps=foot_slide_mean,
        foot_slide_p95_mps=foot_slide_p95,
        ground_pen_max_m=ground_pen_max,
        ground_pen_frac=ground_pen_frac,
        joint_sat_frac=joint_sat_frac,
        joint_sat_worst=worst_name,
        self_collide_min_m=float(min_pair_dist) if np.isfinite(min_pair_dist) else float("nan"),
        self_collide_pair=worst_pair,
        base_z_min=float(q[:, 2].min()),
        base_z_max=float(q[:, 2].max()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--robot", default="r1")
    p.add_argument("--fps", type=float, default=30.0, help="Assumed fps if not stored in the npz")
    p.add_argument("--json", default=None, help="Write per-clip metrics here")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="Only print the aggregate summary")
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

    from holosoma_retargeting.config_types.robot import RobotConfig

    rc = RobotConfig(robot_type=args.robot)
    contact_bodies = list(rc.FOOT_STICKING_LINKS)
    toe_bodies = [b for b in contact_bodies if b.endswith("sphere_5_link")] or contact_bodies[:2]

    files = sorted(Path(args.results_dir).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz under {args.results_dir}")

    rows: list[ClipMetrics] = []
    for f in files:
        m = analyze_clip(f, model, data, toe_bodies, contact_bodies, args.fps)
        if m is None:
            print(f"  UNREADABLE {f.name}")
            continue
        rows.append(m)
        if not args.quiet:
            print(
                f"{m.clip_id[:44]:44s} f={m.frames:5d} cost={m.cost:7.4f} "
                f"slide={m.foot_slide_frac:5.1%}/{m.foot_slide_p95_mps:5.3f}mps "
                f"pen={m.ground_pen_max_m:6.4f}m sat={m.joint_sat_frac:5.1%} ({m.joint_sat_worst})"
            )

    if not rows:
        raise SystemExit("nothing analyzable")

    def agg(attr: str) -> tuple[float, float]:
        v = np.array([getattr(r, attr) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(np.percentile(v, 95))) if v.size else (float("nan"), float("nan"))

    print(f"\n=== aggregate over {len(rows)} clip(s) ===")
    for attr in (
        "cost",
        "foot_slide_frac",
        "foot_slide_mean_mps",
        "foot_slide_p95_mps",
        "ground_pen_max_m",
        "ground_pen_frac",
        "joint_sat_frac",
        "self_collide_min_m",
    ):
        mean, p95 = agg(attr)
        print(f"  {attr:22s} mean={mean:9.5f}  p95={p95:9.5f}")

    sat = sorted(rows, key=lambda r: -r.joint_sat_frac)[:5]
    print("\n  most joint-limit-saturated clips:")
    for r in sat:
        print(f"    {r.joint_sat_frac:6.1%}  {r.joint_sat_worst:28s} {r.clip_id}")

    from collections import Counter

    pair_counts = Counter(r.self_collide_pair for r in rows if r.self_collide_pair != "-")
    if pair_counts:
        print("\n  most frequent self-penetrating body pairs (worst pair per clip):")
        for pair, count in pair_counts.most_common(8):
            depths = [r.self_collide_min_m for r in rows if r.self_collide_pair == pair]
            print(f"    {count:4d} clip(s)  deepest {min(depths):+.4f} m  {pair}")

    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ for r in rows], indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
