#!/usr/bin/env python3
"""Standalone correctness checks for the SOMA BVH loader.

A silently-wrong motion loader produces confident-looking garbage: the retargeter will
happily converge on a rotated, mirrored or unit-scaled skeleton and the failure only
surfaces much later as inexplicably bad policies. These checks are designed to fail loudly
on each specific way the parse can go wrong.

    python validate_soma_loader.py --bvh-dir /tmp/soma_samples
    python validate_soma_loader.py --bvh-dir /data/soma_uniform/bvh/230315 --limit 20
    python validate_soma_loader.py --bvh-dir /tmp/soma_samples --render-dir /tmp/soma_render

What each check catches:

    finite              NaN/Inf from malformed channel rows.
    bone_length         Wrong channel order, wrong per-joint channel slicing, or a
                        misparsed hierarchy. Any *rotation* preserves bone length, so this
                        isolates parsing errors from convention errors.
    stature             Unit errors (a 100x cm/m bug) and a broken stature chain.
    ground_contact      Wrong up-axis, or a vertical offset.
    upright             Wrong Euler composition -- the single check that separates
                        intrinsic from extrinsic, which bone_length cannot.
    root_continuity     Frame-row misalignment or a dropped/duplicated channel.
    chirality           Left/right mirroring, i.e. a handedness flip in the world matrix.
    facing              A yaw error in the world matrix (walk-forward clips only).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from holosoma_retargeting.data_utils.soma_bvh import (
    estimate_height,
    forward_kinematics,
    parse_bvh,
    to_world_frame,
)

# Bone-length coefficient of variation. Rigid bones should be exactly constant; this
# tolerance only absorbs float32/FK round-off.
BONE_LENGTH_CV_TOL = 1e-3
# Stature band for a human adult, in meters.
MIN_HEIGHT_M, MAX_HEIGHT_M = 1.5, 2.0
# How far below the ground plane any joint may go. SOMA clips include ground-contact and
# lying motions where a shoulder or hip can compress slightly below the nominal floor.
MAX_GROUND_PENETRATION_M = 0.10
# The resting floor gap: the 5th percentile of per-frame minimum height should sit close
# to zero for any clip with meaningful ground contact.
RESTING_FLOOR_TOL_M = 0.08
# A root translation jump this large between adjacent 120 fps frames (=> 12 m/s) is a
# parse discontinuity, not human motion.
MAX_ROOT_STEP_M = 0.10


@dataclass
class Config:
    bvh_dir: str
    """Directory of .bvh files to validate (searched recursively)."""

    limit: int = 0
    """Validate at most this many files (0 = all)."""

    render_dir: str | None = None
    """If set, write a PNG contact sheet per clip here."""

    euler_convention: str = "extrinsic"
    """Rotation composition to test; see soma_bvh._local_transforms."""

    verbose: bool = False
    """Print per-check detail for passing clips too."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _rigid_bones(skel) -> list[tuple[int, int]]:
    """Parent/child pairs whose separation is a rigid bone.

    Excludes any child carrying position channels: in SOMA, ``Hips`` has 6 channels, so
    the Root->Hips separation legitimately varies as the character translates and is not
    a bone at all.
    """
    pairs = []
    for j, parent in enumerate(skel.parents):
        if parent < 0:
            continue
        if any("position" in c for c in skel.channels[j]):
            continue
        pairs.append((parent, j))
    return pairs


def validate(path: Path, euler_convention: str) -> tuple[list[Check], np.ndarray, object]:
    skel = parse_bvh(path)
    positions = to_world_frame(forward_kinematics(skel, euler_convention=euler_convention))
    names = skel.names
    checks: list[Check] = []

    def idx(name: str) -> int | None:
        return names.index(name) if name in names else None

    # --- finite ---------------------------------------------------------------
    n_bad = int((~np.isfinite(positions)).sum())
    checks.append(Check("finite", n_bad == 0, f"{n_bad} non-finite value(s)"))

    # --- bone_length ----------------------------------------------------------
    worst_cv, worst_name = 0.0, "-"
    for parent, child in _rigid_bones(skel):
        lengths = np.linalg.norm(positions[:, child, :] - positions[:, parent, :], axis=-1)
        mean = lengths.mean()
        if mean < 1e-9:
            continue
        cv = float(lengths.std() / mean)
        if cv > worst_cv:
            worst_cv, worst_name = cv, names[child]
    checks.append(
        Check(
            "bone_length",
            worst_cv < BONE_LENGTH_CV_TOL,
            f"worst CV {worst_cv:.2e} on {worst_name} (tol {BONE_LENGTH_CV_TOL:.0e})",
        )
    )

    # --- stature --------------------------------------------------------------
    height = estimate_height(skel)
    checks.append(
        Check(
            "stature",
            MIN_HEIGHT_M <= height <= MAX_HEIGHT_M,
            f"{height:.3f} m (band {MIN_HEIGHT_M}-{MAX_HEIGHT_M})",
        )
    )

    # --- ground_contact -------------------------------------------------------
    per_frame_min = positions[..., 2].min(axis=1)
    deepest = float(per_frame_min.min())
    resting = float(np.percentile(per_frame_min, 5))
    checks.append(
        Check(
            "ground_contact",
            deepest > -MAX_GROUND_PENETRATION_M and abs(resting) < RESTING_FLOOR_TOL_M,
            f"deepest {deepest:+.4f} m, resting p5 {resting:+.4f} m",
        )
    )

    # --- upright --------------------------------------------------------------
    # Separates the Euler conventions. A wrong composition collapses the torso so that
    # the Hips->Head axis has no preferred vertical direction.
    hips, head = idx("Hips"), idx("Head")
    if hips is not None and head is not None:
        torso = positions[:, head, :] - positions[:, hips, :]
        torso = torso / np.maximum(np.linalg.norm(torso, axis=-1, keepdims=True), 1e-9)
        mean_cos = float(torso[:, 2].mean())
        above = float(np.mean(positions[:, head, 2] > positions[:, hips, 2]))
        # Deliberately loose: crawling and lying clips legitimately drop the mean cosine.
        checks.append(
            Check(
                "upright",
                mean_cos > 0.3 and above > 0.5,
                f"mean cos(torso,+Z) {mean_cos:+.3f}, head-above-hips {above:.1%}",
            )
        )

    # --- root_continuity ------------------------------------------------------
    root = idx("Hips")
    if root is not None:
        steps = np.linalg.norm(np.diff(positions[:, root, :], axis=0), axis=-1)
        # Scale the per-frame tolerance to the clip's own frame rate.
        tol = MAX_ROOT_STEP_M * (120.0 / max(skel.fps, 1e-6))
        n_spikes = int((steps > tol).sum())
        worst = float(steps.max()) if steps.size else 0.0
        checks.append(
            Check(
                "root_continuity",
                n_spikes == 0,
                f"{n_spikes} step(s) > {tol:.3f} m, worst {worst:.4f} m "
                f"({worst * skel.fps:.2f} m/s)",
            )
        )

    # --- chirality ------------------------------------------------------------
    # Pose-independent mirror test, using only the skeleton -- no assumption about which
    # world axis is which. For an unmirrored human:
    #     (LeftShoulder - RightShoulder) x (Head - Hips)  ==  left x up  ==  forward
    # and the foot independently points forward (toe ahead of ankle). A left/right swap is
    # a reflection, which flips the cross product but not the foot vector, so the dot
    # product goes negative.
    ls, rs = idx("LeftShoulder"), idx("RightShoulder")
    lf, lt = idx("LeftFoot"), idx("LeftToeBase")
    if None not in (ls, rs, hips, head, lf, lt):
        left = positions[:, ls, :] - positions[:, rs, :]
        up = positions[:, head, :] - positions[:, hips, :]
        fwd_body = np.cross(left, up)
        fwd_foot = positions[:, lt, :] - positions[:, lf, :]
        dots = np.einsum("ij,ij->i", fwd_body, fwd_foot)
        agree = float(np.mean(dots > 0))
        checks.append(
            Check("chirality", agree > 0.75, f"body-forward agrees with foot-forward in {agree:.1%} of frames")
        )

    # Eye separation is an unambiguous left/right witness when the face joints are present
    # (BONES-SEED keeps them; they are dropped from the retargeting subset).
    le, re = idx("LeftEye"), idx("RightEye")
    if le is not None and re is not None:
        eye = positions[:, le, :] - positions[:, re, :]
        # +Y is world-left in our target frame.
        agree = float(np.mean(eye[:, 1] > 0))
        checks.append(Check("chirality_eyes", agree > 0.9, f"LeftEye is +Y of RightEye in {agree:.1%} of frames"))

    # --- facing ---------------------------------------------------------------
    # Only meaningful for clips that translate: the direction of travel should agree with
    # the direction the body faces. Catches a yaw error in the world matrix, which no
    # other check here sees.
    if root is not None and None not in (ls, rs, hips, head):
        disp = positions[-1, root, :2] - positions[0, root, :2]
        travel = float(np.linalg.norm(disp))
        if travel > 0.5:
            left = positions[:, ls, :] - positions[:, rs, :]
            up = positions[:, head, :] - positions[:, hips, :]
            fwd = np.cross(left, up)[:, :2].mean(axis=0)
            fwd /= max(np.linalg.norm(fwd), 1e-9)
            cos = float(np.dot(disp / travel, fwd))
            checks.append(
                Check("facing", cos > 0.5, f"travel {travel:.2f} m, cos(travel, body-forward) {cos:+.3f}")
            )
        else:
            checks.append(Check("facing", True, f"skipped, travel only {travel:.2f} m"))

    return checks, positions, skel


def render(path: Path, positions: np.ndarray, skel, out_dir: Path) -> Path:
    """Write a PNG contact sheet: 6 evenly-spaced frames in 3 orthogonal projections."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    bones = _rigid_bones(skel)
    n_show = 6
    frames = np.linspace(0, positions.shape[0] - 1, n_show).astype(int)

    views = [("X (forward)", "Z (up)", 0, 2), ("Y (left)", "Z (up)", 1, 2), ("X (forward)", "Y (left)", 0, 1)]
    fig, axes = plt.subplots(len(views), n_show, figsize=(3 * n_show, 3 * len(views)))

    lo = positions.reshape(-1, 3).min(axis=0)
    hi = positions.reshape(-1, 3).max(axis=0)

    for r, (xl, yl, a, b) in enumerate(views):
        for c, f in enumerate(frames):
            ax = axes[r, c]
            pts = positions[f]
            for parent, child in bones:
                ax.plot(
                    [pts[parent, a], pts[child, a]],
                    [pts[parent, b], pts[child, b]],
                    "-",
                    color="#1f77b4",
                    linewidth=1.0,
                )
            ax.scatter(pts[:, a], pts[:, b], s=2, color="#d62728")
            if b == 2:
                ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
            ax.set_xlim(lo[a] - 0.2, hi[a] + 0.2)
            ax.set_ylim(lo[b] - 0.2, hi[b] + 0.2)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"frame {f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{yl}\nvs {xl}", fontsize=7)

    fig.suptitle(f"{path.stem}  ({positions.shape[0]} frames @ {skel.fps:.1f} fps)", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def main(cfg: Config) -> None:
    bvh_dir = Path(cfg.bvh_dir)
    files = sorted(bvh_dir.rglob("*.bvh"))
    if cfg.limit:
        files = files[: cfg.limit]
    if not files:
        raise SystemExit(f"no .bvh files under {bvh_dir}")

    print(f"Validating {len(files)} clip(s), euler_convention={cfg.euler_convention}\n")
    n_failed_clips = 0
    failed_by_check: dict[str, int] = {}

    for path in files:
        try:
            checks, positions, skel = validate(path, cfg.euler_convention)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR ] {path.name}: {type(exc).__name__}: {exc}")
            n_failed_clips += 1
            continue

        failed = [c for c in checks if not c.passed]
        status = "PASS" if not failed else "FAIL"
        print(f"[{status:6s}] {path.name}  ({positions.shape[0]} frames, {positions.shape[1]} joints)")
        for c in checks:
            if not c.passed:
                failed_by_check[c.name] = failed_by_check.get(c.name, 0) + 1
            if not c.passed or cfg.verbose:
                print(f"           {'x' if not c.passed else 'v'} {c.name:18s} {c.detail}")
        if failed:
            n_failed_clips += 1

        if cfg.render_dir:
            out = render(path, positions, skel, Path(cfg.render_dir))
            print(f"           rendered -> {out}")

    print(f"\n{len(files) - n_failed_clips}/{len(files)} clip(s) passed all checks")
    if failed_by_check:
        print("failures by check:")
        for name, count in sorted(failed_by_check.items(), key=lambda kv: -kv[1]):
            print(f"  {name:18s} {count}")
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
