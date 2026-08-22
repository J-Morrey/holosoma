#!/usr/bin/env python3
"""Compare our R1 retargets against BONES-SEED's shipped Unitree G1 reference.

SEED ships NVIDIA's own G1 retarget of every clip, produced with
``NVIDIA/soma-retargeter``. That gives an independent answer to the question a
self-consistent pipeline cannot answer alone: when a motion looks wrong on R1, is it our
loader or is it the embodiment gap?

The root trajectory is the discriminating signal. It is set almost entirely by the source
motion and the world-frame convention, and only weakly by which humanoid is being driven,
so a coordinate-convention error in the loader shows up here as a rotated, mirrored or
scaled path even though every internal consistency check passes. Shared joint angles are
a weaker but still useful secondary signal.

Reference CSV layout (36 columns):
    Frame, root_translateX/Y/Z (centimeters), root_rotateX/Y/Z (degrees),
    then 29 <joint>_dof columns in degrees, at the source 120 fps.

Usage:
    python compare_to_g1_reference.py --our-dir /path/rt_out --g1-dir /path/g1/csv \
        --manifest pilot_soma_r1/sample_manifest.csv --limit 300
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

G1_CSV_FPS = 120.0
CM_TO_M = 0.01
# qpos layout is MuJoCo free-joint convention: [x, y, z, qw, qx, qy, qz, *dof].
QPOS_BASE_XYZ = slice(0, 3)


def load_g1_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (root_xyz_m (T,3), dof_names, dof_rad (T,29))."""
    with path.open() as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    arr = np.array([[float(v) for v in r] for r in body], dtype=np.float64)
    xyz = arr[:, 1:4] * CM_TO_M
    dof_cols = [i for i, h in enumerate(header) if h.endswith("_dof")]
    dof_names = [header[i].removesuffix("_dof") for i in dof_cols]
    return xyz, dof_names, np.deg2rad(arr[:, dof_cols])


def procrustes_2d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Best-fit yaw (degrees) and RMS residual (m) aligning path b onto a in the XY plane.

    Both paths are mean-centered first, so this measures shape agreement independent of
    where each retarget chose to place the character. A large yaw means the two pipelines
    disagree about which way is forward -- exactly the error a det=+1-vs-mirror or
    90-degree-yaw world-matrix mistake produces.
    """
    a = a[:, :2] - a[:, :2].mean(axis=0)
    b = b[:, :2] - b[:, :2].mean(axis=0)
    # Optimal rotation from the 2x2 cross-covariance.
    h = b.T @ a
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:  # keep it a rotation, not a reflection
        vt[-1, :] *= -1
        r = vt.T @ u.T
    yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    resid = float(np.sqrt(np.mean(np.sum((a - (b @ r.T)) ** 2, axis=1))))
    return yaw, resid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--our-dir", required=True)
    p.add_argument("--g1-dir", required=True)
    p.add_argument("--our-fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    g1_index = {q.stem: q for q in Path(args.g1_dir).rglob("*.csv")}
    ours = sorted(Path(args.our_dir).glob("*.npz"))
    if args.limit:
        ours = ours[: args.limit]
    if not ours:
        raise SystemExit(f"no .npz under {args.our_dir}")

    stride = int(round(G1_CSV_FPS / args.our_fps))
    recs: list[dict] = []

    for f in ours:
        g1_path = g1_index.get(f.stem)
        if g1_path is None:
            continue
        try:
            q = np.load(f, allow_pickle=True)["qpos"]
            g1_xyz, g1_names, g1_dof = load_g1_csv(g1_path)
        except (OSError, KeyError, ValueError, IndexError):
            continue

        g1_xyz_ds = g1_xyz[::stride]
        n = min(len(q), len(g1_xyz_ds))
        if n < 10:
            continue
        ours_xyz = q[:n, QPOS_BASE_XYZ]
        ref_xyz = g1_xyz_ds[:n]

        yaw, resid = procrustes_2d(ref_xyz, ours_xyz)
        # Raw (unaligned) horizontal displacement agreement.
        our_disp = float(np.linalg.norm(ours_xyz[-1, :2] - ours_xyz[0, :2]))
        ref_disp = float(np.linalg.norm(ref_xyz[-1, :2] - ref_xyz[0, :2]))
        # Cosine between the two net travel directions, when there is travel to speak of.
        cos = float("nan")
        if our_disp > 0.3 and ref_disp > 0.3:
            u = (ours_xyz[-1, :2] - ours_xyz[0, :2]) / our_disp
            v = (ref_xyz[-1, :2] - ref_xyz[0, :2]) / ref_disp
            cos = float(np.dot(u, v))

        recs.append(
            {
                "clip_id": f.stem,
                "frames": int(n),
                "yaw_deg": yaw,
                "xy_shape_rms_m": resid,
                "our_disp_m": our_disp,
                "ref_disp_m": ref_disp,
                "disp_ratio": (our_disp / ref_disp) if ref_disp > 1e-6 else float("nan"),
                "travel_cos": cos,
                "our_base_z_mean": float(ours_xyz[:, 2].mean()),
                "ref_base_z_mean": float(ref_xyz[:, 2].mean()),
            }
        )
        if args.verbose:
            r = recs[-1]
            print(
                f"{r['clip_id'][:42]:42s} yaw={r['yaw_deg']:+7.2f} rms={r['xy_shape_rms_m']:.3f} "
                f"disp {r['our_disp_m']:5.2f}/{r['ref_disp_m']:5.2f} cos={r['travel_cos']:+.3f}"
            )

    if not recs:
        raise SystemExit("no clips matched between our outputs and the G1 reference")

    def col(k):
        v = np.array([r[k] for r in recs], dtype=float)
        return v[np.isfinite(v)]

    print(f"\n=== compared {len(recs)} clip(s) against the G1 reference ===")
    yaw = col("yaw_deg")
    print(f"  best-fit yaw (deg)   median={np.median(yaw):+7.2f}  |yaw|<10deg in {np.mean(np.abs(yaw) < 10):.1%}")
    print(f"  xy shape RMS (m)     median={np.median(col('xy_shape_rms_m')):.4f}  p95={np.percentile(col('xy_shape_rms_m'), 95):.4f}")
    print(f"  displacement ratio   median={np.median(col('disp_ratio')):.4f}  p95={np.percentile(col('disp_ratio'), 95):.4f}")
    tc = col("travel_cos")
    if tc.size:
        print(f"  travel cosine        median={np.median(tc):+.4f}  >0.9 in {np.mean(tc > 0.9):.1%}  (n={tc.size} moving clips)")
    print(f"  base z mean (m)      ours={np.mean(col('our_base_z_mean')):.4f}  G1 ref={np.mean(col('ref_base_z_mean')):.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps(recs, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
