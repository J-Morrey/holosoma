#!/usr/bin/env python3
"""Convert NVIDIA SOMA-format BVH motion into the retargeting pipeline's .npz contract.

Writes one ``<clip_id>.npz`` per input BVH containing ``global_joint_positions``
``(T, 22, 3)`` float32 in meters (Z-up, X-forward) and a scalar ``height`` in meters,
which is what ``examples/robot_retarget.py``'s generic ``.npz`` fallback expects.

Use ``soma_uniform/bvh/`` from BONES-SEED rather than ``soma_proportional/``: the
retargeter applies a single per-sequence scale factor, so a unified-proportion skeleton
avoids per-performer proportion variance across SEED's 500+ performers.

Examples:
    # Convert a directory tree of BVH files
    python prep_soma_bvh_for_rt.py --input-dir /data/soma_uniform/bvh --output-dir /data/soma_npz

    # Convert only the clips named in a sample manifest
    python prep_soma_bvh_for_rt.py --input-dir /data/soma_uniform/bvh \
        --output-dir /data/soma_npz --clip-list sample_clip_ids.txt
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from holosoma_retargeting.data_utils.soma_bvh import load_soma_bvh

# A SOMA performer is a human adult; anything outside this band means the stature
# estimate or the unit scale is wrong.
MIN_PLAUSIBLE_HEIGHT_M = 1.4
MAX_PLAUSIBLE_HEIGHT_M = 2.1


@dataclass
class Config:
    """Configuration for SOMA BVH -> .npz conversion."""

    input_dir: str
    """Root of the SOMA BVH tree (searched recursively for *.bvh)."""

    output_dir: str
    """Destination directory for the .npz files."""

    clip_list: str | None = None
    """Optional file of clip IDs (one per line, no extension) to restrict the conversion."""

    target_fps: float = 30.0
    """Output frame rate. SOMA source is 120 fps; 30 matches the frame rate BONES-SEED's
    reference Unitree G1 CSVs were converted to, and divides 120 exactly so decimation
    introduces no interpolation error."""

    workers: int = 8
    """Parallel worker processes."""

    overwrite: bool = False
    """Reconvert clips whose .npz already exists."""

    flatten: bool = True
    """Write all .npz files directly into output_dir rather than mirroring the input tree.
    robot_retarget.py resolves a clip as ``data_path / f"{task_name}.npz"``, so a flat
    layout is what the retargeter expects."""


def convert_one(bvh_path: Path, out_path: Path, target_fps: float) -> dict:
    """Convert a single BVH file. Returns a result record; never raises."""
    record: dict = {"clip_id": out_path.stem, "source": str(bvh_path), "ok": False}
    try:
        data = load_soma_bvh(bvh_path, target_fps=target_fps)
        positions = data["global_joint_positions"]
        height = float(data["height"])

        if positions.shape[0] < 2:
            raise ValueError(f"only {positions.shape[0]} frame(s) after resampling")
        if not np.isfinite(positions).all():
            raise ValueError("non-finite values in global_joint_positions")
        if not (MIN_PLAUSIBLE_HEIGHT_M <= height <= MAX_PLAUSIBLE_HEIGHT_M):
            raise ValueError(f"implausible height {height:.3f} m -- check units/stature chain")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            global_joint_positions=positions,
            height=np.float32(height),
            fps=data["fps"],
            joint_names=data["joint_names"],
            source_fps=data["source_fps"],
        )
        record.update(
            ok=True,
            frames=int(positions.shape[0]),
            joints=int(positions.shape[1]),
            height=height,
            min_z=float(positions[..., 2].min()),
            duration_s=float(positions.shape[0] / target_fps),
        )
    except Exception as exc:  # noqa: BLE001 - a bad clip must not abort a 14k-clip run
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
    return record


def _resolve_inputs(cfg: Config) -> list[Path]:
    input_dir = Path(cfg.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input-dir does not exist: {input_dir}")

    bvh_files = sorted(input_dir.rglob("*.bvh"))
    if cfg.clip_list:
        wanted = {
            line.strip()
            for line in Path(cfg.clip_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        # Accept either bare clip IDs or paths relative to input_dir.
        wanted = {Path(w).stem for w in wanted}
        bvh_files = [p for p in bvh_files if p.stem in wanted]
        found = {p.stem for p in bvh_files}
        if missing := wanted - found:
            print(f"WARNING: {len(missing)} clip ID(s) in {cfg.clip_list} not found under {input_dir}")
            for m in sorted(missing)[:10]:
                print(f"  missing: {m}")
    return bvh_files


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bvh_files = _resolve_inputs(cfg)
    if not bvh_files:
        raise SystemExit("no BVH files matched")

    def out_for(p: Path) -> Path:
        if cfg.flatten:
            return output_dir / f"{p.stem}.npz"
        return output_dir / p.relative_to(cfg.input_dir).with_suffix(".npz")

    jobs = [(p, out_for(p)) for p in bvh_files]
    if not cfg.overwrite:
        skipped = sum(1 for _, o in jobs if o.exists())
        jobs = [(p, o) for p, o in jobs if not o.exists()]
        if skipped:
            print(f"Skipping {skipped} already-converted clip(s); pass --overwrite to redo.")

    print(f"Converting {len(jobs)} clip(s) at {cfg.target_fps} fps with {cfg.workers} worker(s)")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(convert_one, p, o, cfg.target_fps): p for p, o in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)
            if not rec["ok"]:
                print(f"  FAIL {rec['clip_id']}: {rec['error']}")
            if i % 500 == 0 or i == len(jobs):
                ok = sum(r["ok"] for r in results)
                print(f"  [{i}/{len(jobs)}] ok={ok} fail={len(results) - ok}", flush=True)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    manifest = output_dir / "conversion_manifest.jsonl"
    with manifest.open("w") as fh:
        for rec in sorted(results, key=lambda r: r["clip_id"]):
            fh.write(json.dumps(rec) + "\n")

    print(f"\nConverted {len(ok)}/{len(results)}  ({len(failed)} failed)")
    if ok:
        heights = np.array([r["height"] for r in ok])
        durations = np.array([r["duration_s"] for r in ok])
        min_zs = np.array([r["min_z"] for r in ok])
        print(f"  height   min/mean/max: {heights.min():.3f} / {heights.mean():.3f} / {heights.max():.3f} m")
        print(f"  duration min/mean/max: {durations.min():.2f} / {durations.mean():.2f} / {durations.max():.2f} s")
        print(f"  total motion: {durations.sum() / 3600:.2f} h")
        print(f"  min_z    min/mean/max: {min_zs.min():+.4f} / {min_zs.mean():+.4f} / {min_zs.max():+.4f} m")
    print(f"  manifest: {manifest}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
