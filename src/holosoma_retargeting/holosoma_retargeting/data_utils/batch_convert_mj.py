#!/usr/bin/env python3
"""Run data_conversion/convert_data_format_mj.py over a directory, in parallel.

The upstream script takes one --input-file and one --output-name per invocation, which is
impractical for thousands of clips. This shells out to it concurrently and records a
per-clip manifest so failures are visible rather than silently missing from the output.

Each worker is a separate process invoking the upstream CLI unmodified, so the conversion
itself is exactly what a manual single-clip run would produce.

    python batch_convert_mj.py --input-dir <retarget/> --output-dir <motions_50fps/> \
        --robot r1 --data-format soma --input-fps 30 --output-fps 50 --workers 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SCRIPT = PKG / "data_conversion" / "convert_data_format_mj.py"


def convert_one(args: tuple) -> dict:
    src, dst, robot, fmt, obj, in_fps, out_fps, python = args
    rec = {"clip_id": Path(src).stem, "ok": False}
    t0 = time.time()
    cmd = [
        python, str(SCRIPT),
        "--input-file", str(src),
        "--robot", robot,
        "--data-format", fmt,
        "--object-name", obj,
        "--input-fps", str(in_fps),
        "--output-fps", str(out_fps),
        "--headless", "--once",
        "--output-name", str(dst),
    ]
    try:
        p = subprocess.run(cmd, cwd=str(PKG), capture_output=True, text=True, timeout=900)
        if p.returncode == 0 and Path(f"{dst}.npz").exists():
            rec["ok"] = True
        else:
            tail = (p.stderr or p.stdout or "").strip().splitlines()
            rec["error"] = tail[-1][:300] if tail else f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout"
    except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the batch
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--robot", default="r1")
    p.add_argument("--data-format", default="soma")
    p.add_argument("--object-name", default="ground")
    p.add_argument("--input-fps", type=int, default=30)
    p.add_argument("--output-fps", type=int, default=50)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.input_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {a.input_dir}")

    # The retargeter names outputs "<clip>_<augmentation>.npz"; strip it so downstream clip
    # IDs match the manifest.
    jobs = []
    for f in files:
        stem = f.stem[: -len("_original")] if f.stem.endswith("_original") else f.stem
        dst = out / stem
        if not a.overwrite and Path(f"{dst}.npz").exists():
            continue
        jobs.append((f, dst, a.robot, a.data_format, a.object_name, a.input_fps, a.output_fps, a.python))

    print(f"converting {len(jobs)} clip(s) {a.input_fps} -> {a.output_fps} fps, {a.workers} workers")
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  FAIL {r['clip_id']}: {r.get('error')}")
            if i % 200 == 0 or i == len(jobs):
                ok = sum(x["ok"] for x in results)
                print(f"  [{i}/{len(jobs)}] ok={ok} fail={len(results)-ok}", flush=True)

    man = out / "conversion_manifest.jsonl"
    with man.open("w") as fh:
        for r in sorted(results, key=lambda x: x["clip_id"]):
            fh.write(json.dumps(r) + "\n")
    ok = sum(r["ok"] for r in results)
    print(f"\nconverted {ok}/{len(results)}; manifest {man}")
    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
