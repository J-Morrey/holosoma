#!/usr/bin/env python3
"""Draw a reproducible, stratified sample of BONES-SEED clip IDs.

An alphabetical or first-N sample of SEED is close to useless for finding failure modes:
86% of the 142,220 clips are ``standing``, and the interesting cases (crawling, ground
contact, jumping, stunts) are each well under 2% of the corpus. A naive 10% sample would
be ~10% of a monoculture.

This sampler therefore does two things:

1. **Guarantees hard-case coverage.** Named buckets (ground contact, crouching, jumping,
   fast dynamics, climbing, inverted, stunts, impaired gait) each get a floor quota, so
   they appear in force even though they are rare.
2. **Stratifies the bulk** across the (category x duration-tercile) grid in proportion to
   the corpus, and within each cell spreads draws round-robin over distinct performers so
   the 522 actors are covered as evenly as the cell size allows.

Requires pandas + pyarrow. On this machine that is the base interpreter, not the
``hsretargeting`` env:

    /home/mindgoblin/miniconda3/bin/python sample_bones_seed.py \
        --metadata /mnt/fast/bones-seed/metadata/seed_metadata_v004.parquet \
        --out-dir ./pilot_soma_r1 --target 14222
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Hard-case buckets, in priority order: a clip is assigned to the first bucket it matches,
# so the rarer and more mechanically demanding buckets win ties.
#
# Patterns are matched case-insensitively as substrings against the concatenation of
# content_body_position, content_type_of_movement, category and content_uniform_style.
# Note the corpus contains the misspelling "croaching" alongside "crouching" (874 vs 2590
# clips), so both spellings are listed -- matching on "crouch" alone silently drops a
# third of the crouching data.
HARD_BUCKETS: dict[str, str] = {
    "inverted": r"handstand|standing on hands|on hands",
    "stunts_martial": r"\bstunts\b|martial arts|\bmagic\b",
    "ground_contact": r"crawl|all fours|lying|sitting on floor|kneel|on heels|sitting on heels",
    "climbing": r"climb",
    "jumping": r"jump|leap|\bhop\b",
    "fast_dynamic": r"run|sprint|\bfast\b|roll|dive|kick|\bfall|hurry",
    "crouching": r"crouch|croach",
    "impaired": r"injured",
    "sitting_furniture": r"sitting on chair|sitting on bench|\bsitting\b",
}

# Floor quota per hard bucket, as a fraction of the overall target. These sum to well under
# 1.0; whatever is left goes to the stratified bulk draw.
HARD_BUCKET_SHARE: dict[str, float] = {
    "inverted": 0.010,
    "stunts_martial": 0.015,
    "ground_contact": 0.070,
    "climbing": 0.030,
    "jumping": 0.070,
    "fast_dynamic": 0.040,
    "crouching": 0.035,
    "impaired": 0.020,
    "sitting_furniture": 0.020,
}

SOURCE_FPS = 120.0


@dataclass
class Config:
    metadata: str
    out_dir: str
    target: int = 14222
    seed: int = 20260821
    max_duration_s: float = 60.0
    min_duration_s: float = 1.0


def _parse_args() -> Config:
    # argparse rather than tyro: this script only needs pandas, so it can run under the
    # base interpreter, which is the one that has pandas on this machine.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="Path to seed_metadata_v004.parquet")
    p.add_argument("--out-dir", required=True, help="Directory for the clip-ID list and manifest")
    p.add_argument("--target", type=int, default=14222, help="Clips to sample (14222 = 10%% of 142220)")
    p.add_argument("--seed", type=int, default=20260821, help="RNG seed; fixed for reproducibility")
    p.add_argument(
        "--max-duration-s",
        type=float,
        default=60.0,
        help="Drop clips longer than this. The corpus max is 21617 frames (180 s at 120 fps); "
        "a few very long takes would dominate pilot wall-clock without adding proportionate coverage.",
    )
    p.add_argument("--min-duration-s", type=float, default=1.0, help="Drop clips shorter than this")
    a = p.parse_args()
    return Config(
        metadata=a.metadata,
        out_dir=a.out_dir,
        target=a.target,
        seed=a.seed,
        max_duration_s=a.max_duration_s,
        min_duration_s=a.min_duration_s,
    )


def _label_hard(df: pd.DataFrame) -> pd.Series:
    haystack = (
        df.content_body_position.fillna("").astype(str)
        + " | "
        + df.content_type_of_movement.fillna("").astype(str)
        + " | "
        + df.category.fillna("").astype(str)
        + " | "
        + df.content_uniform_style.fillna("").astype(str)
    ).str.lower()

    out = pd.Series("bulk", index=df.index, dtype=object)
    for name, pattern in HARD_BUCKETS.items():
        rx = re.compile(pattern, re.I)
        hit = haystack.str.contains(rx) & (out == "bulk")
        out[hit] = name
    return out


def _round_robin_by_actor(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Pick n clips from pool, cycling over distinct performers to spread actor coverage."""
    if n <= 0 or pool.empty:
        return []
    if n >= len(pool):
        return pool.filename.tolist()

    by_actor: dict[str, list[str]] = {}
    for actor, grp in pool.groupby("take_actor", sort=False):
        ids = grp.filename.tolist()
        rng.shuffle(ids)
        by_actor[actor] = ids

    actors = list(by_actor)
    rng.shuffle(actors)

    picked: list[str] = []
    while len(picked) < n:
        progressed = False
        for actor in actors:
            if not by_actor[actor]:
                continue
            picked.append(by_actor[actor].pop())
            progressed = True
            if len(picked) == n:
                break
        if not progressed:
            break
    return picked


def main(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.metadata)
    n_total = len(df)

    df["duration_s"] = df.move_duration_frames / SOURCE_FPS
    eligible = df[(df.duration_s >= cfg.min_duration_s) & (df.duration_s <= cfg.max_duration_s)].copy()
    print(f"corpus {n_total}, eligible after duration filter " f"[{cfg.min_duration_s}, {cfg.max_duration_s}] s: {len(eligible)}")

    eligible["hard_bucket"] = _label_hard(eligible)
    eligible["dur_bin"] = pd.qcut(eligible.duration_s, 3, labels=["short", "mid", "long"], duplicates="drop")

    print("\nhard-bucket availability in corpus:")
    avail = eligible.hard_bucket.value_counts()
    for name in list(HARD_BUCKETS) + ["bulk"]:
        print(f"  {name:20s} {int(avail.get(name, 0)):>7d}")

    picked: dict[str, str] = {}  # filename -> bucket it was drawn for

    # --- 1. hard-case floor quotas -------------------------------------------
    print("\nhard-case draw:")
    for name in HARD_BUCKETS:
        quota = int(round(cfg.target * HARD_BUCKET_SHARE[name]))
        pool = eligible[eligible.hard_bucket == name]
        take = min(quota, len(pool))
        for fid in _round_robin_by_actor(pool, take, rng):
            picked[fid] = name
        note = "" if take == quota else f"  (capped: only {len(pool)} available)"
        print(f"  {name:20s} quota {quota:>5d} -> took {take:>5d}{note}")

    # --- 2. stratified bulk fill ---------------------------------------------
    remaining = cfg.target - len(picked)
    print(f"\nbulk draw: {remaining} clip(s) across category x duration bins")

    bulk = eligible[~eligible.filename.isin(picked)]
    cells = bulk.groupby(["category", "dur_bin"], observed=True)
    weights = cells.size()
    total_w = weights.sum()

    # Largest-remainder allocation so the per-cell counts sum exactly to `remaining`.
    exact = weights / total_w * remaining
    alloc = np.floor(exact).astype(int)
    deficit = remaining - int(alloc.sum())
    if deficit > 0:
        order = (exact - alloc).sort_values(ascending=False).index
        for key in list(order)[:deficit]:
            alloc[key] += 1

    for key, grp in cells:
        n = int(alloc.loc[key])
        for fid in _round_robin_by_actor(grp, n, rng):
            picked[fid] = "bulk"

    # --- 3. write out ---------------------------------------------------------
    sample = eligible[eligible.filename.isin(picked)].copy()
    sample["sample_bucket"] = sample.filename.map(picked)
    sample = sample.sort_values("filename").reset_index(drop=True)

    ids_path = out_dir / "sample_clip_ids.txt"
    ids_path.write_text("\n".join(sample.filename) + "\n")

    keep_cols = [
        "filename",
        "sample_bucket",
        "category",
        "package",
        "take_actor",
        "actor_height_cm",
        "actor_gender",
        "move_duration_frames",
        "duration_s",
        "content_body_position",
        "content_type_of_movement",
        "content_uniform_style",
        "is_mirror",
        "move_soma_uniform_path",
        "move_g1_path",
    ]
    manifest_path = out_dir / "sample_manifest.csv"
    sample[keep_cols].to_csv(manifest_path, index=False)

    bvh_paths = out_dir / "sample_soma_uniform_paths.txt"
    bvh_paths.write_text("\n".join(sample.move_soma_uniform_path) + "\n")
    g1_paths = out_dir / "sample_g1_paths.txt"
    g1_paths.write_text("\n".join(sample.move_g1_path) + "\n")

    # --- 4. report -----------------------------------------------------------
    print(f"\nsampled {len(sample)} clip(s) = {len(sample) / n_total:.2%} of corpus")
    print(f"  distinct performers : {sample.take_actor.nunique()} / {df.take_actor.nunique()}")
    print(f"  distinct categories : {sample.category.nunique()} / {df.category.nunique()}")
    print(f"  total motion        : {sample.duration_s.sum() / 3600:.2f} h")
    print(f"  duration min/med/max: {sample.duration_s.min():.1f} / {sample.duration_s.median():.1f} / {sample.duration_s.max():.1f} s")
    print("\nby sample_bucket:")
    print(sample.sample_bucket.value_counts().to_string())
    print("\nby category (sample vs corpus share):")
    s_share = sample.category.value_counts(normalize=True)
    c_share = df.category.value_counts(normalize=True)
    for cat in c_share.index:
        print(f"  {cat:28s} {s_share.get(cat, 0.0):6.2%} vs {c_share[cat]:6.2%}   (n={int(sample.category.eq(cat).sum())})")

    print(f"\nwrote:\n  {ids_path}\n  {manifest_path}\n  {bvh_paths}\n  {g1_paths}")


if __name__ == "__main__":
    main(_parse_args())
