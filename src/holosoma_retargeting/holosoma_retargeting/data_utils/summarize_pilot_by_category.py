#!/usr/bin/env python3
"""Join retargeting quality metrics to BONES-SEED metadata and rank categories by difficulty.

Produces the filter list for a full run: which motion categories and body positions the
target robot cannot physically represent, with concrete example clip IDs.

Joint-limit saturation is the primary signal. A clip pinned against its joint stops is
asking for more range than the embodiment has, which is an infeasible *motion* rather than
a failed *solve* -- the distinction that matters when deciding what to exclude versus what
to re-tune. Foot sliding and ground penetration are reported alongside as secondary
evidence.

Requires pandas, so run it with the base interpreter:

    /home/mindgoblin/miniconda3/bin/python summarize_pilot_by_category.py \
        --metrics quality.json --manifest pilot_soma_r1/sample_manifest.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# A clip above this joint-saturation fraction is flagged as likely embodiment-limited.
SAT_FLAG = 0.10
MIN_GROUP = 8  # don't rank groups smaller than this; the estimate would be noise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", required=True, help="JSON from analyze_retarget_quality.py --json")
    p.add_argument("--manifest", required=True, help="sample_manifest.csv from sample_bones_seed.py")
    p.add_argument("--out-md", default=None, help="Write markdown tables here")
    p.add_argument("--examples", type=int, default=3, help="Example clip IDs per flagged group")
    args = p.parse_args()

    m = pd.DataFrame(json.loads(Path(args.metrics).read_text()))
    meta = pd.read_csv(args.manifest)

    # robot_retarget.py names outputs "<clip>_<aug>.npz", and robot_only uses the single
    # augmentation "original". Strip that suffix so clip IDs join against the manifest.
    m["clip_id"] = m.clip_id.str.replace(r"_original$", "", regex=True)

    df = m.merge(meta, left_on="clip_id", right_on="filename", how="inner")
    unmatched = len(m) - len(df)
    if unmatched:
        print(f"note: {unmatched} metric row(s) had no manifest match and were dropped")
    if df.empty:
        raise SystemExit("no overlap between metrics and manifest")

    df["sat_flag"] = df.joint_sat_frac > SAT_FLAG
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"Analyzed {len(df)} retargeted clip(s) joined to metadata "
         f"({df.filename.nunique()} unique, {df.take_actor.nunique()} performers).")
    emit()
    emit(f"Overall: joint-limit saturation mean {df.joint_sat_frac.mean():.2%}, "
         f"p95 {df.joint_sat_frac.quantile(0.95):.2%}; "
         f"{df.sat_flag.mean():.1%} of clips exceed the {SAT_FLAG:.0%} flag threshold.")
    emit()

    for key, label in (("category", "Category"), ("sample_bucket", "Hard-case bucket"), ("content_body_position", "Body position")):
        g = df.groupby(key).agg(
            n=("clip_id", "size"),
            sat_mean=("joint_sat_frac", "mean"),
            flagged=("sat_flag", "mean"),
            slide_p95=("foot_slide_p95_mps", "mean"),
            pen_max=("ground_pen_max_m", "mean"),
            selfcol=("self_collide_min_m", "mean"),
        )
        g = g[g.n >= MIN_GROUP].sort_values("sat_mean", ascending=False)
        if g.empty:
            continue
        emit(f"### {label} ranked by joint-limit saturation")
        emit()
        emit(f"| {label} | n | joint sat | % flagged | foot slide p95 (m/s) | ground pen (m) | closest self-dist (m) |")
        emit("|---|---:|---:|---:|---:|---:|---:|")
        for name, r in g.iterrows():
            emit(f"| {name} | {int(r.n)} | {r.sat_mean:.2%} | {r.flagged:.0%} | "
                 f"{r.slide_p95:.3f} | {r.pen_max:.4f} | {r.selfcol:+.4f} |")
        emit()

        worst = g.head(5).index.tolist()
        emit(f"Example clip IDs for the {label.lower()} groups with the highest saturation:")
        emit()
        for name in worst:
            sub = df[df[key] == name].nlargest(args.examples, "joint_sat_frac")
            ids = ", ".join(f"`{r.clip_id}` ({r.joint_sat_frac:.0%})" for _, r in sub.iterrows())
            emit(f"- **{name}** — {ids}")
        emit()

    emit("### Worst individual clips by joint-limit saturation")
    emit()
    emit("| clip | category | body position | joint sat | worst joint |")
    emit("|---|---|---|---:|---|")
    for _, r in df.nlargest(15, "joint_sat_frac").iterrows():
        emit(f"| `{r.clip_id}` | {r.category} | {r.content_body_position} | "
             f"{r.joint_sat_frac:.1%} | {r.joint_sat_worst} |")
    emit()

    sat_by_joint = df.joint_sat_worst.value_counts()
    emit("### Which R1 joints saturate most often (count of clips where each is the worst offender)")
    emit()
    for j, c in sat_by_joint.head(12).items():
        emit(f"- `{j}` — {c} clip(s)")
    emit()

    if args.out_md:
        Path(args.out_md).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
