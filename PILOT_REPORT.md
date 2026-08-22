# SOMA → OmniRetarget → Unitree R1 pilot report

**Date:** 2026-08-21
**Branch:** `add_soma_format` (7 commits, off `add_R1`)
**Scope:** add a `soma` motion format to the retargeting pipeline and pilot-retarget a
stratified 10% sample of BONES-SEED to the Unitree R1.

---

## TL;DR

`--data-format soma` works end to end on the R1 config. The loader is correct to a high
standard of evidence, and conversion of the full 10% sample succeeded on **14,222/14,222
clips with zero failures**. Retargeting itself has a **0% exception rate** across every clip
attempted.

There is **one real quality defect**: R1 drives its own hands into its hips in ~90% of
clips, up to 7.6 cm deep. It is not a loader bug and it is not fixable with the obvious
knob — enabling the self-collision constraint makes the QP *infeasible* even at a 5 mm
margin. That needs a decision before a full run.

The full 142K run projects to **~8–10 days** on this 24-core box. Not weeks, but not
overnight either, and worth parallelising across machines.

**Recommendation: fix the wrist/hip issue, then proceed.** Everything else is ready.

---

## 1. Did it work?

### Pipeline stages

| Stage | Result |
|---|---|
| SOMA BVH parse + FK | 14,222 / 14,222 clips (100%) |
| BVH → `.npz` conversion | **14,222 / 14,222 (100%), 0 failures** |
| Retarget to R1 (`robot_only`) | **0 exceptions / 0 CVXPY failures** on all clips attempted |
| `convert_data_format_mj.py --output_fps 50` | works; emits the full RL schema |
| `evaluation/eval_retargeting.py` | works (after fixing two pre-existing bugs) |

Conversion stats for the 10% sample: height 1.766–1.767 m across all 14,222 clips
(a 1 mm spread, confirming `soma_uniform` is genuinely proportion-normalised), duration
1.1–59.1 s, **27.9 h of motion**.

### Retarget coverage caveat — read this

Retargeting the full 14,222 takes ~20 h, which exceeded the session. Retarget statistics
below come from **two** runs:

- **66 clips** in `find_files()`'s sorted order. Not representative: only 5 performers, 29
  of 44 analysed clips were `Basic Locomotion Neutral`, 43 of 44 `standing`.
- **A stratified 700-clip subset** (`pilot_soma_r1/stratified_subset_ids.txt`), a uniform
  draw from the already-stratified manifest, covering **351 performers, 19 categories and
  all 10 hard-case buckets**. This is the representative set and the basis for §3.

This is itself a finding: **`find_files()` sorts, so a partially completed run yields an
alphabetical prefix.** Any interrupted full run gives biased partial results. Fix before
the full run — see §6.

### Failure taxonomy

Nothing failed at the solver level. The failures encountered were all in *tooling*, all
pre-existing, and all fixed:

| Failure | Cause | Status |
|---|---|---|
| `TypeError: 'NoneType' is not iterable` | `--retargeter.debug` without `--visualize`: `draw_keypoints` returns `None` with no viser server, then the cleanup path iterates it. Affects every format and robot. | Documented; use both flags together |
| `FileNotFoundError` in eval | `evaluate_robot_only_trajectory` probed only `.pt`/`.npy`. Also broke `smplx`. | Fixed (`d9bdb87`) |
| `KeyError: 'LeftToeBase'` | `extract_foot_sticking_sequence_velocity` hardcoded `L_Toe`/`R_Toe` dict keys, ignoring its own `foot_names` arg | Fixed (`3030a7d`) |
| 327 clips "implausible height" | My own first-pass stature estimator — see §2.1 | Fixed (`b53db67`) |

---

## 2. Loader correctness

This was the main risk: a silently wrong loader produces confident-looking garbage that
only surfaces much later as inexplicably bad policies. Four independent lines of evidence.

### 2.1 Internal consistency (`validate_soma_loader.py`)

7/7 reference clips pass all 9 checks. Worst bone-length coefficient of variation
**3.7e-15** — machine precision, meaning the per-joint channel slicing and hierarchy parse
are exactly right.

Crucially, the checks **discriminate** rather than merely pass. Both plausible-but-wrong
conventions are rejected 0/7:

| Configuration | Result |
|---|---|
| Correct (extrinsic Euler, det=+1 world matrix) | **7/7 pass** |
| Intrinsic Euler instead | 0/7 — `upright` and `chirality_eyes` fail |
| LAFAN's `transform_y_up_to_z_up` instead | 0/7 — `chirality` 0%, `facing` −0.938 |

### 2.2 Two conventions upstream disagreed; the data settled both

**Euler composition.** GMR PR #169 uses scipy-*extrinsic* (`R = Rx·Ry·Rz`);
`soma-retargeter`'s Warp kernel accumulates `q *= axis_quat` over `[z,y,x]`
(`R = Rz·Ry·Rx`). These are not equivalent, and bone-length checks cannot separate them —
any rotation preserves bone length. Measured on `egypt_dance_R_003__A275`:

| | HeadEnd z | head above hips | cos(torso, +Z) |
|---|---:|---:|---:|
| **extrinsic** | **1.725 m** | **100.0%** | **+0.994** |
| intrinsic | 0.975 m | 52.6% | −0.006 |

Intrinsic collapses the figure. **Extrinsic is correct.** GMR's *comment* claims parity
with the Warp kernel; that claim does not survive this test, so the two upstream
implementations genuinely disagree and we follow the data.

**World matrix.** Deliberately **not** `src.utils.transform_y_up_to_z_up`, which the LAFAN
branch uses. That matrix is `[[1,0,0],[0,0,1],[0,1,0]]` with **det = −1** — a mirror. It
suits LAFAN's source convention but would flip left/right on SOMA's explicitly
right-handed data. We use the det = +1 permutation `(x,y,z) → (z,x,y)`, which is
bit-identical to `soma-retargeter`'s `MAYA` option.

Confirmed by `Neutral_walk_forward_002__A057` travelling **9.94 m at cos(travel,
body-forward) = +0.988** — the character walks in the direction it faces.

### 2.3 The stature bug I introduced and fixed

Worth recording because the failure was silent-adjacent. My first estimator read
`Hips.offset[1]` as hip height, reasoning that rest-pose feet sit at z=0. That holds for
standing takes but `Hips.offset` actually encodes the clip's **starting** root placement.
BONES-SEED's `sit_on_heels_*` takes have `Hips.offset = (0.27, 29.30, −12.08)` — a seated
root — versus `(0.00, 101.28, 0.00)` standing, while **every bone length in the two files
is identical to three decimals**.

Result: 1.06 m stature on 327 of 11,001 clips, correctly caught by the plausibility guard
as a suspected unit error. Now computed from bone-length chains only, scaled by a constant
calibrated against `soma_zero_frame0.bvh` where FK puts the feet at exactly z = 0.000000
(1.768537 m / 162.3761 cm → **1.089161**). Because it multiplies bone lengths rather than
assuming them, it also scales correctly for `soma_proportional`.

Post-fix: previously-failing clips give 1.7666 m vs 1.7668 m standing — a **0.2 mm** spread.

### 2.4 Independent cross-check against the shipped G1 reference (n=66)

The strongest evidence, because it is external. Predicted-vs-observed:

| Quantity | Predicted | Observed |
|---|---|---|
| Relative yaw vs NVIDIA (MAYA vs MUJOCO) | **−90.00°** | **−90.01°** (median) |
| Displacement ratio (1.2/1.767 ÷ 1.32/1.70) | 0.875 | 0.895 |
| Base-z ratio (1.2 / 1.32) | 0.909 | 0.9035 |
| XY path shape RMS after alignment | — | 0.086 m median |

The −90° yaw is **expected, not an error**: `soma-retargeter` ships
`retarget_source_facing_direction: "Mujoco"` = `Rx(+90°)`, and
`mujoco @ maya.T` is exactly −90° about Z. Both are proper rotations (det = +1), so nothing
is mirrored — the small shape RMS and two independently-predicted scale ratios confirm the
motion content agrees. Three independent predictions landing within ~2% is not consistent
with a coordinate bug.

---

## 3. Quality

### 3.1 `evaluation/eval_retargeting.py` (n=40)

| Metric | Value |
|---|---|
| `penetration_duration` | **0.000000** |
| `penetration_max_depths` | **0.000000** |
| `sliding_duration` | 0.0308 (3.1% of contact frames) |
| `max_toe_sliding_velocities` | **0.0126 m/s** ± 0.0022 |
| `opt_cost` | 0.2215 ± 0.0324 |

### 3.2 The four artifacts, called out individually

**Ground penetration — clean.** Exactly zero, by both the official eval and my analyzer,
across every clip. Your multi-point foot-contact sphere configuration is doing its job.

**Foot sliding — good, ~1.3 cm/s.** Trust the official eval's 0.0126 m/s. My
`analyze_retarget_quality.py` reports a much worse-looking 95% / 1.5 m/s because it infers
contact from toe height (< 3 cm), which counts airborne takeoff/landing frames during jumps
as contact. The official eval derives contact from the *human* motion's velocity and is the
right number. I flagged this caveat in the tool's docstring rather than quietly reporting
the flattering number.

**Joint-limit saturation — low, 2.7% mean / 5.1% p95.** No clip in the analysed set exceeded
a 10% flag threshold. R1's 26 DoF are coping with this motion better than I expected. The
joints that saturate most often are `ankle_roll` (walking/turning) and
`right_shoulder_roll` (jumping).

**Self-penetration — the real problem.** Hands into hips, in ~90% of clips:

| Body pair | Clips | Deepest |
|---|---:|---:|
| `right_hip_roll_link ↔ right_wrist_roll_link` | 18 | **−0.0762 m** |
| `left_hip_roll_link ↔ left_wrist_roll_link` | 9 | −0.0704 m |
| `right_hip_pitch_link ↔ right_wrist_roll_link` | 8 | −0.0613 m |
| `left_hip_pitch_link ↔ left_wrist_roll_link` | 5 | −0.0617 m |

This is not caught during retargeting because `SelfCollisionConfig.enable` defaults to
**`False`** with an empty `pairs` list, so no self-collision constraint is ever built.

### 3.3 My honest read of the rendered results

I rendered both the source human skeleton and the retargeted R1 (see
`data_utils/render_soma_*`/`render_retarget_frames.py`).

*Source human* (`Neutral_walk_forward_002__A057`): a clean, upright, anatomically correct
walking figure. Head at 1.75 m, feet planted on the ground line, legs alternating through a
proper gait cycle, body advancing 0 → 10 m along +X. One diagnostic detail visible: a
stray joint pinned at the world origin in every frame while the body walks away — that is
`Root`, which is why it is excluded from the keypoint set. Had I kept it, every clip would
have fed the interaction-mesh Laplacian a phantom joint anchored at the origin.

*Retargeted R1, walking*: **locomotion looks genuinely good.** Upright posture, feet
planted exactly on z=0, clean 0 → 7 m travel, legs alternating correctly. Two visible
compromises: the wrist markers sit almost directly on top of the hip markers (the
penetration, plainly visible), and the source head/shoulder keypoints sit consistently
*above* the R1 links — residual upper-body tracking error from proportion mismatch.

*Retargeted R1, throwing* (`Relaxed_throw_ball_003__A057`, the worst penetration case):
revealing. The penetration occurs in the **arms-at-rest** frames, not during the throw —
the raised-arm frames are clear. So the failure mode is the neutral arms-down pose: a
human's hands hang naturally beside the hips, and after scaling to R1's narrower torso
those targets land inside the hip geometry. That is exactly why it affects ~90% of clips —
nearly every clip has arms-down phases.

---

## 4. Foot-sticking tolerance sweep

25 runs, 5 mechanically distinct clips × 5 tolerances.

| tolerance | opt cost | joint sat | slide p95 (m/s) | ground pen |
|---:|---:|---:|---:|---:|
| 0.001 (default) | 0.30741 | 5.69% | 0.498 | 0.0000 |
| 0.005 | 0.30510 | 5.49% | 0.638 | 0.0000 |
| **0.02 (chosen)** | **0.30507** | 5.57% | 0.751 | 0.0000 |
| 0.05 | 0.30507 | 5.60% | 0.767 | 0.0000 |
| 0.10 | 0.30507 | 5.63% | 0.736 | 0.0000 |

**The tolerance is a weak lever on this data.** Cost is identical to five decimals for
≥ 0.02: the constraint simply stops binding once the window exceeds ~5 mm, because the
natural solution already stays within it. The default 0.001 is measurably but marginally
worse (~1% higher cost, higher saturation, deeper self-penetration).

**Chosen: `--retargeter.foot-sticking-tolerance 0.02`** — matches the documented LAFAN
value, sits safely inside the flat region, lowest cost. Do not expect it to fix anything
else; the dominant artifact is not foot-related.

A subtlety worth knowing for future tuning: `velocity_threshold` in
`extract_foot_sticking_sequence_velocity` is compared against a raw `np.diff` of positions
with **no `dt`**, so the contact test is per-*frame*, not per-second, and its effective
strictness scales with output frame rate. Contact is also inferred from XY velocity alone
with no height check, so a toe can be flagged as sticking while airborne.

---

## 5. R1-specific findings

### Is `robot_only` the right tool? Partly — and you were right to flag it

Most SEED motion is free-space with no object or terrain annotation, so
`--task-type robot_only --task-config.object-name ground` is correct. But this means the
pilot exercises OmniRetarget's **kinematic-feasibility machinery and almost none of its
interaction-preservation**, which is its headline contribution. For SEED's free-space
locomotion and gesture bulk, OmniRetarget is close to a general-purpose IK retargeter, and
a lighter tool (GMR, `soma-retargeter` itself) would likely produce comparable results
faster.

Where OmniRetarget *would* earn its keep is the ~14,600 `Interactions` and ~3,400
`climbing box` clips, plus `Object Manipulation` (11,620) and `Object Interaction`
(10,817) — but SEED ships no object pose annotations, so the interaction-mesh objective
has nothing to preserve against. **If interaction preservation is why you chose
OmniRetarget, that value is not being realised on this dataset**, and it is worth deciding
that deliberately rather than by default.

### Categories at risk

With the honest caveat that no analysed clip exceeded a 10% joint-saturation flag, R1's
26 DoF handle this corpus better than expected. Ranked concerns:

1. **Everything with arms at rest — the wrist/hip issue.** Not a category, which is what
   makes it serious: it is not filterable. Affects ~90% of clips. Example IDs:
   `Relaxed_throw_ball_003__A057` (−0.0762 m), `Relaxed_throw_ball_003__A057_M`
   (−0.0704 m), `Relaxed_throw_ball_004__A057_M` (−0.0671 m).
2. **Jumping** — highest joint saturation seen (`Jump_Right_001__A017`, 6.3%, saturating
   `right_shoulder_roll_joint`). R1 has no wrist pitch/yaw, so arm counter-swing has
   nowhere to go.
3. **Walking and turning** — `ankle_roll` saturation ~5% (`Turn_Start_Walk_0315_001__A018`,
   `Neutral_walk_forward_002__A057`, `Turn_Start_Walk_0225_002__A017`). Benign but the
   most consistent saturator.
4. **Untested at scale in this pilot:** `inverted`/handstand (10 clips in the stratified
   subset), `stunts_martial` (13), `climbing` (15). These are the most likely to be
   infeasible and the sample is too small to rank them. `Martial Arts` has only 20 clips
   in the entire corpus and can be dropped without loss.

**Concrete R1 DoF gaps vs G1.** R1 has 26 actuated joints against G1's 29. The missing
capability that shows up in the data is the **wrist**: R1 exposes only
`{left,right}_wrist_roll_joint`, whereas the G1 reference targets `wrist_yaw_link`. One
wrist DoF instead of three is what prevents the hands tucking clear of the hips.

---

## 6. Cost projection

Measured throughput, 22 workers on 24 cores:

| | Value |
|---|---|
| Single clip, uncontended, 96 frames | 21.5 s → **0.224 s/frame** |
| Steady-state throughput, 22-way | **~8–12 clips/min** |
| 10% sample (14,222 clips, 27.9 h motion) | **~20–30 h (~1 day)** |
| **Full 142,220 clips (288 h motion)** | **~200–240 h = 8–10 days** |

Note the parallel-efficiency gap: a single solve uses all cores, so per-clip time under
22-way contention is ~4× the uncontended figure. Naively multiplying the single-clip time
by clip count and dividing by worker count **underestimates by ~4×** — my own first estimate
was 8 h and the real answer is ~1 day.

**Plainly: the full run is 8–10 days on this machine, not weeks.** It is embarrassingly
parallel across clips, so 4 machines brings it to ~2–3 days. Conversion is cheap
(14,222 clips in ~12 min at 20 workers, so the full set is ~2 h). Disk: `.npz` outputs are
~50 KB/clip, so the full run is ~7 GB — negligible. The 45 GB `soma_uniform.tar.gz` is
**not seekable**, so extracting any subset costs a full streaming pass (~1 h); extract
everything you will ever need in one go.

---

## 7. Frame-rate decision (and a corrected premise)

I used **30 fps** (120 → 30, integer decimation, no interpolation error).

**Your premise that the G1 CSVs were converted to 30 fps is incorrect.** I checked 8 clips
against `move_duration_frames`: the row ratio is exactly **1.000**. The shipped G1
reference is frame-for-frame at the source **120 fps**.

That weakens the main argument for 30 fps. Retargeting at 30 and then running
`convert_data_format_mj.py --output_fps 50` means 120 → 30 → 50, i.e. decimation followed
by *upsampling* — the 50 fps output contains interpolated frames that never existed.

I kept 30 fps for the pilot because it is 4× cheaper than 120 and human motion is mostly
below 10 Hz, so 30 fps is above Nyquist for the content. **For the full run I would use
60 fps**: it divides 120 exactly, and 60 → 50 is a downsample rather than an interpolation.
Cost is 2× the 30 fps figure (~16–20 days single-machine), which is the real trade.

---

## 8. What I would change before the full run

**Blocking:**

1. **Resolve the wrist/hip penetration.** Enabling the constraint is *not* the answer — I
   tested it. With pairs `{left,right}_hip_{roll,pitch}_link ↔ {left,right}_wrist_roll_link`
   the QP is **infeasible at 0.03, 0.015 and even 0.005 m** margins. The motion genuinely
   wants the hand where the hip is. Options, in my order of preference:
   - Add a small outward offset to the hand keypoint target in `JOINTS_MAPPINGS`, or map
     `LeftHand → left_elbow`-relative rather than to `wrist_roll_link`, so the target sits
     outside the hip volume. Cheapest and most likely to work.
   - Use `SelfCollisionConfig.windows` to enforce only on frames that actually violate,
     avoiding the global infeasibility.
   - Convert self-collision from a hard constraint to a soft penalty.
   - Accept it and filter, but at ~90% of clips that is not viable.

2. **Make `find_files()` order deterministic-but-shuffled** (or add `--limit`/`--shard`).
   As it stands an interrupted run yields an alphabetically biased prefix, which cost me a
   full re-run to get representative statistics.

**Worth doing:**

3. Switch to **60 fps** (§7).
4. **Shard across machines** — 8–10 days on one box, ~2–3 days on four.
5. Add a `--clip-list` option to `parallel_robot_retarget.py` so a curated subset does not
   need a symlink directory.
6. Consider whether OmniRetarget is the right tool for the free-space bulk (§5); a cheaper
   retargeter for the ~75% that is pure locomotion/gesture would cut the projection
   substantially, reserving OmniRetarget for interaction clips if object annotations ever
   land.
7. `--retargeter.debug` should not require `--visualize`; one-line guard in
   `interaction_mesh_retargeter.retarget_motion`.

---

## 9. Deliverables

| Item | Path |
|---|---|
| Loader (parser + FK) | `src/holosoma_retargeting/holosoma_retargeting/data_utils/soma_bvh.py` |
| BVH → npz converter | `data_utils/prep_soma_bvh_for_rt.py` |
| Loader validator | `data_utils/validate_soma_loader.py` |
| Stratified sampler | `data_utils/sample_bones_seed.py` |
| Quality analyzer | `data_utils/analyze_retarget_quality.py` |
| G1 reference comparison | `data_utils/compare_to_g1_reference.py` |
| Per-category summary | `data_utils/summarize_pilot_by_category.py` |
| Stick-figure renderer | `data_utils/render_retarget_frames.py` |
| Format registration | `config_types/data_type.py` (`SOMA_DEMO_JOINTS`, `("soma","r1")`, `("soma","g1")`) |
| **Reproducible clip list (14,222)** | `pilot_soma_r1/sample_clip_ids.txt` |
| Clip metadata + buckets | `pilot_soma_r1/sample_manifest.csv` |
| Stratified 700 subset | `pilot_soma_r1/stratified_subset_ids.txt` |

### Tuned config values

```bash
# Conversion
python data_utils/prep_soma_bvh_for_rt.py \
  --input-dir <soma_uniform/bvh> --output-dir <npz> \
  --clip-list pilot_soma_r1/sample_clip_ids.txt \
  --target-fps 30            # use 60 for the full run -- see section 7

# Retargeting
python examples/parallel_robot_retarget.py \
  --data-dir <npz> --task-type robot_only --robot r1 --data-format soma \
  --save-dir <out> \
  --task-config.object-name ground \
  --task-config.ground-range -15 15 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 22

# RL-ready conversion (per clip)
python data_conversion/convert_data_format_mj.py \
  --input-file <clip.npz> --robot r1 --data-format soma --object-name ground \
  --input-fps 30 --output-fps 50 --headless --once --output-name <out/clip>
```

`--task-config.ground-range -15 15` is wider than the LAFAN example's `-10 10` because SEED
clips travel up to ~10 m and the default would clip them.

---

## 10. Things I had to guess at

Flagged honestly; each is a real judgement call, not a certainty.

1. **Joint subset — 22 body keypoints.** Chosen to mirror LAFAN's 22 so the `("soma","r1")`
   mapping parallels your tested `("lafan","r1")` one-for-one. Dropped all 40 finger joints
   (your R1 mapping terminates at one wrist link, so fingers add no signal and would skew
   the Laplacian), 4 face joints, `Neck2` (keeping one neck joint matches LAFAN's single
   `Neck`), the toe tips, and `Root`. **Dropping `Root` is the one I am most confident
   about** — it is pinned at the world origin in every SEED clip and visibly so in the
   render.

2. **World-frame yaw.** I chose forward = +X (`MAYA`); NVIDIA's SEED G1 data uses
   forward = −Y (`MUJOCO`); OmniRetarget's own LAFAN path uses a third, mirrored
   convention. For flat-ground `robot_only` the retargeter is yaw-equivariant so it does
   not affect quality, but **if the downstream SONIC stage assumes the G1 reference frame,
   change `BVH_TO_WORLD` in `soma_bvh.py` to `[[1,0,0],[0,0,-1],[0,1,0]]`.** One line.

3. **Stature constant 1.089161.** Calibrated from one authoritative rest-pose file. It
   makes every `soma_uniform` clip 1.7667 m, which is self-consistent, but the absolute
   value rests on treating `HeadEnd` as the top of the skull. If true stature should include
   a scalp allowance, everything scales by a few mm — harmless, since only the *ratio*
   `ROBOT_HEIGHT / height` matters.

4. **A-pose handling: none, deliberately.** SOMA's rest pose is an A-pose, and both
   reference implementations do something about it (`soma-retargeter` solves against
   `soma_zero_frame0.bvh`; GMR bakes it into per-joint `rot_offset` quaternions). Neither
   is needed here: OmniRetarget's contract is world *positions*, so there is no orientation
   offset to factor out. I am confident in this, but it is the assumption I would revisit
   first if the arms look systematically wrong.

5. **`soma_uniform` over `soma_proportional`**, per your instruction. Verified correct:
   every clip yields 1.7667 m ± 0.2 mm, so per-sequence direct scaling sees no
   performer-proportion variance across 522 actors.

6. **Metadata is `v004`, not the `v003` in your brief** — `seed_metadata_v003.parquet` does
   not exist on disk; the dataset README's HF config block is stale. I used
   `seed_metadata_v004.parquet` (142,220 rows, matching the advertised count).

7. **`("soma","g1")` mapping added** beyond the ask, so the same clip can be retargeted to
   G1 as a control to separate loader bugs from embodiment limits. Unused so far — the
   shipped reference made it unnecessary — but it is there.

8. **Clips > 60 s excluded** from the sample (corpus max is 21,617 frames = 180 s). A
   handful of very long takes would otherwise dominate pilot wall-clock without adding
   proportionate coverage. This drops ~4% of the corpus; revisit for the full run.
