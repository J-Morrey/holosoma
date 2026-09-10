# Full BONES-SEED → Unitree R1 retargeting on SLURM

Runbook for the full 142,220-clip run. Derived from the 1% pilot
(`/mnt/fast/soma_r1_1pct`), which completed 1,422/1,422 with zero failures.

---

## 0. Which node — the CPU one, decisively

**Use the 72-core server node. Do not use a GPU node.**

Verified in the source, not assumed:

- The QP is solved by **CLARABEL** (`interaction_mesh_retargeter.py:757`), a CPU-only
  interior-point solver. No CUDA build exists.
- `grep` for `torch|cuda|cupy|jax` across `interaction_mesh_retargeter.py`,
  `robot_retarget.py` and `parallel_robot_retarget.py` returns **nothing**.
- MuJoCo is used only for FK, Jacobians and distance queries — CPU paths.

An RTX 5090 or Blackwell 6000 would sit completely idle. The workload is embarrassingly
parallel across clips and scales with core count.

**Thread pinning is not worth it.** Measured on 40 clips / 22 workers: default threading
268 s vs `OMP_NUM_THREADS=1` 258 s — a **3.7%** difference. Results were **bit-identical
40/40**, so thread count does not perturb output. Set it anyway (costs nothing, guarantees
reproducibility) but do not expect a speedup.

---

## 1. Environment setup

```bash
# Conda env (the pilot used Python 3.11)
conda create -n hsretarget python=3.11 -y && conda activate hsretarget

git clone <your-fork>/holosoma && cd holosoma
git checkout add_soma_format          # the branch with SOMA support + the R1 fixes
pip install -e src/holosoma_retargeting

# Verify: must print the 4 R1-specific fixes as present
python - <<'EOF'
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.data_type import MotionDataConfig
import mujoco, os
os.chdir(os.path.dirname(__import__('holosoma_retargeting').__file__))
rc = RobotConfig(robot_type="r1")
assert rc.Q_INIT_SEED == {"3": 0.6, "9": 0.6}, "knee seed missing"
assert "23" in rc.MANUAL_COST, "shoulder-yaw rest cost missing"
m = mujoco.MjModel.from_xml_path("models/r1/r1_26dof.xml")
assert m.nq == 33 and m.nbody == 40, "hand keypoint frames missing"
assert MotionDataConfig(data_format="soma", robot_type="r1").resolved_joints_mapping["LeftHand"] == "left_hand_link"
print("OK: soma format registered, all four R1 fixes present")
EOF
```

`pandas`/`pyarrow` are needed only for the sampler/metadata step and are **not** in this
env by design — run those with any interpreter that has them.

---

## 2. Stage the data

The 45 GB `soma_uniform.tar.gz` is **gzip, so not seekable** — extracting any subset costs a
full streaming pass (~1 h). Extract everything once.

```bash
DATA=/scratch/$USER/seed          # use node-local or fast parallel scratch, not NFS home
mkdir -p $DATA && cd $DATA

# ~1 h, produces ~150 GB of BVH
tar xzf /path/to/soma_uniform.tar.gz -C $DATA

# BVH -> human-keypoint npz, 30 fps. Fast: ~12 min for 14k clips at 20 workers
python data_utils/prep_soma_bvh_for_rt.py \
    --input-dir $DATA/soma_uniform/bvh \
    --output-dir $DATA/npz --target-fps 60 --workers 28
```

Omit `--clip-list` to convert the whole corpus. Conversion had a **0% failure rate** on the
pilot's 14,222 clips.

**Consider 60 fps instead.** SEED is natively 120 fps; the pilot used 30, which means the
final 50 fps output contains interpolated frames. `--target-fps 60` divides 120 exactly and
downsamples to 50 rather than interpolating. It costs ~2× the retargeting time. Decide now —
changing later means redoing everything.

### Disk budget

| item | full-corpus size |
|---|---|
| extracted BVH | ~150 GB (deletable after step 2) |
| `npz/` human keypoints | ~6 GB |
| `retarget/` (30 fps qpos) | ~16 GB |
| **`motions_50fps/`** | **~230 GB** |

Extrapolated from the 1% pilot (2.3 GB for 1,422 clips). Provision **≥ 400 GB**, or
~550 GB if you keep the BVH.

---

## 3. Shard for SLURM

`parallel_robot_retarget.py` globs its `--data-dir` and has **no `--clip-list`**, so shard
by building per-shard symlink directories.

```bash
python - <<'EOF'
import pathlib, math
SRC   = pathlib.Path("/scratch/$USER/seed/npz")
OUT   = pathlib.Path("/scratch/$USER/seed/shards")
NSHARD = 32
files = sorted(SRC.glob("*.npz"))
OUT.mkdir(parents=True, exist_ok=True)
# Round-robin, NOT contiguous: clip IDs are date-prefixed, so contiguous blocks would give
# one shard all the long takes from a single session and badly skew shard runtimes.
for i, f in enumerate(files):
    d = OUT / f"{i % NSHARD:03d}"
    d.mkdir(exist_ok=True)
    t = d / f.name
    if not t.exists():
        t.symlink_to(f)
print(f"{len(files)} clips into {NSHARD} shards, ~{math.ceil(len(files)/NSHARD)} each")
EOF
```

Round-robin matters: SEED clip IDs carry a date prefix, so contiguous blocks concentrate
each capture session — and its characteristic clip lengths — into one shard.

---

## 4. Retarget

If the 72-core node is a single machine, **one job with 64–68 workers is simplest** and
avoids all sharding. Use the array form only if you must checkpoint against wall-clock
limits or spread across several CPU nodes.

### Single node, no sharding

```bash
#!/bin/bash
#SBATCH --job-name=r1-retarget
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/retarget-%j.out

source ~/miniconda3/etc/profile.d/conda.sh && conda activate hsretarget
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATA=/scratch/$USER/seed
cd $(python -c "import holosoma_retargeting,os;print(os.path.dirname(holosoma_retargeting.__file__))")

python examples/parallel_robot_retarget.py \
    --data-dir $DATA/npz \
    --task-type robot_only --robot r1 --data-format soma \
    --save-dir $DATA/retarget \
    --task-config.object-name ground --task-config.ground-range -15 15 \
    --retargeter.foot-sticking-tolerance 0.02 \
    --max-workers 68
```

Leave ~4 cores free for I/O and the parent process.

### Array form, if you need checkpointing

```bash
#SBATCH --array=0-31%8
#SBATCH --cpus-per-task=8
...
SH=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
python examples/parallel_robot_retarget.py \
    --data-dir $DATA/shards/$SH --save-dir $DATA/retarget \
    ... --max-workers 8
```

All shards write to one `--save-dir`; filenames are unique per clip so there is no
collision. The runner **skips clips whose output already exists**, so a requeued or
timed-out job resumes cleanly — no special restart handling needed.

### The parameters, and why

| flag | value | reason |
|---|---|---|
| `--retargeter.foot-sticking-tolerance` | `0.02` | Swept 0.001–0.10. The 0.001 default is measurably worse; ≥0.02 is equivalent because the constraint stops binding. Matches the documented LAFAN value. |
| `--task-config.ground-range` | `-15 15` | SEED clips travel up to ~10 m; the README's `-10 10` would clip them. |
| `--task-type` / `--object-name` | `robot_only` / `ground` | SEED ships no object or terrain annotations. |

Do **not** pass `--robot-config.*` overrides. They are unnecessary — the R1 defaults on this
branch already carry the fixes — and `robot_retarget.py:623` silently discards them unless
you also pass `--robot-config.robot-type r1`.

---

## 5. Convert to RL-ready 50 fps

```bash
#SBATCH --cpus-per-task=72 --time=12:00:00
python data_utils/batch_convert_mj.py \
    --input-dir $DATA/retarget --output-dir $DATA/motions_50fps \
    --robot r1 --data-format soma --input-fps 30 --output-fps 50 --workers 64
```

Measured 4.2 s/clip; ~2.6 h for the full corpus at 64 workers. Also resumable — it skips
existing outputs. **If you chose 60 fps in step 2, pass `--input-fps 60`.**

---

## 6. Validate before training

```bash
# Retargeting quality
python data_utils/analyze_retarget_quality.py \
    --results-dir $DATA/retarget --robot r1 --data-format soma \
    --quiet --json $DATA/quality.json

# Category breakdown (needs pandas)
python data_utils/summarize_pilot_by_category.py \
    --metrics $DATA/quality.json --manifest <seed_metadata_v004-derived manifest> \
    --out-md $DATA/quality_report.md
```

Compare against the 1% pilot, which is the acceptance baseline:

| metric | pilot value | flag if |
|---|---|---|
| `ground_pen_max_m` | **0.00000** | anything > 0 |
| `knee_extension_frac` | 0.9% | > 5% (knee-seed fix regressed) |
| `track_err_mean_m` | 0.0377 | > 0.05 |
| `joint_sat_frac` | 4.3% | > 8% |
| `self_collide_min_m` | −0.032 | expected; see limitations |
| failure rate | **0%** | any non-zero |

---

## 7. Time and cost

Measured: **8.15 clips/min on 24 cores** (22 workers), steady state.

Scaling is near-linear in cores for this workload, but memory bandwidth will take a cut at
72-way, so assume 2.5–3× rather than 3×:

| | throughput | 142,220 clips |
|---|---|---|
| 24 cores (the pilot box) | 8.15/min | ~12 days |
| **72 cores, 2.5× assumed** | ~20/min | **~5 days** |
| 72 cores, 3× optimistic | ~24/min | ~4 days |

At 60 fps input, double those.

**Measure before committing.** Run one shard (~4,400 clips) first, time it, and extrapolate.
Do not trust my 2.5× guess — I got a parallel-scaling estimate wrong by 4× earlier in this
project by extrapolating single-clip time.

---

## 8. Two things that will bite

**`find_files()` sorts.** A partially completed run yields an *alphabetical* prefix, not a
random subset. During the pilot the first 44 clips covered only 5 performers and were 43/44
`standing` — it looked like a clean result and was badly unrepresentative. **Never judge
quality from a partial run.** If you must, analyse a random subset of completed clips.

**Beware `pkill -f` in job scripts.** The pattern will match the wrapper's own command line
and kill its process group; `nohup` does not protect against SIGTERM. This silently killed a
pilot run at 7/83. Under SLURM use `scancel`; if you must detach, use `setsid` and signal a
completion marker file rather than polling `pgrep`.

---

## 9. Known limitations to carry forward

Fully documented in `PILOT_REPORT.md` §11. The short version:

- **Ground contact is clean** — zero penetration, foot sliding ~1.3 cm/s. This is the
  artifact that actually breaks tracking policies, and it is absent.
- **Self-penetration is present** — hands into hips, ~28% of frames at ~4 cm. A genuine
  embodiment limit; six candidate fixes were tested and falsified. No mainstream G1
  pipeline enforces self-collision either. Expect it to surface as **cross-seed MPJPE
  variance, not in reward curves** — evaluate across ≥5 seeds.
- **References are kinematic only** — no torque or dynamic-feasibility constraints.
- **No filtering was applied.** If you want a quality gate, `analyze_retarget_quality.py`
  emits `leg_pin_frac` and `knee_extension_frac` per clip; the field convention (NMR)
  rejects a clip when >5% of its frames self-intersect.
