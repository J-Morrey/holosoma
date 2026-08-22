"""Parser and forward-kinematics for NVIDIA SOMA-format BVH motion capture.

SOMA BVH (as shipped in BONES-SEED and ``NVIDIA/soma-retargeter``) differs from the
LAFAN1/nokov BVH files the rest of this package consumes in ways that defeat the
vendored ``lafan1.extract.read_bvh`` reader:

* **Mixed channel counts.** ``Root`` and ``Hips`` carry 6 channels
  (``Xposition Yposition Zposition Zrotation Yrotation Xrotation``); the other 76
  joints carry 3 (``Zrotation Yrotation Xrotation``). The LAFAN reader assumes a
  single channel count for the whole file.
* **77 or 78 joints** including a full 40-joint finger set and 4 face joints.
* **Optional ``End Site`` blocks.** The ``NVIDIA/soma-retargeter`` samples contain 16;
  the BONES-SEED files contain none. Both must parse.
* **Variable ``OFFSET`` arity.** SEED motion files write 3 tokens, but
  ``soma_retargeter/configs/soma/soma_zero_frame0.bvh`` writes 6 (the extra 3 being a
  rest rotation). We take the first 3 and ignore any tail.

Conventions, from ``NVlabs/SOMA-X/assets/SOMA_procedural_transforms.json``
(``"conventions"`` block) and verified against BONES-SEED ``soma_uniform`` files:

    units: centimeters, up_axis: +Y, forward_axis: +Z, right_handed,
    local_euler_order: XYZ, 120 fps (Frame Time 0.008333)

Every bone extends along its own local **+X** axis, so drawing the hierarchy with
identity rotations yields a garbage figure with the spine and both legs collinear.
Full FK over the motion channels is mandatory.

This module deliberately exposes *positions only*. OmniRetarget's motion-data contract
is world joint positions ``(T, J, 3)``, so SOMA's A-pose calibration rest pose needs no
special handling -- there is no orientation offset to factor out, unlike a
rotation-retargeting consumer such as GMR or soma-retargeter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

# BVH is authored in centimeters; the retargeting pipeline works in meters.
CM_TO_M = 0.01

# Source frame rate of every BONES-SEED and soma-retargeter BVH we have inspected.
SOMA_SOURCE_FPS = 120.0

# SOMA BVH world frame -> OmniRetarget world frame.
#
#   BVH +X (character's left)    -> +Y (world left)
#   BVH +Y (up)                  -> +Z (up)
#   BVH +Z (character's forward) -> +X (robot forward)
#
# i.e. the cyclic permutation (x, y, z) -> (z, x, y). det = +1, so this is a proper
# rotation and preserves handedness -- left/right are NOT swapped.
#
# NOTE: this is deliberately *not* ``src.utils.transform_y_up_to_z_up``, which the LAFAN
# branch uses. That matrix is [[1,0,0],[0,0,1],[0,1,0]] with det = -1 -- a mirror, which
# suits LAFAN's left-handed source convention but would flip left/right on SOMA's
# explicitly right-handed data. It also leaves the character facing world +Y rather than
# +X, a 90 deg yaw away from the robot's rest facing.
#
# This matrix is bit-identical to ``FacingDirectionType.MAYA`` in
# ``NVIDIA/soma-retargeter/soma_retargeter/utils/space_conversion_utils.py``
# (= Ry(90 deg) @ Rz(90 deg)).
BVH_TO_WORLD = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Chain from Hips to the top of the skull. Used for a pose-independent stature estimate.
_STATURE_CHAIN = ["Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd"]

_ROT_CHANNELS = {"Xrotation", "Yrotation", "Zrotation"}
_POS_CHANNELS = {"Xposition", "Yposition", "Zposition"}

EulerConvention = Literal["intrinsic", "extrinsic"]


@dataclass
class BvhSkeleton:
    """Parsed BVH hierarchy and motion channels, in the file's native cm / Y-up frame."""

    names: list[str]
    parents: list[int]  # -1 for the root
    offsets: np.ndarray  # (J, 3) rest translation from parent, cm
    channels: list[list[str]]  # per-joint channel names, in file order
    motion: np.ndarray  # (T, total_channels)
    frame_time: float

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else float("nan")

    def index(self, name: str) -> int:
        return self.names.index(name)


def parse_bvh(path: str | Path) -> BvhSkeleton:
    """Parse a BVH file with per-joint channel counts.

    Handles ``End Site`` blocks, ``OFFSET`` lines with more than 3 tokens, and joints
    with differing channel counts and rotation orders.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []

    stack: list[int] = []
    # Index of the joint whose block we are currently inside, or None while inside an
    # End Site block (whose OFFSET must be discarded rather than treated as a joint).
    current: int | None = None
    in_end_site = False
    motion_start: int | None = None
    frame_time = 1.0 / SOMA_SOURCE_FPS
    declared_frames: int | None = None

    for lineno, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        head = tokens[0]

        if head in ("ROOT", "JOINT"):
            parent = stack[-1] if stack else -1
            names.append(tokens[1])
            parents.append(parent)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            current = len(names) - 1
            in_end_site = False
        elif line.startswith("End Site"):
            # Leaf tip: carries an OFFSET but no channels and is not a joint.
            in_end_site = True
        elif head == "{":
            if not in_end_site and current is not None:
                stack.append(current)
        elif head == "}":
            if in_end_site:
                in_end_site = False
            elif stack:
                stack.pop()
        elif head == "OFFSET":
            if not in_end_site and current is not None:
                # soma_zero_frame0.bvh writes 6 tokens (xyz translation + xyz rest
                # rotation). Take the translation and drop any tail.
                offsets[current] = [float(v) for v in tokens[1:4]]
        elif head == "CHANNELS":
            if not in_end_site and current is not None:
                count = int(tokens[1])
                names_ = tokens[2 : 2 + count]
                if len(names_) != count:
                    raise ValueError(f"{path}:{lineno + 1}: CHANNELS declares {count} but lists {len(names_)}")
                channels[current] = names_
        elif head == "MOTION":
            motion_start = lineno + 1
            break

    if motion_start is None:
        raise ValueError(f"{path}: no MOTION section found")
    if not names:
        raise ValueError(f"{path}: no joints found")

    # Parse the MOTION header, then the frame rows.
    cursor = motion_start
    while cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line:
            continue
        if line.startswith("Frames:"):
            declared_frames = int(float(line.split(":", 1)[1]))
        elif line.lower().startswith("frame time:"):
            frame_time = float(line.split(":", 1)[1])
            break
        else:
            # Tolerate an unexpected header ordering by rewinding.
            cursor -= 1
            break

    rows = []
    for line in lines[cursor:]:
        line = line.strip()
        if not line:
            continue
        rows.append(np.fromstring(line, sep=" ", dtype=np.float64))

    if not rows:
        raise ValueError(f"{path}: MOTION section contains no frames")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        bad = next(i for i, r in enumerate(rows) if len(r) != width)
        raise ValueError(f"{path}: frame {bad} has {len(rows[bad])} values, expected {width}")

    motion = np.stack(rows, axis=0)

    expected = sum(len(c) for c in channels)
    if width != expected:
        raise ValueError(f"{path}: motion rows have {width} values but hierarchy declares {expected} channels")

    # The declared frame count is authoritative; some writers append trailing rows.
    if declared_frames is not None and motion.shape[0] != declared_frames:
        motion = motion[:declared_frames]

    return BvhSkeleton(
        names=names,
        parents=parents,
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=channels,
        motion=motion,
        frame_time=frame_time,
    )


def _local_transforms(
    skel: BvhSkeleton, euler_convention: EulerConvention
) -> tuple[np.ndarray, np.ndarray]:
    """Split the motion matrix into per-joint local translations and rotations.

    Returns ``(local_pos (T, J, 3), local_quat (T, J, 4))`` with quaternions in
    scipy's xyzw order, in the file's native cm / Y-up frame.
    """
    num_frames, num_joints = skel.num_frames, skel.num_joints

    # Joints without position channels stay at their rest offset for every frame.
    local_pos = np.tile(skel.offsets[None, :, :], (num_frames, 1, 1))
    local_quat = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
    local_quat[..., 3] = 1.0  # identity, xyzw

    axis_to_col = {"X": 0, "Y": 1, "Z": 2}
    cursor = 0
    for j, joint_channels in enumerate(skel.channels):
        if not joint_channels:
            continue
        block = skel.motion[:, cursor : cursor + len(joint_channels)]
        cursor += len(joint_channels)

        rot_order = ""
        rot_values: list[np.ndarray] = []
        for col, channel in enumerate(joint_channels):
            if channel in _POS_CHANNELS:
                local_pos[:, j, axis_to_col[channel[0]]] = block[:, col]
            elif channel in _ROT_CHANNELS:
                rot_order += channel[0]
                rot_values.append(block[:, col])
            else:
                raise ValueError(f"Unsupported BVH channel {channel!r} on joint {skel.names[j]!r}")

        if rot_order:
            eulers = np.stack(rot_values, axis=-1)  # (T, n) in channel order
            # SOMA channel order is "Zrotation Yrotation Xrotation" for every joint.
            #
            # "extrinsic" (scipy uppercase) composes R = Rx @ Ry @ Rz -- rotations about
            # fixed world axes, applied in written channel order. This is the correct
            # reading for SOMA and the default.
            #
            # "intrinsic" (scipy lowercase) composes R = Rz @ Ry @ Rx instead.
            #
            # The two are NOT equivalent and bone-length checks cannot tell them apart, since
            # any rotation preserves bone length. They are separated by uprightness, stature
            # and ground contact; ``validate_soma_loader.py`` asserts all three. Measured on
            # BONES-SEED egypt_dance_R_003__A275 (599 frames):
            #
            #                     HeadEnd z    head-above-hips    cos(torso, +Z)
            #   extrinsic           1.725 m           100.0%           +0.994
            #   intrinsic           0.975 m            52.6%           -0.006
            #
            # Intrinsic collapses the figure -- HeadEnd sits at hip height and the torso has
            # no preferred vertical direction. Extrinsic agrees with GMR PR #169's loader.
            # Note that GMR's *comment* claims parity with soma-retargeter's Warp kernel
            # (which accumulates ``q *= axis_quat(ch, angle)`` over [z, y, x], i.e.
            # R = Rz @ Ry @ Rx); that claim does not survive this test, so the two upstream
            # implementations genuinely disagree and we follow the one the data supports.
            seq = rot_order.lower() if euler_convention == "intrinsic" else rot_order.upper()
            local_quat[:, j, :] = Rotation.from_euler(seq, eulers, degrees=True).as_quat()

    return local_pos, local_quat


def forward_kinematics(
    skel: BvhSkeleton, euler_convention: EulerConvention = "extrinsic"
) -> np.ndarray:
    """Compute global joint positions by FK, in the file's native cm / Y-up frame.

    Returns ``(T, J, 3)``.
    """
    local_pos, local_quat = _local_transforms(skel, euler_convention)
    num_frames, num_joints = skel.num_frames, skel.num_joints

    global_pos = np.zeros((num_frames, num_joints, 3), dtype=np.float64)
    global_rot: list[Rotation] = [Rotation.identity()] * num_joints

    for j, parent in enumerate(skel.parents):
        rot_j = Rotation.from_quat(local_quat[:, j, :])
        if parent < 0:
            global_pos[:, j, :] = local_pos[:, j, :]
            global_rot[j] = rot_j
        else:
            # A BVH child's local translation is expressed in its parent's frame.
            global_pos[:, j, :] = global_pos[:, parent, :] + global_rot[parent].apply(local_pos[:, j, :])
            global_rot[j] = global_rot[parent] * rot_j

    return global_pos


def to_world_frame(positions_cm: np.ndarray) -> np.ndarray:
    """Convert BVH-frame centimeters to OmniRetarget-frame meters."""
    return (positions_cm @ BVH_TO_WORLD.T) * CM_TO_M


def estimate_height(skel: BvhSkeleton) -> float:
    """Estimate the performer's stature in meters, independent of pose.

    ``Hips.offset`` is the rest translation of the pelvis above a ground plane on which
    the rest-pose feet sit (verified: min Y over all joints of ``soma_zero_frame0.bvh``
    is exactly 0.0), so its Y component is the hip height. Adding the summed bone lengths
    of the Hips -> HeadEnd chain gives stature without depending on the clip's pose --
    which matters because a clip may begin mid-crouch or mid-air.

    Falls back to the longest available prefix of the chain if the skull-tip joints are
    absent from a given file.
    """
    try:
        hips = skel.index("Hips")
    except ValueError as exc:
        raise ValueError("SOMA BVH is missing the 'Hips' joint") from exc

    hip_height_cm = float(skel.offsets[hips, 1])

    spine_cm = 0.0
    for name in _STATURE_CHAIN:
        if name not in skel.names:
            break
        spine_cm += float(np.linalg.norm(skel.offsets[skel.index(name)]))

    return (hip_height_cm + spine_cm) * CM_TO_M


def resample(positions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample a ``(T, J, 3)`` position array along time.

    Uses integer striding when the ratio is integral (the common 120 -> 30 and 120 -> 60
    cases), which avoids introducing interpolation error, and linear interpolation
    otherwise.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if abs(source_fps - target_fps) < 1e-9:
        return positions

    ratio = source_fps / target_fps
    if abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
        return positions[:: int(round(ratio))]

    num_frames = positions.shape[0]
    duration = (num_frames - 1) / source_fps
    num_out = int(np.floor(duration * target_fps)) + 1
    src_t = np.arange(num_frames) / source_fps
    dst_t = np.arange(num_out) / target_fps

    flat = positions.reshape(num_frames, -1)
    out = np.empty((num_out, flat.shape[1]), dtype=positions.dtype)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, flat[:, c])
    return out.reshape(num_out, *positions.shape[1:])


def load_soma_bvh(
    path: str | Path,
    joint_subset: list[str] | None = None,
    target_fps: float | None = 30.0,
    euler_convention: EulerConvention = "extrinsic",
) -> dict:
    """Load a SOMA BVH file into OmniRetarget's motion-data contract.

    Args:
        path: Path to the ``.bvh`` file.
        joint_subset: Joint names to keep, in output order. Defaults to
            ``SOMA_DEMO_JOINTS`` from the format registry.
        target_fps: Output frame rate, or None to keep the source rate.
        euler_convention: Rotation-channel composition; see ``_local_transforms``.

    Returns:
        Dict with ``global_joint_positions`` ``(T, J, 3)`` float32 in meters,
        Z-up/X-forward, plus ``height``, ``fps``, ``joint_names`` and provenance fields.
    """
    if joint_subset is None:
        from holosoma_retargeting.config_types.data_type import SOMA_DEMO_JOINTS

        joint_subset = SOMA_DEMO_JOINTS

    skel = parse_bvh(path)

    missing = [n for n in joint_subset if n not in skel.names]
    if missing:
        raise ValueError(f"{path}: SOMA BVH is missing expected joints: {missing}")

    positions_cm = forward_kinematics(skel, euler_convention=euler_convention)
    positions_m = to_world_frame(positions_cm)

    keep = [skel.index(n) for n in joint_subset]
    positions_m = positions_m[:, keep, :]

    source_fps = skel.fps
    out_fps = source_fps if target_fps is None else float(target_fps)
    positions_m = resample(positions_m, source_fps, out_fps)

    return {
        "global_joint_positions": positions_m.astype(np.float32),
        "height": np.float32(estimate_height(skel)),
        "fps": np.float32(out_fps),
        "joint_names": np.array(joint_subset),
        "source_fps": np.float32(source_fps),
        "source_file": str(path),
    }
