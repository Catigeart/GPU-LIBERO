"""Explicit host boundaries for fixture upload and diagnostic download.

The control path in ``controller.py`` never calls these helpers. They are for
the replay harness and timed upload/download measurements only.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

from .controller import PandaOSCWorkspace


INT32_FIELDS = {"active", "policy_step", "needs_update", "cache_valid", "error", "ranks", "quat_valid"}
FP32_FIELDS = {"quat_cast_wxyz"}
OUTPUT_FIELDS = (
    "mass_inv", "A_full", "A_pos", "A_ori", "lambda_full", "lambda_pos",
    "lambda_ori", "nullspace_matrix", "ranks", "position_error", "orientation_error",
    "nullspace_torque", "torques_raw", "torques_clipped", "ctrl",
    "goal_pos", "goal_rot", "gripper", "needs_update", "cache_valid", "error",
    "quat2mat_input", "quat_cast_wxyz", "quat2mat_result", "quat_valid",
)
PHYSICAL_FIELDS = (
    "phys_mass", "phys_jacobian", "phys_q", "phys_qvel", "phys_bias",
    "phys_ee_pos", "phys_ee_rot", "phys_ee_velocity",
)


def upload_arrays(workspace: PandaOSCWorkspace, values: Mapping[str, np.ndarray]) -> None:
    """Copy shape- and dtype-checked arrays to existing CUDA buffers."""
    for name, value in values.items():
        if name not in workspace.WIDTHS and name not in INT32_FIELDS and name not in FP32_FIELDS:
            raise KeyError(f"unknown OSC buffer: {name}")
        buffer = getattr(workspace, name)
        array = np.asarray(value)
        expected = np.dtype("int32" if name in INT32_FIELDS else "float32" if name in FP32_FIELDS else "float64")
        if array.dtype != expected or array.shape != buffer.shape:
            raise ValueError(
                f"{name}: expected {expected} {buffer.shape}, got {array.dtype} {array.shape}"
            )
        buffer.assign(np.ascontiguousarray(array))


def capture_native_manifest(workspace: PandaOSCWorkspace) -> dict[str, dict]:
    """Inspect actual resident Warp arrays before any result is downloaded."""
    manifest = {}
    names = (*workspace.WIDTHS, *INT32_FIELDS, *FP32_FIELDS)
    for name in names:
        buffer = getattr(workspace, name)
        if not buffer.device.is_cuda:
            raise ValueError(f"{name} is not on CUDA")
        expected = wp.int32 if name in INT32_FIELDS else wp.float32 if name in FP32_FIELDS else scalar.wp_real
        if buffer.dtype != expected:
            raise ValueError(f"{name} has native dtype {buffer.dtype}, expected {expected}")
        manifest[name] = {
            "dtype": "int32" if expected == wp.int32 else "float32" if expected == wp.float32 else "float64",
            "shape": list(buffer.shape),
            "device": str(buffer.device),
        }
    return manifest


def download_arrays(
    workspace: PandaOSCWorkspace, fields: Sequence[str] = OUTPUT_FIELDS,
) -> dict[str, np.ndarray]:
    """Download after execution; native buffer types are checked first."""
    manifest = capture_native_manifest(workspace)
    return {name: getattr(workspace, name).numpy() for name in fields if name in manifest}


FIXTURE_PHYSICAL = {
    "phys_mass": "cache_effective_mass_matrix",
    "phys_jacobian": "cache_effective_J_full",
    "phys_q": "physical_qpos",
    "phys_qvel": "physical_qvel",
    "phys_bias": "torque_compensation",
    "phys_ee_pos": "physical_ee_pos",
    "phys_ee_rot": "physical_ee_ori_mat",
    "phys_ee_velocity": None,
}


def _fixture_rows(fixture: Mapping[str, np.ndarray], name: str, call_ids: Sequence[int]) -> np.ndarray:
    rows = np.asarray(fixture[name])[np.asarray(call_ids, dtype=np.intp)]
    if rows.dtype != np.float64:
        raise ValueError(f"fixture {name} must be float64, got {rows.dtype}")
    return np.ascontiguousarray(rows.reshape((len(call_ids), -1)))


def initialize_from_fixture(
    workspace: PandaOSCWorkspace,
    fixture: Mapping[str, np.ndarray],
    first_call_ids: Sequence[int],
) -> None:
    """Seed only the original *pre-call* controller state of each world."""
    if len(first_call_ids) != workspace.nworld:
        raise ValueError("one initial call id per world is required")
    kp = _fixture_rows(fixture, "kp", first_call_ids)
    kd = _fixture_rows(fixture, "kd", first_call_ids)
    torque_low = _fixture_rows(fixture, "torque_low", first_call_ids)
    torque_high = _fixture_rows(fixture, "torque_high", first_call_ids)
    arm_range = np.stack((torque_low, torque_high), axis=-1)
    grip_range = np.asarray(fixture["gripper_ctrlrange"])[np.asarray(first_call_ids, dtype=np.intp)]
    if grip_range.dtype != np.float64 or grip_range.shape != (workspace.nworld, 2, 2):
        raise ValueError("gripper_ctrlrange must have shape [N,2,2] float64")
    ctrl_range = np.concatenate((arm_range, grip_range), axis=1)
    flags = np.asarray(fixture["new_update_before"])[np.asarray(first_call_ids, dtype=np.intp)]
    upload_arrays(workspace, {
        "goal_pos": _fixture_rows(fixture, "goal_pos_before", first_call_ids),
        "goal_rot": _fixture_rows(fixture, "goal_ori_before", first_call_ids),
        "gripper": _fixture_rows(fixture, "gripper_before", first_call_ids),
        "q0": _fixture_rows(fixture, "initial_joint", first_call_ids),
        "gains": np.ascontiguousarray(np.concatenate((kp, kd), axis=1)),
        "ctrl_range": np.ascontiguousarray(ctrl_range.reshape((workspace.nworld, 18))),
        "active": np.ones(workspace.nworld, dtype=np.int32),
        "needs_update": flags.astype(np.int32),
        "cache_valid": np.zeros(workspace.nworld, dtype=np.int32),
    })
    # If the first call reuses a pre-existing CPU cache, seed that cache. The
    # normal reset -> first-action fixture refreshes it on device instead.
    if not bool(np.all(flags)):
        pre_map = {
            "mass": "cache_pre_mass_matrix",
            "jacobian": "cache_pre_J_full",
            "q": "cache_pre_joint_pos",
            "qvel": "cache_pre_joint_vel",
            "ee_pos": "cache_pre_ee_pos",
            "ee_rot": "cache_pre_ee_ori_mat",
            "ee_velocity": None,
        }
        initial = {}
        for dst, src in pre_map.items():
            if src is not None:
                initial[dst] = _fixture_rows(fixture, src, first_call_ids)
        initial["ee_velocity"] = np.ascontiguousarray(np.concatenate((
            _fixture_rows(fixture, "cache_pre_ee_pos_vel", first_call_ids),
            _fixture_rows(fixture, "cache_pre_ee_ori_vel", first_call_ids),
        ), axis=1))
        initial["cache_valid"] = np.ones(workspace.nworld, dtype=np.int32)
        upload_arrays(workspace, initial)


def upload_fixture_step(
    workspace: PandaOSCWorkspace,
    fixture: Mapping[str, np.ndarray],
    call_ids: Sequence[int],
) -> None:
    """Upload one CPU physical snapshot/action per world, never CPU post-state."""
    if len(call_ids) != workspace.nworld:
        raise ValueError("one call id per world is required")
    values = {
        dst: _fixture_rows(fixture, src, call_ids)
        for dst, src in FIXTURE_PHYSICAL.items() if src is not None
    }
    values["phys_ee_velocity"] = np.ascontiguousarray(np.concatenate((
        _fixture_rows(fixture, "cache_effective_ee_pos_vel", call_ids),
        _fixture_rows(fixture, "cache_effective_ee_ori_vel", call_ids),
    ), axis=1))
    values["action"] = _fixture_rows(fixture, "action", call_ids)
    values["policy_step"] = np.asarray(fixture["policy_step"])[np.asarray(call_ids, dtype=np.intp)].astype(np.int32)
    upload_arrays(workspace, values)


def download_input_snapshot(workspace: PandaOSCWorkspace) -> dict[str, np.ndarray]:
    """Read exact uploaded device inputs under CPU fixture field names."""
    capture_native_manifest(workspace)
    velocity = workspace.phys_ee_velocity.numpy()
    return {
        "mass_matrix": workspace.phys_mass.numpy().reshape((-1, 7, 7)),
        "J_full": workspace.phys_jacobian.numpy().reshape((-1, 6, 7)),
        "joint_pos": workspace.phys_q.numpy(),
        "joint_vel": workspace.phys_qvel.numpy(),
        "torque_compensation": workspace.phys_bias.numpy(),
        "ee_pos": workspace.phys_ee_pos.numpy(),
        "ee_ori_mat": workspace.phys_ee_rot.numpy().reshape((-1, 3, 3)),
        "ee_pos_vel": velocity[:, :3],
        "ee_ori_vel": velocity[:, 3:],
        "initial_joint": workspace.q0.numpy(),
        "action": workspace.action.numpy(),
        "policy_step": workspace.policy_step.numpy(),
    }


def download_cache_snapshot(workspace: PandaOSCWorkspace) -> dict[str, np.ndarray]:
    """Read the actual post-selection controller cache from CUDA."""
    capture_native_manifest(workspace)
    velocity = workspace.ee_velocity.numpy()
    return {
        "cache_mass_matrix": workspace.mass.numpy().reshape((-1, 7, 7)),
        "cache_J_full": workspace.jacobian.numpy().reshape((-1, 6, 7)),
        "cache_joint_pos": workspace.q.numpy(),
        "cache_joint_vel": workspace.qvel.numpy(),
        "cache_ee_pos": workspace.ee_pos.numpy(),
        "cache_ee_ori_mat": workspace.ee_rot.numpy().reshape((-1, 3, 3)),
        "cache_ee_pos_vel": velocity[:, :3],
        "cache_ee_ori_vel": velocity[:, 3:],
    }
