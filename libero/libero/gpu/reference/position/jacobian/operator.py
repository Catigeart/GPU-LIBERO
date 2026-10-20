"""Given prepared cdof/subtree_com, evaluate mj_jac and mj_jacSite on CUDA.

MuJoCo 2.3.7 reference: src/engine/engine_support.c:208-257, 312-314,
Apache-2.0; whole source file SHA256
e532f0275e7e8c46e95271e0033e6b6319dc3c832fe52521c77efb3966947a77.
Only the simple Jacobian scalar expression and topology traversal are used.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

import hashlib
import json
from typing import Mapping

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

STATUS_OK = 0
STATUS_COUNT = 1
STATUS_BODY = 2
STATUS_SITE = 3
STATUS_ACTIVE = 4
STATUS_NUMERIC = 5

STATIC_FIELDS = ("body_rootid", "body_parentid", "body_dofnum", "body_dofadr",
                 "dof_parentid", "site_bodyid")


def device_static_sha(fields: Mapping[str, np.ndarray]) -> str:
    """Canonical topology hash shared with the independent derivation tool."""
    digest = hashlib.sha256()
    for key in sorted(("nv", "nbody", "nsite", *STATIC_FIELDS)):
        value = np.asarray(fields[key], dtype="<i4")
        meta = [key, list(value.shape), "<i4"]
        digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def model_sha256(fields: Mapping[str, np.ndarray], *, model_mjb_sha256: str) -> str:
    """Bind the topology hash to the full compiled model's MJB SHA256."""
    payload = {"mjb_sha256": model_mjb_sha256,
               "device_static_sha256": device_static_sha(fields)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class StaticModel:
    """Validate and freeze all topology used by the original ancestor walk."""

    def __init__(self, fields: Mapping[str, np.ndarray], *, model_mjb_sha256: str):
        if len(model_mjb_sha256) != 64:
            raise ValueError("Full compiled MJB SHA256 required")
        self.nv = int(np.asarray(fields["nv"]))
        self.nbody = int(np.asarray(fields["nbody"]))
        self.nsite = int(np.asarray(fields["nsite"]))
        if self.nv <= 0 or self.nbody <= 0 or self.nsite <= 0:
            raise ValueError("Invalid model dimensions")
        lengths = {"body_rootid": self.nbody, "body_parentid": self.nbody,
                   "body_dofnum": self.nbody, "body_dofadr": self.nbody,
                   "dof_parentid": self.nv, "site_bodyid": self.nsite}
        self.fields = {"nv": np.asarray(self.nv, dtype="<i4"),
                       "nbody": np.asarray(self.nbody, dtype="<i4"),
                       "nsite": np.asarray(self.nsite, dtype="<i4")}
        for name, count in lengths.items():
            raw = np.asarray(fields[name])
            if raw.dtype.kind not in "iu" or raw.shape != (count,) or (
                    raw.size and (raw.min() < -(2**31) or raw.max() >= 2**31)):
                raise ValueError(f"Invalid static field {name}: {raw.shape}, {raw.dtype}")
            self.fields[name] = np.ascontiguousarray(raw, dtype="<i4")
        parent = self.fields["body_parentid"]
        root = self.fields["body_rootid"]
        num = self.fields["body_dofnum"]
        adr = self.fields["body_dofadr"]
        dp = self.fields["dof_parentid"]
        site = self.fields["site_bodyid"]
        if parent[0] != 0 or root[0] != 0 or num[0] != 0:
            raise ValueError("World body topology differs")
        for body in range(1, self.nbody):
            seen = set()
            cursor = body
            last = body
            while cursor:
                if cursor in seen or cursor < 0 or cursor >= self.nbody:
                    raise ValueError("Body parent chain invalid or cyclic")
                seen.add(cursor)
                last = cursor
                cursor = int(parent[cursor])
            if root[body] != last:
                raise ValueError(f"Body root differs at {body}: {root[body]} != {last}")
            if num[body] < 0 or (num[body] and (adr[body] < 0 or
               adr[body] + num[body] > self.nv)):
                raise ValueError(f"Body dof block invalid at {body}")
        if np.any((site < 0) | (site >= self.nbody)):
            raise ValueError("Site body ID out of range")
        if np.any((dp < -1) | (dp >= self.nv)):
            raise ValueError("Dof parent ID out of range")
        for dof in range(self.nv):
            seen = set()
            cursor = dof
            while cursor >= 0:
                if cursor in seen:
                    raise ValueError("Dof ancestor chain is cyclic")
                seen.add(cursor)
                cursor = int(dp[cursor])
        mask = np.zeros((self.nbody, self.nv), dtype=np.int32)
        # Exactly reproduce mj_jac: skip fixed bodies and start with the last
        # dof of the first movable body, then follow dof_parentid to -1.
        for body in range(self.nbody):
            cursor = body
            while cursor and num[cursor] == 0:
                cursor = int(parent[cursor])
            if not cursor:
                continue
            dof = int(adr[cursor] + num[cursor] - 1)
            while dof >= 0:
                mask[body, dof] = 1
                dof = int(dp[dof])
        self.ancestor_mask = mask
        self.model_mjb_sha256 = model_mjb_sha256
        self.device_static_sha256 = device_static_sha(self.fields)
        self.sha256 = model_sha256(self.fields, model_mjb_sha256=model_mjb_sha256)


@wp.kernel(enable_backward=False)
def _jacobian(
    cdof: wp.array3d(dtype=scalar.wp_real),
    subtree_com: wp.array3d(dtype=scalar.wp_real),
    points: wp.array3d(dtype=scalar.wp_real),
    body_ids: wp.array2d(dtype=wp.int32),
    site_xpos: wp.array3d(dtype=scalar.wp_real),
    site_ids: wp.array2d(dtype=wp.int32),
    query_count: wp.array(dtype=wp.int32),
    active_world: wp.array(dtype=wp.int32),
    body_rootid: wp.array(dtype=wp.int32),
    site_bodyid: wp.array(dtype=wp.int32),
    ancestor_mask: wp.array2d(dtype=wp.int32),
    jacp: wp.array4d(dtype=scalar.wp_real),
    jacr: wp.array4d(dtype=scalar.wp_real),
    status: wp.array2d(dtype=wp.int32),
    nv: int, nbody: int, nsite: int, capacity: int, site_mode: int,
):
    world, query = wp.tid()
    # Every launch overwrites dirty outputs before inspecting dynamic IDs.
    for row in range(3):
        for dof in range(nv):
            jacp[world, query, row, dof] = scalar.wp_real(0.0)
            jacr[world, query, row, dof] = scalar.wp_real(0.0)
    status[world, query] = STATUS_OK
    active = active_world[world]
    if active == 0:
        return
    if active != 1:
        status[world, query] = STATUS_ACTIVE
        return
    count = query_count[world]
    if count < 0 or count > capacity:
        status[world, query] = STATUS_COUNT
        return
    if query >= count:
        return

    body = int(0)
    px = scalar.wp_real(0.0)
    py = scalar.wp_real(0.0)
    pz = scalar.wp_real(0.0)
    if site_mode == 1:
        site = site_ids[world, query]
        if site < 0 or site >= nsite:
            status[world, query] = STATUS_SITE
            return
        body = site_bodyid[site]
        px = site_xpos[world, site, 0]
        py = site_xpos[world, site, 1]
        pz = site_xpos[world, site, 2]
    else:
        body = body_ids[world, query]
        if body < 0 or body >= nbody:
            status[world, query] = STATUS_BODY
            return
        px = points[world, query, 0]
        py = points[world, query, 1]
        pz = points[world, query, 2]
    root = body_rootid[body]
    ox = px - subtree_com[world, root, 0]
    oy = py - subtree_com[world, root, 1]
    oz = pz - subtree_com[world, root, 2]
    invalid = int(0)
    if not wp.isfinite(ox) or not wp.isfinite(oy) or not wp.isfinite(oz):
        invalid = 1
    if invalid == 0:
        for dof in range(nv):
            if ancestor_mask[body, dof] == 1:
                ax = cdof[world, dof, 0]
                ay = cdof[world, dof, 1]
                az = cdof[world, dof, 2]
                lx = cdof[world, dof, 3]
                ly = cdof[world, dof, 4]
                lz = cdof[world, dof, 5]
                if (not wp.isfinite(ax) or not wp.isfinite(ay) or
                    not wp.isfinite(az) or not wp.isfinite(lx) or
                    not wp.isfinite(ly) or not wp.isfinite(lz)):
                    invalid = 1
                    break
                cx = ay * oz - az * oy
                cy = az * ox - ax * oz
                cz = ax * oy - ay * ox
                vx = lx + cx
                vy = ly + cy
                vz = lz + cz
                if (not wp.isfinite(vx) or not wp.isfinite(vy) or
                    not wp.isfinite(vz)):
                    invalid = 1
                    break
                jacr[world, query, 0, dof] = ax
                jacr[world, query, 1, dof] = ay
                jacr[world, query, 2, dof] = az
                jacp[world, query, 0, dof] = vx
                jacp[world, query, 1, dof] = vy
                jacp[world, query, 2, dof] = vz
    if invalid == 1:
        status[world, query] = STATUS_NUMERIC
        for row in range(3):
            for dof in range(nv):
                jacp[world, query, row, dof] = scalar.wp_real(0.0)
                jacr[world, query, row, dof] = scalar.wp_real(0.0)


class JacobianWorkspace:
    """Preallocated arrays; `launch` performs one pure device dispatch.

    Inputs are disjoint from outputs. `upload`/`download` are explicit testing
    and application boundaries, never called by `launch` or graph replay.
    """

    def __init__(self, static: StaticModel, nworld: int, capacity: int,
                 *, device: str = "cuda:0"):
        if nworld <= 0 or capacity <= 0:
            raise ValueError("N and Q must be positive")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("JacobianWorkspace requires CUDA")
        self.static = static
        self.nworld = nworld
        self.capacity = capacity
        self.nv = static.nv
        self.nbody = static.nbody
        self.nsite = static.nsite
        self.body_rootid = wp.array(static.fields["body_rootid"], dtype=wp.int32,
                                    device=self.device)
        self.site_bodyid = wp.array(static.fields["site_bodyid"], dtype=wp.int32,
                                    device=self.device)
        self.ancestor_mask = wp.array(static.ancestor_mask, dtype=wp.int32,
                                      device=self.device)
        F, I = scalar.wp_real, wp.int32
        N, Q = nworld, capacity
        self.cdof = wp.zeros((N, self.nv, 6), dtype=F, device=self.device)
        self.subtree_com = wp.zeros((N, self.nbody, 3), dtype=F, device=self.device)
        self.points = wp.zeros((N, Q, 3), dtype=F, device=self.device)
        self.body_ids = wp.zeros((N, Q), dtype=I, device=self.device)
        self.site_xpos = wp.zeros((N, self.nsite, 3), dtype=F, device=self.device)
        self.site_ids = wp.zeros((N, Q), dtype=I, device=self.device)
        self.query_count = wp.zeros(N, dtype=I, device=self.device)
        self.active_world = wp.zeros(N, dtype=I, device=self.device)
        self.jacp = wp.zeros((N, Q, 3, self.nv), dtype=F, device=self.device)
        self.jacr = wp.zeros((N, Q, 3, self.nv), dtype=F, device=self.device)
        self.status = wp.zeros((N, Q), dtype=I, device=self.device)

    def upload(self, **arrays: np.ndarray) -> None:
        """Shape/dtype checked host upload, only outside timed launch loops."""
        allowed = ("cdof", "subtree_com", "points", "body_ids", "site_xpos",
                   "site_ids", "query_count", "active_world")
        for name, value in arrays.items():
            if name not in allowed:
                raise KeyError(name)
            target = getattr(self, name)
            raw = np.asarray(value)
            expected = np.dtype("float64" if name in ("cdof", "subtree_com", "points",
                                                    "site_xpos") else "int32")
            if raw.dtype != expected or raw.shape != target.shape:
                raise ValueError(f"{name}: expected {target.shape} {expected}, got {raw.shape} {raw.dtype}")
            target.assign(np.ascontiguousarray(raw))

    def launch(self, *, site: bool = False, stream=None) -> None:
        """Enqueue exactly one kernel; Graph capturable, no host data access."""
        wp.launch(_jacobian, dim=(self.nworld, self.capacity),
                  inputs=[self.cdof, self.subtree_com, self.points, self.body_ids,
                          self.site_xpos, self.site_ids, self.query_count,
                          self.active_world, self.body_rootid, self.site_bodyid,
                          self.ancestor_mask, self.jacp, self.jacr, self.status,
                          self.nv, self.nbody, self.nsite, self.capacity,
                          int(site)], device=self.device, stream=stream)

    def download(self) -> dict[str, np.ndarray]:
        """Copy output after explicit completion, outside steady-state loops."""
        return {"jacp": self.jacp.numpy(), "jacr": self.jacr.numpy(),
                "status": self.status.numpy()}
