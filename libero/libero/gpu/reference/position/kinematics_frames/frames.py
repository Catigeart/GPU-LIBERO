"""FP64 body-inertial, geom and site frames from already computed body frames.

The arithmetic follows MuJoCo 2.3.7 engine_support.c:1438-1472,
engine_util_blas.c:157-166 and engine_util_spatial.c:60-70,144-181.
MuJoCo is Apache-2.0; source c9246e1f5006379d599e0bcddf159a8616d31441.
No body recurrence or forward dynamics is part of this module.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import hashlib
import json
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

MODEL_FIELDS = (
    "body_ipos", "body_iquat", "body_sameframe",
    "geom_bodyid", "geom_pos", "geom_quat", "geom_sameframe",
    "site_bodyid", "site_pos", "site_quat", "site_sameframe",
)
OUTPUT_FIELDS = ("xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat")
FLOAT_FIELDS = set(MODEL_FIELDS) - {"body_sameframe", "geom_bodyid", "geom_sameframe", "site_bodyid", "site_sameframe"}
STATUS_BAD_MASK = 1
TAIL_STATIC_SHA256 = {
    "real": "535d068c12b15eb62f2e6711f225dc781866f05235c40b70c608cf70e7690de5",
    "synthetic_test": "7868da0fc6d8db9eda80a3f3522040e2a76b580bf49618b913bf16a8c5c14094",
}


def _tail_static_sha256(fields):
    """Field/shape/dtype-labelled little-endian fingerprint at the device boundary."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        a = np.asarray(fields[name])
        a = np.ascontiguousarray(a.astype("<f8" if a.dtype.kind == "f" else "<i8", copy=False))
        digest.update(json.dumps([name, a.shape, a.dtype.str], separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(a.tobytes())
    return digest.hexdigest()


@wp.func
def _position(body_xpos: wp.array(dtype=scalar.wp_real), body_xmat: wp.array(dtype=scalar.wp_real),
              local_pos: wp.array(dtype=scalar.wp_real), out: wp.array(dtype=scalar.wp_real),
              body_offset: int, local_offset: int, out_offset: int):
    for row in range(3):
        mat = 9 * body_offset + 3 * row
        loc = 3 * local_offset
        v = body_xmat[mat] * local_pos[loc]
        v = v + body_xmat[mat + 1] * local_pos[loc + 1]
        v = v + body_xmat[mat + 2] * local_pos[loc + 2]
        out[3 * out_offset + row] = v + body_xpos[3 * body_offset + row]


@wp.func
def _matrix(body_xquat: wp.array(dtype=scalar.wp_real), local_quat: wp.array(dtype=scalar.wp_real),
            out: wp.array(dtype=scalar.wp_real), tmpquat: wp.array(dtype=scalar.wp_real),
            body_offset: int, local_offset: int,
            out_offset: int):
    a = 4 * body_offset
    b = 4 * local_offset
    # MuJoCo mju_mulQuat, wxyz; preserve its left-to-right evaluation.
    q0 = body_xquat[a] * local_quat[b]
    q0 = q0 - body_xquat[a + 1] * local_quat[b + 1]
    q0 = q0 - body_xquat[a + 2] * local_quat[b + 2]
    q0 = q0 - body_xquat[a + 3] * local_quat[b + 3]
    q1 = body_xquat[a] * local_quat[b + 1]
    q1 = q1 + body_xquat[a + 1] * local_quat[b]
    q1 = q1 + body_xquat[a + 2] * local_quat[b + 3]
    q1 = q1 - body_xquat[a + 3] * local_quat[b + 2]
    q2 = body_xquat[a] * local_quat[b + 2]
    q2 = q2 - body_xquat[a + 1] * local_quat[b + 3]
    q2 = q2 + body_xquat[a + 2] * local_quat[b]
    q2 = q2 + body_xquat[a + 3] * local_quat[b + 1]
    q3 = body_xquat[a] * local_quat[b + 3]
    q3 = q3 + body_xquat[a + 1] * local_quat[b + 2]
    q3 = q3 - body_xquat[a + 2] * local_quat[b + 1]
    q3 = q3 + body_xquat[a + 3] * local_quat[b]
    tmpquat[4 * out_offset] = q0
    tmpquat[4 * out_offset + 1] = q1
    tmpquat[4 * out_offset + 2] = q2
    tmpquat[4 * out_offset + 3] = q3
    m = 9 * out_offset
    if q0 == scalar.wp_real(1.0) and q1 == scalar.wp_real(0.0) and q2 == scalar.wp_real(0.0) and q3 == scalar.wp_real(0.0):
        out[m] = scalar.wp_real(1.0)
        out[m + 1] = scalar.wp_real(0.0)
        out[m + 2] = scalar.wp_real(0.0)
        out[m + 3] = scalar.wp_real(0.0)
        out[m + 4] = scalar.wp_real(1.0)
        out[m + 5] = scalar.wp_real(0.0)
        out[m + 6] = scalar.wp_real(0.0)
        out[m + 7] = scalar.wp_real(0.0)
        out[m + 8] = scalar.wp_real(1.0)
    else:
        q00 = q0 * q0
        q01 = q0 * q1
        q02 = q0 * q2
        q03 = q0 * q3
        q11 = q1 * q1
        q12 = q1 * q2
        q13 = q1 * q3
        q22 = q2 * q2
        q23 = q2 * q3
        q33 = q3 * q3
        out[m] = ((q00 + q11) - q22) - q33
        out[m + 4] = ((q00 - q11) + q22) - q33
        out[m + 8] = ((q00 - q11) - q22) + q33
        out[m + 1] = scalar.wp_real(2.0) * (q12 - q03)
        out[m + 2] = scalar.wp_real(2.0) * (q13 + q02)
        out[m + 3] = scalar.wp_real(2.0) * (q12 + q03)
        out[m + 5] = scalar.wp_real(2.0) * (q23 - q01)
        out[m + 6] = scalar.wp_real(2.0) * (q13 - q02)
        out[m + 7] = scalar.wp_real(2.0) * (q23 + q01)


@wp.func
def _copy_position(source: wp.array(dtype=scalar.wp_real), output: wp.array(dtype=scalar.wp_real),
                   src: int, dst: int):
    for j in range(3):
        output[3 * dst + j] = source[3 * src + j]


@wp.func
def _copy_matrix(source: wp.array(dtype=scalar.wp_real), output: wp.array(dtype=scalar.wp_real),
                 src: int, dst: int):
    for j in range(9):
        output[9 * dst + j] = source[9 * src + j]


@wp.func
def _zero_frame(pos: wp.array(dtype=scalar.wp_real), mat: wp.array(dtype=scalar.wp_real), dst: int):
    for j in range(3):
        pos[3 * dst + j] = scalar.wp_real(0.0)
    for j in range(9):
        mat[9 * dst + j] = scalar.wp_real(0.0)


@wp.func
def _zero_tmp(tmpquat: wp.array(dtype=scalar.wp_real), dst: int):
    for j in range(4):
        tmpquat[4 * dst + j] = scalar.wp_real(0.0)


@wp.kernel
def _status(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    w = wp.tid()
    status[w] = wp.where(active[w] == 0 or active[w] == 1, 0, STATUS_BAD_MASK)


@wp.kernel
def _body(active: wp.array(dtype=wp.int32), body_xpos: wp.array(dtype=scalar.wp_real),
          body_xquat: wp.array(dtype=scalar.wp_real), body_xmat: wp.array(dtype=scalar.wp_real),
          body_ipos: wp.array(dtype=scalar.wp_real), body_iquat: wp.array(dtype=scalar.wp_real),
          body_sameframe: wp.array(dtype=wp.int32), xipos: wp.array(dtype=scalar.wp_real),
          ximat: wp.array(dtype=scalar.wp_real), body_tmpquat: wp.array(dtype=scalar.wp_real), nbody: int):
    tid = wp.tid()
    w = tid // nbody
    i = tid - w * nbody
    if active[w] != 1:
        _zero_frame(xipos, ximat, tid)
        _zero_tmp(body_tmpquat, tid)
    elif i == 0:
        _zero_frame(xipos, ximat, tid)
        _zero_tmp(body_tmpquat, tid)
        ximat[9 * tid] = scalar.wp_real(1.0)
        ximat[9 * tid + 4] = scalar.wp_real(1.0)
        ximat[9 * tid + 8] = scalar.wp_real(1.0)
    elif body_sameframe[i] == 1:
        _copy_position(body_xpos, xipos, tid, tid)
        _copy_matrix(body_xmat, ximat, tid, tid)
        _zero_tmp(body_tmpquat, tid)
    else:
        _position(body_xpos, body_xmat, body_ipos, xipos, tid, i, tid)
        _matrix(body_xquat, body_iquat, ximat, body_tmpquat, tid, i, tid)


@wp.kernel
def _objects(active: wp.array(dtype=wp.int32), body_xpos: wp.array(dtype=scalar.wp_real),
             body_xquat: wp.array(dtype=scalar.wp_real), body_xmat: wp.array(dtype=scalar.wp_real),
             xipos: wp.array(dtype=scalar.wp_real), ximat: wp.array(dtype=scalar.wp_real),
             bodyid: wp.array(dtype=wp.int32), local_pos: wp.array(dtype=scalar.wp_real),
             local_quat: wp.array(dtype=scalar.wp_real), sameframe: wp.array(dtype=wp.int32),
             out_pos: wp.array(dtype=scalar.wp_real), out_mat: wp.array(dtype=scalar.wp_real),
             tmpquat: wp.array(dtype=scalar.wp_real),
             nbody: int, nobject: int):
    tid = wp.tid()
    w = tid // nobject
    i = tid - w * nobject
    if active[w] != 1:
        _zero_frame(out_pos, out_mat, tid)
        _zero_tmp(tmpquat, tid)
    else:
        b = w * nbody + bodyid[i]
        mode = sameframe[i]
        if mode == 0:
            _position(body_xpos, body_xmat, local_pos, out_pos, b, i, tid)
            _matrix(body_xquat, local_quat, out_mat, tmpquat, b, i, tid)
        elif mode == 1:
            _copy_position(body_xpos, out_pos, b, tid)
            _copy_matrix(body_xmat, out_mat, b, tid)
            _zero_tmp(tmpquat, tid)
        else:
            _copy_position(xipos, out_pos, b, tid)
            _copy_matrix(ximat, out_mat, b, tid)
            _zero_tmp(tmpquat, tid)


class FrameWorkspace:
    """One-time validated allocation/upload; ``launch`` only enqueues device work.

    Static arrays are shared by all worlds. Every dynamic field and output is a
    flat, contiguous scalar Warp array with world as its outermost dimension.
    The caller owns updates to the preallocated dynamic buffers and downloads.
    """

    def __init__(self, model: dict, nworld: int, *, device="cuda:0"):
        if nworld < 1:
            raise ValueError("nworld must be positive")
        missing = set(MODEL_FIELDS) - set(model)
        extra = set(model) - set(MODEL_FIELDS)
        if missing or extra:
            raise ValueError(f"static field set differs: missing={sorted(missing)}, extra={sorted(extra)}")
        prepared = {}
        count = None
        for name in MODEL_FIELDS:
            a = np.asarray(model[name])
            dtype = np.dtype("float64") if name in FLOAT_FIELDS else np.dtype("int32")
            if a.dtype != dtype or not a.flags.c_contiguous:
                raise TypeError(f"{name}: require contiguous {dtype}")
            if name == "body_ipos":
                count = a.shape[0]
            prepared[name] = a
        b = count
        g = prepared["geom_bodyid"].shape[0]
        s = prepared["site_bodyid"].shape[0]
        shapes = {"body_ipos": (b, 3), "body_iquat": (b, 4), "body_sameframe": (b,),
                  "geom_bodyid": (g,), "geom_pos": (g, 3), "geom_quat": (g, 4), "geom_sameframe": (g,),
                  "site_bodyid": (s,), "site_pos": (s, 3), "site_quat": (s, 4), "site_sameframe": (s,)}
        for name, shape in shapes.items():
            if prepared[name].shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {prepared[name].shape}")
        if b < 1 or g < 1 or s < 1:
            raise ValueError("nonempty body, geom, site arrays required")
        if not np.all(np.isin(prepared["body_sameframe"], (0, 1))):
            raise ValueError("body_sameframe must be 0/1")
        for kind in ("geom", "site"):
            if not np.all(np.isin(prepared[f"{kind}_sameframe"], (0, 1, 2))):
                raise ValueError(f"{kind}_sameframe must be 0/1/2")
            ids = prepared[f"{kind}_bodyid"]
            if np.any(ids < 0) or np.any(ids >= b):
                raise ValueError(f"{kind}_bodyid out of bounds")
        for name in FLOAT_FIELDS:
            if not np.all(np.isfinite(prepared[name])):
                raise ValueError(f"{name} nonfinite")
        self.tail_static_sha256 = _tail_static_sha256(prepared)
        if self.tail_static_sha256 not in TAIL_STATIC_SHA256.values():
            raise ValueError("unsupported static frame-tail model fingerprint")
        self.device = wp.get_device(device)
        self.nworld, self.nbody, self.ngeom, self.nsite = nworld, b, g, s
        self.static = {name: wp.array(a.reshape(-1), dtype=(scalar.wp_real if name in FLOAT_FIELDS else wp.int32), device=self.device)
                       for name, a in prepared.items()}
        self.body_xpos = wp.empty(nworld * b * 3, dtype=scalar.wp_real, device=self.device)
        self.body_xquat = wp.empty(nworld * b * 4, dtype=scalar.wp_real, device=self.device)
        self.body_xmat = wp.empty(nworld * b * 9, dtype=scalar.wp_real, device=self.device)
        self.active_world = wp.empty(nworld, dtype=wp.int32, device=self.device)
        self.xipos = wp.empty(nworld * b * 3, dtype=scalar.wp_real, device=self.device)
        self.ximat = wp.empty(nworld * b * 9, dtype=scalar.wp_real, device=self.device)
        self.geom_xpos = wp.empty(nworld * g * 3, dtype=scalar.wp_real, device=self.device)
        self.geom_xmat = wp.empty(nworld * g * 9, dtype=scalar.wp_real, device=self.device)
        self.site_xpos = wp.empty(nworld * s * 3, dtype=scalar.wp_real, device=self.device)
        self.site_xmat = wp.empty(nworld * s * 9, dtype=scalar.wp_real, device=self.device)
        self.body_tmpquat = wp.empty(nworld * b * 4, dtype=scalar.wp_real, device=self.device)
        self.geom_tmpquat = wp.empty(nworld * g * 4, dtype=scalar.wp_real, device=self.device)
        self.site_tmpquat = wp.empty(nworld * s * 4, dtype=scalar.wp_real, device=self.device)
        self.status = wp.empty(nworld, dtype=wp.int32, device=self.device)

    def upload(self, body_xpos, body_xquat, body_xmat, active_world, *, stream=None):
        """Boundary operation; never call in a resident launch or trace interval."""
        active = np.asarray(active_world)
        if active.shape != (self.nworld,) or active.dtype != np.int32 or not active.flags.c_contiguous:
            raise ValueError("active_world: require contiguous int32 [N]")
        for name, src, shape, dtype in (
            ("body_xpos", body_xpos, (self.nworld, self.nbody, 3), np.float64),
            ("body_xquat", body_xquat, (self.nworld, self.nbody, 4), np.float64),
            ("body_xmat", body_xmat, (self.nworld, self.nbody, 9), np.float64),
            ("active_world", active_world, (self.nworld,), np.int32),
        ):
            arr = np.asarray(src)
            if arr.shape != shape or arr.dtype != dtype or not arr.flags.c_contiguous:
                raise ValueError(f"{name}: require contiguous {dtype} {shape}")
            if dtype == np.float64 and not np.all(np.isfinite(arr[active == 1])):
                raise ValueError(f"{name}: nonfinite active input")
            wp.copy(getattr(self, name), wp.array(arr.reshape(-1), dtype=(scalar.wp_real if dtype == np.float64 else wp.int32), device=self.device), stream=stream)

    def launch(self, *, stream=None):
        wp.launch(_status, dim=self.nworld, inputs=[self.active_world, self.status], device=self.device, stream=stream)
        wp.launch(_body, dim=self.nworld * self.nbody,
                  inputs=[self.active_world, self.body_xpos, self.body_xquat, self.body_xmat,
                          self.static["body_ipos"], self.static["body_iquat"], self.static["body_sameframe"],
                          self.xipos, self.ximat, self.body_tmpquat, self.nbody], device=self.device, stream=stream)
        for kind, count in (("geom", self.ngeom), ("site", self.nsite)):
            wp.launch(_objects, dim=self.nworld * count,
                      inputs=[self.active_world, self.body_xpos, self.body_xquat, self.body_xmat,
                              self.xipos, self.ximat, self.static[f"{kind}_bodyid"],
                              self.static[f"{kind}_pos"], self.static[f"{kind}_quat"],
                              self.static[f"{kind}_sameframe"], getattr(self, f"{kind}_xpos"),
                              getattr(self, f"{kind}_xmat"), getattr(self, f"{kind}_tmpquat"),
                              self.nbody, count], device=self.device, stream=stream)

    def download(self):
        return {name: getattr(self, name).numpy().reshape(
            self.nworld, {"xipos": self.nbody, "ximat": self.nbody,
                          "geom_xpos": self.ngeom, "geom_xmat": self.ngeom,
                          "site_xpos": self.nsite, "site_xmat": self.nsite}[name],
            9 if name.endswith("xmat") or name == "ximat" else 3)
            for name in OUTPUT_FIELDS} | {"status": self.status.numpy()}

    def download_stages(self):
        return {kind + "_tmpquat": getattr(self, kind + "_tmpquat").numpy().reshape(
            self.nworld, count, 4) for kind, count in
            (("body", self.nbody), ("geom", self.ngeom), ("site", self.nsite))}
