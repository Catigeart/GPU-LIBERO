"""Stable per-pair contact deduplication on CUDA.

This implements only the position equality and stable compression from
MuJoCo 2.3.7 ``src/engine/engine_collision_driver.c:954-988`` (Apache-2.0),
whose source file SHA-256 is
``cd120732b9eea5c26c4db5dbfd8def2435c0a4c2e521dc1715264dd2b1258cd8``.
The original C fragment is preserved in the read-only 2B0 bridge; no collision
generation, contact-frame calculation, or global pair ordering occurs here.

All arrays are FP64 or int32 and world-major. ``launch`` reads only resident
device inputs and writes only resident outputs; upload/download are boundaries.
Status: 0 success/inactive, 1 pair_count outside [0,P], 2 raw_count outside
[0,50], 3 nonfinite valid raw field, 4 active_world outside {0,1}. Invalid
pairs return the standard cleared outputs and never read their raw payload.
"""

from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

MAX_CONTACTS = 50  # MuJoCo 2.3.7 mjMAXCONPAIR
FIELDS = 10


@wp.kernel(enable_backward=False)
def dedup_contacts(
    raw: wp.array4d(dtype=scalar.wp_real),
    raw_count: wp.array2d(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    active_world: wp.array(dtype=wp.int32),
    keep_mask: wp.array3d(dtype=wp.int32),
    keep_indices: wp.array3d(dtype=wp.int32),
    compact: wp.array4d(dtype=scalar.wp_real),
    compact_count: wp.array2d(dtype=wp.int32),
    status: wp.array2d(dtype=wp.int32),
    npair: int,
):
    pair_slot = wp.tid()
    world = pair_slot // npair
    pair = pair_slot - world * npair

    # A fresh result is written on every call, including invalid and inactive
    # slots. No stale result can survive a lowered count or deactivation.
    status[world, pair] = 0
    compact_count[world, pair] = 0
    for i in range(MAX_CONTACTS):
        keep_mask[world, pair, i] = 0
        keep_indices[world, pair, i] = -1
        for k in range(FIELDS):
            compact[world, pair, i, k] = scalar.wp_real(0.0)

    active = active_world[world]
    if active == 0:
        return
    if active != 1:
        status[world, pair] = 4
        return
    valid_pairs = pair_count[world]
    if valid_pairs < 0 or valid_pairs > npair:
        status[world, pair] = 1
        return
    if pair >= valid_pairs:
        return
    count = raw_count[world, pair]
    if count < 0 or count > MAX_CONTACTS:
        status[world, pair] = 2
        return

    # Only valid rows are inspected. A NaN in padding is allowed, while an
    # invalid valid field is reported before doing equality comparisons.
    for i in range(count):
        for k in range(FIELDS):
            if not wp.isfinite(raw[world, pair, i, k]):
                status[world, pair] = 3
                return

    out = wp.int32(0)
    for i in range(count):
        duplicate = wp.bool(False)
        for j in range(i + 1, count):
            if (raw[world, pair, i, 1] == raw[world, pair, j, 1]
                    and raw[world, pair, i, 2] == raw[world, pair, j, 2]
                    and raw[world, pair, i, 3] == raw[world, pair, j, 3]):
                duplicate = True
                break
        if not duplicate:
            keep_mask[world, pair, i] = 1
            keep_indices[world, pair, out] = i
            for k in range(FIELDS):
                compact[world, pair, out, k] = raw[world, pair, i, k]
            out += 1
    compact_count[world, pair] = out


class ContactDedupWorkspace:
    """Preallocated device buffers and explicit host transfer boundaries.

    ``raw``: [N,P,50,10] FP64; ``raw_count``: [N,P], ``pair_count`` and
    ``active_world``: [N] int32. Output arrays have the contract's shapes.
    Inputs and outputs are allocated separately and never alias. ``launch``
    performs no host data access, allocation, or synchronization.
    """

    def __init__(self, nworld: int, npair: int, device: str = "cuda:0") -> None:
        if not isinstance(nworld, int) or not isinstance(npair, int) or nworld < 1 or npair < 1:
            raise ValueError("nworld and npair must be positive integers")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("ContactDedupWorkspace requires CUDA")
        self.nworld = nworld
        self.npair = npair
        n, p = nworld, npair
        self.raw = wp.empty((n, p, MAX_CONTACTS, FIELDS), dtype=scalar.wp_real, device=self.device)
        self.raw_count = wp.empty((n, p), dtype=wp.int32, device=self.device)
        self.pair_count = wp.empty((n,), dtype=wp.int32, device=self.device)
        self.active_world = wp.empty((n,), dtype=wp.int32, device=self.device)
        self.keep_mask = wp.empty((n, p, MAX_CONTACTS), dtype=wp.int32, device=self.device)
        self.keep_indices = wp.empty((n, p, MAX_CONTACTS), dtype=wp.int32, device=self.device)
        self.compact = wp.empty((n, p, MAX_CONTACTS, FIELDS), dtype=scalar.wp_real, device=self.device)
        self.compact_count = wp.empty((n, p), dtype=wp.int32, device=self.device)
        self.status = wp.empty((n, p), dtype=wp.int32, device=self.device)

    def upload(self, raw: np.ndarray, raw_count: np.ndarray,
               pair_count: np.ndarray, active_world: np.ndarray) -> None:
        """Shape/type-check and upload inputs outside the device launch path."""
        values = (raw, raw_count, pair_count, active_world)
        names = ("raw", "raw_count", "pair_count", "active_world")
        shapes = ((self.nworld, self.npair, MAX_CONTACTS, FIELDS),
                  (self.nworld, self.npair), (self.nworld,), (self.nworld,))
        dtypes = (np.dtype("float64"), np.dtype("int32"), np.dtype("int32"), np.dtype("int32"))
        for name, value, shape, dtype in zip(names, values, shapes, dtypes):
            arr = np.asarray(value)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(f"{name}: expected {shape} {dtype}, got {arr.shape} {arr.dtype}")
        for name, value in zip(names, values):
            getattr(self, name).assign(np.ascontiguousarray(value))

    def launch(self, *, stream=None) -> None:
        """Enqueue the production CUDA kernel, including under Graph capture."""
        wp.launch(dedup_contacts, dim=self.nworld * self.npair,
                  inputs=[self.raw, self.raw_count, self.pair_count,
                          self.active_world, self.keep_mask, self.keep_indices,
                          self.compact, self.compact_count, self.status, self.npair],
                  device=self.device, stream=stream)

    def download(self) -> dict[str, np.ndarray]:
        """Download all outputs at the explicit observation boundary."""
        return {name: getattr(self, name).numpy() for name in
                ("keep_mask", "keep_indices", "compact", "compact_count", "status")}
