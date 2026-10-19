"""Cold, immutable topology for the single-substep smooth mass solve.

The descriptor is shared by worlds. It contains no state-dependent numbers.
Layout: nv, roots, subgroup width, bounds[roots+1], depth[nv], ancestors[nv].
Ancestor bits are relative to the component's first DOF. A width of zero
selects the existing general packed-tree solver, without a size restriction.
"""
import numpy as np


def build_forward_mass_plan(parent, madr, nm):
    parent = np.asarray(parent, dtype=np.int32)
    madr = np.asarray(madr, dtype=np.int32)
    nv = parent.size
    if parent.ndim != 1 or madr.shape != parent.shape or not nv:
        raise ValueError("smooth mass topology shape")
    bounds = np.r_[np.flatnonzero(parent == -1), nv].astype(np.int32)
    if bounds.size < 2 or bounds[0] != 0:
        raise ValueError("smooth mass topology has no first root")
    depths = np.zeros(nv, dtype=np.int32)
    masks = np.zeros(nv, dtype=np.uint32)
    contiguous = True
    for begin, end in zip(bounds[:-1], bounds[1:]):
        for i in range(int(begin), int(end)):
            p = int(parent[i])
            if p >= i or p < -1:
                raise ValueError("smooth mass topology is not parent ordered")
            if p >= 0:
                depths[i] = depths[p] + 1
                contiguous &= p >= begin
                if i - begin < 32 and p >= begin:
                    masks[i] = masks[p] | np.uint32(1 << (p - int(begin)))
    lengths = np.diff(np.r_[madr, nm])
    if madr[0] != 0 or not np.array_equal(lengths, depths + 1):
        raise ValueError("smooth packed mass rows disagree with the ancestor tree")
    longest = int(np.max(np.diff(bounds)))
    width = max(8, 1 << (longest - 1).bit_length()) if contiguous and longest <= 32 else 0
    return np.concatenate((np.asarray([nv, len(bounds) - 1, width], dtype=np.int32),
                           bounds, depths, masks.view(np.int32)))
