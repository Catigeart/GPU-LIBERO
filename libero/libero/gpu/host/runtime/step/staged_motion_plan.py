"""Immutable branch prefixes for cooperative native FP64 velocity/RNE.

Layout: branches, offsets[branches+1], bodies[offsets[-1]], owner[nbody].
Each root-to-leaf path is computed privately; only the first path containing
a body publishes it. The descriptor has no state-dependent numerical cache.
"""
import numpy as np


def build_motion_plan(parent):
    parent = np.asarray(parent, dtype=np.int32)
    if parent.ndim != 1 or not parent.size or parent[0] != 0:
        raise ValueError('motion topology world root')
    children = np.zeros(parent.size, np.int32)
    for body in range(1, parent.size):
        if not 0 <= parent[body] < body:
            raise ValueError('motion topology is not parent ordered')
        children[parent[body]] += 1
    paths, offsets = [], [0]
    owner = np.full(parent.size, -1, np.int32)
    for leaf in np.flatnonzero(children[1:] == 0) + 1:
        path = []
        body = int(leaf)
        while body:
            path.append(body)
            body = int(parent[body])
        path.reverse()
        branch = len(offsets) - 1
        for body in path:
            if owner[body] == -1:
                owner[body] = branch
        paths.extend(path)
        offsets.append(len(paths))
    return np.concatenate((np.asarray([len(offsets)-1],np.int32),
                           np.asarray(offsets,np.int32),np.asarray(paths,np.int32),owner))


def motion_threads(plan, nbody):
    branches = int(plan[0])
    path_cells = int(plan[1 + branches])
    # Avoid excessive duplicated prefixes on bushy trees with deep shared
    # trunks. The original level traversal remains available without a limit.
    if not branches or path_cells > 4 * (nbody - 1):
        return 0
    return 128 if branches <= 16 else 256
