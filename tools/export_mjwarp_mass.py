"""Export CPU 2.3.7 packed-mass indexing for cross-engine validation only.

Run with the independent CPU reference Python. No simulation is advanced.
"""
import argparse
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['all'])
    args = parser.parse_args()
    if mujoco.__version__ != '2.3.7':
        raise ValueError('the independent mass layout requires MuJoCo 2.3.7')
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = load_catalog(args.build_root)
    tasks = [t for t in tasks if any(s in ('all', t['suite'], t['suite'] + ':' + str(t['task_id'])) for s in args.tasks)]
    for task in tasks:
        path = Path(task['directory']) / 'model.mjb'
        target = args.output / (Path(task['directory']).name + '.npz')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if target.is_file():
            old = np.load(target)
            if str(old['model_sha256']) == digest and str(old['mujoco']) == '2.3.7':
                continue
            raise ValueError('stale mass layout: ' + str(target))
        model = mujoco.MjModel.from_binary_path(str(path))
        basis = np.zeros(model.nM, np.float64)
        matrix = np.zeros((model.nv, model.nv), np.float64)
        indices = np.full(matrix.shape, -1, np.int32)
        for k in range(model.nM):
            basis.fill(0)
            basis[k] = 1
            mujoco.mj_fullM(model, matrix, basis)
            if np.any((matrix != 0) & (matrix != 1)):
                raise ValueError('unexpected packed mass coefficient')
            indices[matrix == 1] = k
        if set(indices[indices >= 0]) != set(range(model.nM)):
            raise ValueError('packed mass coverage mismatch')
        np.savez(target, indices=indices, nv=model.nv, nM=model.nM,
                 model_sha256=digest, mujoco='2.3.7')
    print(json.dumps(dict(tasks=len(tasks), output=str(args.output), physics_steps=0)))


if __name__ == '__main__':
    main()
