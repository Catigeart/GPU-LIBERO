"""Enumerate and instantiate every shipped LIBERO model before GPU compilation."""
import gc
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10')


def task_catalog():
    from libero.libero.benchmark import task_maps
    from libero.libero import get_libero_path
    root = Path(get_libero_path('bddl_files'))
    return [dict(key=task.problem_folder + '/' + task.bddl_file,
                 suite=suite, task_id=index, name=task.name,
                 bddl=str(root / task.problem_folder / task.bddl_file))
            for suite in SUITES for index, task in enumerate(task_maps[suite].values())]


def _prepare_scene(job):
    task, root_string, seed = job
    import mujoco
    import numpy as np
    from libero.gpu.model import build_task_model
    from libero.gpu.host.adapters.environment.batch_env import _prepare_shared_fixtures
    from libero.gpu.host.runtime.core.portable_intake import prepare_native_intake
    from libero.gpu.host.runtime.core.product_paths import build_project
    from libero.libero import get_libero_path
    from libero.libero.benchmark.init_state_io import load_init_states
    root = Path(root_string)
    path = Path(task['bddl'])
    destination = root / 'scenes' / (task['suite'] + '__' + Path(task['key']).stem)
    destination.mkdir(parents=True, exist_ok=True)
    saved = destination / 'scene.json'
    bddl_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if saved.exists():
        row = json.loads(saved.read_text())
        if (row['bddl_sha256'] != bddl_sha or row['seed'] != seed or
                hashlib.sha256((destination / 'model.mjb').read_bytes()).hexdigest() != row['model_sha256']):
            raise ValueError('prepared scene differs: ' + task['key'])
        return row
    tick = time.monotonic()
    built = build_task_model(path, use_camera_obs=False, has_renderer=False,
                             has_offscreen_renderer=False, hard_reset=False)
    _prepare_shared_fixtures(built, seed)
    with build_project(root):
        binding = prepare_native_intake(built)
    model = built.native_model
    mujoco.mj_saveModel(model, str(destination / 'model.mjb'), None)
    (destination / 'scene.xml').write_text(built.xml)
    states_path = Path(get_libero_path('init_states')) / task['suite'] / (Path(task['key']).stem + '.pruned_init')
    initial = np.asarray(load_init_states(states_path), dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 1 + model.nq + model.nv or not np.isfinite(initial).all():
        raise ValueError('invalid official initial states: ' + task['key'])
    np.save(destination / 'initial_states.npy', initial, allow_pickle=False)
    row = dict(key=task['key'], suite=task['suite'], task_id=task['task_id'], name=task['name'],
        seed=seed, bddl_sha256=bddl_sha, model_sha256=built.metadata['model_sha256'],
        dimensions=built.metadata['dimensions'], options=built.metadata['options'],
        static_arrays_sha256=built.metadata['static_arrays_sha256'],
        constraint_package=str(binding.package), directory=str(destination),
        initial_states_shape=list(initial.shape), seconds=time.monotonic()-tick)
    if hashlib.sha256((destination / 'model.mjb').read_bytes()).hexdigest() != row['model_sha256']:
        raise ValueError('serialized scene identity differs: ' + task['key'])
    saved.write_text(json.dumps(row, indent=2, sort_keys=True)+'\n')
    del built, binding, model, initial
    gc.collect()
    return row


def prepare_catalog(root, *, jobs=1, seed=0, progress=None):
    root = Path(root).resolve()
    tasks = task_catalog()
    result = {}
    work = [(task, str(root), seed) for task in tasks]
    context = get_context('spawn')
    with context.Pool(jobs) as pool:
        for row in pool.imap_unordered(_prepare_scene, work):
            result[row['key']] = row
            if progress:
                progress(row, len(result), len(tasks))
    rows = [result[task['key']] for task in tasks]
    for row in rows:
        directory = Path(row['directory'])
        row['files'] = dict({'model.mjb': row['model_sha256']}, **{
            name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('scene.xml', 'initial_states.npy')})
    encoded = dict(schema='libero-all-task-instances-v1', seed=seed, count=len(rows), tasks=rows)
    (root / 'TASK_INSTANCES.json').write_text(json.dumps(encoded, indent=2, sort_keys=True)+'\n')
    return rows


def write_parameter_catalog(rows, destination):
    """Explicit portable template parameters, with all 130 task instances."""
    data = dict(schema='libero-template-parameters-v1', count=len(rows),
        newton_instances=[dict(nv=nv, nM=nm) for nv, nm in sorted({
            (row['dimensions']['nv'], row['dimensions']['nM']) for row in rows})],
        tasks=[{k: row[k] for k in ('key', 'suite', 'task_id', 'dimensions', 'options',
                                   'static_arrays_sha256', 'bddl_sha256')} for row in rows])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parameters', type=Path)
    args = parser.parse_args()
    def progress(row, done, total):
        print(json.dumps(dict(done=done, total=total, task=row['key'], nv=row['dimensions']['nv'])), flush=True)
    rows = prepare_catalog(args.build_root, jobs=args.jobs, seed=args.seed, progress=progress)
    if args.parameters:
        write_parameter_catalog(rows, args.parameters)
    print(json.dumps(dict(status='instantiated', count=len(rows))), flush=True)


if __name__ == '__main__':
    main()
