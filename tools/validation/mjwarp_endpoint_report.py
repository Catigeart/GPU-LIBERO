"""CPU-only, explicitly scoped analysis of retained MJWarp comparison outputs.

This reporter never changes the original strict statuses or runs physics. A
successful command means that the evidence was readable and internally bound;
the different numerical conclusions remain separate fields in its output.
"""
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog

import numpy as np

from .common import select_tasks, save_json
from .mjwarp_policy import compare_replays
from .mjwarp_trajectory_reference import variation_checks


FIELDS = ('qpos', 'qvel', 'xpos', 'xquat', 'ctrl', 'time')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(parser):
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['all'])


def physics_binding(signature):
    """Permit validation-only reassessment, never unlike executed physics."""
    result = {key: signature[key] for key in
              ('sources', 'forward_helper', 'scene_helper', 'repeat_helper', 'protocol')}
    result['actions_helper'] = signature.get('actions_helper')
    return result


def robot_indices(scene):
    qids, vids = [], []
    for body, kind, qa, da in zip(scene['jnt_bodyid'], scene['jnt_type'],
                                  scene['jnt_qposadr'], scene['jnt_dofadr']):
        if not str(scene['body_names'][body]).startswith(('robot0_', 'gripper0_')):
            continue
        qids.extend(range(int(qa), int(qa) + (7 if kind == 0 else 4 if kind == 1 else 1)))
        vids.extend(range(int(da), int(da) + (6 if kind == 0 else 3 if kind == 1 else 1)))
    if len(qids) != 9 or len(vids) != 9 or len(set(qids)) != 9 or len(set(vids)) != 9:
        raise ValueError('Expected the admitted Panda nine-DOF joint mapping')
    return qids, vids


def compare_scopes(candidate, first, second, scene):
    qids, vids = robot_indices(scene)

    def state(values):
        # ctrl contains closed-loop OSC responses to equal high-level actions,
        # not a recording of identical actuator commands (which must be exact).
        return dict(qpos=values['qpos'], qvel=values['qvel'],
                    controller_output=values['ctrl'], time=values['time'])

    def boundary(values):
        return dict(robot_position=values['qpos'][:, qids],
                    robot_velocity=values['qvel'][:, vids],
                    body_position=values['xpos'], body_quaternion=values['xquat'],
                    controller_output=values['ctrl'], time=values['time'])

    result = dict(robot_qpos_indices=qids, robot_dof_indices=vids)
    for name, project in (('full_state', state), ('task_boundary', boundary)):
        checks = compare_replays(*(project(v) for v in (candidate, first, second)))
        result[name] = dict(passed=all(c['passed'] for c in checks), checks=checks)
    return result


def analyze_task(folder, task, scene, scene_receipt, expected_physics=None):
    folder = Path(folder)
    summary_path, endpoint = folder / 'summary.json', folder / 'endpoints.npz'
    summary = json.loads(summary_path.read_text())
    execution = summary.get('execution', {})
    correctness_complete = (execution.get('kind') == 'correctness_only'
                            and execution.get('controls_per_mode') == 25
                            and set(execution.get('completed_modes', [])) == {'cuda', 'upstream'})
    if summary.get('stage', 'trajectory') != 'trajectory' or not (summary.get('timings') or correctness_complete):
        raise ValueError('Incomplete trajectory report: ' + str(folder))
    signature = summary['signature']
    if expected_physics is not None and physics_binding(signature) != expected_physics:
        raise ValueError('Executed physics/reference sources differ: ' + str(folder))
    identity = signature['identity']
    if identity['task'] != task['key'] or identity['worlds'] != 2:
        raise ValueError('Task/world binding differs: ' + str(folder))
    if identity['fixed_scene_inputs_sha256'] != scene_receipt['data_sha256']:
        raise ValueError('Original scene input binding differs: ' + str(folder))
    expected_endpoint = summary.get('endpoints_sha256') or summary.get('assessment', {}).get('endpoints_sha256')
    if not expected_endpoint or digest(endpoint) != expected_endpoint:
        raise ValueError('Missing or changed endpoint receipt: ' + str(folder))
    with np.load(endpoint, allow_pickle=False) as archive:
        arrays = {mode: {key: archive[mode + '.' + key].copy() for key in FIELDS}
                  for mode in ('cuda', 'upstream')}
        all_endpoints = {key: archive[key].copy() for key in archive.files}
    diagnostic = summary.get('original_repeat_diagnostic')
    repeat_sha = None
    if diagnostic is not None:
        repeated_path = folder / 'original-repeat-endpoints.npz'
        with np.load(repeated_path, allow_pickle=False) as archive:
            repeated_all = {key: archive[key].copy() for key in archive.files}
        # Historical repeat files did not have a hash receipt. Check every
        # saved observation, and record their current digest without pretending
        # that it was recorded before the original execution.
        modes = {mode: {key: all_endpoints[mode + '.' + key] for key in repeated_all}
                 for mode in ('cuda', 'upstream')}
        observed = variation_checks(modes['cuda'], modes['upstream'], repeated_all)
        if observed != diagnostic['checks']:
            raise ValueError('Original repeat observations differ: ' + str(folder))
        repeated = {key: repeated_all[key] for key in FIELDS}
        repeat_sha = digest(repeated_path)
    else:
        repeated = arrays['upstream']
    result = compare_scopes(arrays['cuda'], arrays['upstream'], repeated, scene)
    result.update(task=task['suite'] + ':' + str(task['task_id']),
                  strict_passed=summary['passed'],
                  initial_numerical_passed=summary.get('initial_numerical_passed', True),
                  mandatory_local_passed=summary['endpoint_leaf_checks']['passed'],
                  original_repeat_available=diagnostic is not None,
                  evidence=dict(summary_sha256=digest(summary_path),
                                endpoints_sha256=expected_endpoint,
                                scene_inputs_sha256=scene_receipt['data_sha256'],
                                observed_repeat_sha256=repeat_sha),
                  report=str(summary_path))
    return result


def main(args):
    from libero.gpu.host.backends.mjwarp.scene_inputs import read
    tasks = select_tasks(load_catalog(args.build_root), args.tasks)
    rows, binding = [], None
    for task in tasks:
        folder = args.input / (task['suite'] + '-' + str(task['task_id']))
        scene, receipt = read(args.build_root, task)
        row = analyze_task(folder, task, scene, receipt, binding)
        if binding is None:
            binding = physics_binding(json.loads((folder / 'summary.json').read_text())['signature'])
        rows.append(row)
    result = dict(role='Supplemental scoped analysis; original strict statuses unchanged',
                  policy='Existing compare_replays: max/rms1024eps floor, 10x original spread, exact time',
                  full_state_scope='All qpos/qvel, OSC controller outputs and time; no acceleration/internal-force claim',
                  task_boundary_scope='Robot nine-DOF qpos/qvel, all body poses, OSC outputs and time; excludes free-object velocities and internal forces',
                  pose_epoch='xpos/xquat are public pre-final-Euler frames; full_state also checks post-Euler qpos',
                  repeat_binding='Current digest and all stored variation observations; historical repeats lacked a pre-execution hash receipt',
                  integrations=0, timing_repeated=False, expected=len(tasks), completed=len(rows),
                  strict_passed=sum(row['strict_passed'] for row in rows),
                  full_state_passed=sum(row['full_state']['passed'] for row in rows),
                  task_boundary_passed=sum(row['task_boundary']['passed'] for row in rows),
                  executed_physics=binding, rows=rows)
    save_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('rows', 'executed_physics')}, indent=2))
    return 0
