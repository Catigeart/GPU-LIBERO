"""Retained-output reports must preserve failures and reject unbound evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from validation.mjwarp_endpoint_report import analyze_task, digest, physics_binding
from validation.mjwarp_trajectory_reference import variation_checks


def fixture(tmp_path, free_velocity=0.):
    scene = dict(body_names=np.array(['world', 'robot0_link', 'object']),
                 jnt_bodyid=np.array([1] * 9 + [2]), jnt_type=np.array([3] * 9 + [0]),
                 jnt_qposadr=np.arange(10), jnt_dofadr=np.arange(10))
    first = dict(qpos=np.zeros((2, 16)), qvel=np.zeros((2, 15)),
                 xpos=np.zeros((2, 3, 3)), xquat=np.zeros((2, 3, 4)),
                 ctrl=np.zeros((2, 9)), time=np.ones(2))
    candidate = {key: value.copy() for key, value in first.items()}
    candidate['qvel'][0, 12] = free_velocity
    np.savez(tmp_path / 'endpoints.npz', **{mode + '.' + key: value for mode, values in
             (('cuda', candidate), ('upstream', first)) for key, value in values.items()})
    signature = dict(sources={'kernel': 'physical-v1'}, forward_helper='forward-v1',
                     scene_helper='scene-v1', repeat_helper='repeat-v1', protocol={'worlds': 2},
                     identity=dict(task='suite/task.bddl', worlds=2, fixed_scene_inputs_sha256='scene-data'))
    summary = dict(signature=signature, timings={'cuda': {}, 'upstream': {}}, passed=False,
                   initial_numerical_passed=False, endpoint_leaf_checks={'passed': True},
                   endpoints_sha256=digest(tmp_path / 'endpoints.npz'))
    write(tmp_path, summary)
    return scene, summary, candidate, first


def write(folder, summary):
    (folder / 'summary.json').write_text(json.dumps(summary))


def analyze(folder, scene, binding=None):
    return analyze_task(folder, {'key': 'suite/task.bddl', 'suite': 'suite', 'task_id': 0},
                        scene, {'data_sha256': 'scene-data'}, binding)


def test_free_velocity_failure_is_visible_despite_task_boundary_pass(tmp_path):
    scene, summary, _, _ = fixture(tmp_path, free_velocity=.01)
    result = analyze(tmp_path, scene)
    assert result['task_boundary']['passed']
    assert not result['full_state']['passed']
    assert not result['strict_passed'] and not result['initial_numerical_passed']
    assert json.loads((tmp_path / 'summary.json').read_text()) == summary


def test_changed_endpoint_cannot_be_used(tmp_path):
    scene, _, _, _ = fixture(tmp_path)
    with (tmp_path / 'endpoints.npz').open('ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='endpoint receipt'):
        analyze(tmp_path, scene)


def test_executed_sources_must_be_shared(tmp_path):
    scene, summary, _, _ = fixture(tmp_path)
    binding = physics_binding(summary['signature'])
    binding['sources'] = {'kernel': 'different'}
    with pytest.raises(ValueError, match='physics/reference sources'):
        analyze(tmp_path, scene, binding)


def test_changed_fixture_input_cannot_be_used(tmp_path):
    scene, summary, _, _ = fixture(tmp_path)
    summary['signature']['identity']['fixed_scene_inputs_sha256'] = 'different'
    write(tmp_path, summary)
    with pytest.raises(ValueError, match='scene input binding'):
        analyze(tmp_path, scene)


def test_original_repeat_must_match_all_retained_observations(tmp_path):
    scene, summary, candidate, first = fixture(tmp_path)
    repeated = {key: value.copy() for key, value in first.items()}
    summary['original_repeat_diagnostic'] = dict(checks=variation_checks(candidate, first, repeated))
    np.savez(tmp_path / 'original-repeat-endpoints.npz', **repeated)
    write(tmp_path, summary)
    assert analyze(tmp_path, scene)['original_repeat_available']
    repeated['qvel'][0, 12] = .02
    np.savez(tmp_path / 'original-repeat-endpoints.npz', **repeated)
    with pytest.raises(ValueError, match='repeat observations'):
        analyze(tmp_path, scene)


def test_partial_run_has_no_scoped_success(tmp_path):
    scene, summary, _, _ = fixture(tmp_path)
    summary['stage'] = 'initial-forward'
    write(tmp_path, summary)
    with pytest.raises(ValueError, match='Incomplete trajectory'):
        analyze(tmp_path, scene)


def test_untimed_complete_run_is_readable_without_inventing_timings(tmp_path):
    scene, summary, _, _ = fixture(tmp_path)
    summary['timings']={}
    summary['execution']=dict(kind='correctness_only',controls_per_mode=25,completed_modes=['cuda','upstream'])
    write(tmp_path,summary)
    assert analyze(tmp_path,scene)['full_state']['passed']
    summary['execution']['completed_modes']=['cuda']
    write(tmp_path,summary)
    with pytest.raises(ValueError,match='Incomplete trajectory'):analyze(tmp_path,scene)
