"""CPU-only topology safety and independent mass-matrix/factor dispatch."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import mass_matrix as module
from libero.gpu.host.backends.mjwarp.position import Position
from libero.gpu.host.backends.mjwarp.stages import PhysicsStages
from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import validate_topology


def topology():
    return [np.array(x, np.int32) for x in ([0, 0, 1, 1], [1, 2, 3], [-1, 0, 0],
        [0, 1, 3], [1, 2, 2], [0, 0, 1, 0, 2])] + [[np.array(x, np.int32) for x in ([0], [1], [2, 3])]]


def test_valid_branched_rows_have_unique_ownership():
    validate_topology(*topology())


@pytest.mark.parametrize('fault', ('body-cycle', 'dof-cycle', 'body-range', 'row-range',
    'wrong-ancestor', 'layer-depth', 'layer-duplicate', 'layer-missing'))
def test_unsafe_topology_rejected_before_device_traversal(fault):
    a = topology()
    if fault == 'body-cycle': a[0][2] = 2
    elif fault == 'dof-cycle': a[2][1] = 1
    elif fault == 'body-range': a[1][0] = 4
    elif fault == 'row-range': a[3][1] = 0
    elif fault == 'wrong-ancestor': a[5][1] = 1
    elif fault == 'layer-depth': a[6][1][0] = 2
    elif fault == 'layer-duplicate': a[6][2] = np.array([2, 2, 3], np.int32)
    else: a[6][2] = np.array([2], np.int32)
    with pytest.raises(ValueError, match='Mass matrix'):
        validate_topology(*a)


def model():
    return NS(nv=3, nbody=4, opt=NS(integrator=types.IntegratorType.EULER,
        enableflags=0, run_collision_detection=True))


@pytest.mark.parametrize('variant', ('default', 'sleep', 'implicit', 'empty'))
def test_upstream_and_fallback_do_not_load_cuda(monkeypatch, variant):
    m = model(); calls = []
    if variant == 'sleep': m.opt.enableflags = types.EnableBit.SLEEP
    elif variant == 'implicit': m.opt.integrator = types.IntegratorType.IMPLICIT
    elif variant == 'empty': m.nv = 0
    monkeypatch.setattr(module, 'MassMatrix', lambda *a: pytest.fail('optional CUDA load'))
    monkeypatch.setattr(module.smooth, 'crb', lambda *a: calls.append('crb'))
    driver = module.MassMatrices(m, None, None, implementation='upstream' if variant == 'default' else 'cuda')
    driver.enqueue()
    assert calls == ['crb'] and driver.identity['selected'] == 'upstream'
    assert bool(driver.identity['fallback']) == (variant != 'default')


@pytest.mark.parametrize('matrix_cuda,factor_cuda', ((False, False), (True, False), (False, True), (True, True)))
@pytest.mark.parametrize('collision', (False, True))
def test_common_position_schedule_has_one_matrix_and_one_factor(monkeypatch, matrix_cuda, factor_cuda, collision):
    from libero.gpu.host.backends.mjwarp import position as p
    m = model(); m.opt.run_collision_detection = collision; d = NS(); calls = []
    for owner, name, label in ((p.forward, 'fwd_position', 'whole-position'),
            (p.forward, 'fwd_kinematics', 'kinematics'), (p.smooth, 'crb', 'upstream-matrix'),
            (p.smooth, 'tendon_armature', 'tendon'), (p.smooth, 'factor_m', 'upstream-factor'),
            (p.forward.collision_driver, 'collision', 'collision'),
            (p.forward.constraint, 'make_constraint', 'constraints'), (p.smooth, 'transmission', 'transmission')):
        monkeypatch.setattr(owner, name, lambda *a, label=label: calls.append(label))
    matrix = NS(selected='cuda' if matrix_cuda else 'upstream', enqueue=lambda: calls.append('cuda-matrix'))
    factor = NS(selected='cuda' if factor_cuda else 'upstream',
                position_binding=NS(enqueue=lambda: calls.append('cuda-factor')))
    stages = PhysicsStages(m, d, mass_matrix=matrix, mass_factor=factor)
    stages.position()
    if not matrix_cuda and not factor_cuda:
        assert calls == ['whole-position']
    else:
        assert calls == ['kinematics', 'cuda-matrix' if matrix_cuda else 'upstream-matrix', 'tendon',
                         'cuda-factor' if factor_cuda else 'upstream-factor'] + (['collision'] if collision else []) + ['constraints', 'transmission']


def test_matrix_selection_does_not_replace_integration():
    calls = []
    stages = PhysicsStages(model(), NS(), mass_matrix=NS(selected='cuda'),
                          mass_factor=NS(integrate=lambda: calls.append('existing-integrator')))
    stages.integrate()
    assert calls == ['existing-integrator']


def test_invalid_selector_rejected():
    with pytest.raises(ValueError, match='implementation'):
        module.MassMatrices(model(), None, None, implementation='typo')
