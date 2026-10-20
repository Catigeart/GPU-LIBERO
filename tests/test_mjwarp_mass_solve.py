"""CPU-only whole-caller fallback and nonduplicated acceleration scheduling."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import acceleration


def model_data():
    m=NS(nv=16,qLD_block_total=117,block_dim=NS(small_cholesky=64,cholesky_solve=64),
         M_tiles=[NS(size=1,elemid=NS(size=0)),NS(size=6,elemid=NS(size=0)),NS(size=9,elemid=NS(size=81))],
         opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    d=NS(nworld=2,qLD=NS(ndim=2,shape=(2,117)))
    return m,d


@pytest.mark.parametrize('variant,reason',[
    ('sparse','unported sparse LDL mass region or factor layout'),
    ('discrete','discrete effective metric'),
    ('sleep','active-DOF compaction'),
    ('scalar','unqualified scalar mass block size'),
    ('tile','unqualified tile mass block size'),
    ('threads','unqualified mass block dimension'),
    ('empty','no qualified mass blocks')])
def test_unqualified_domain_falls_back_before_cuda_load(monkeypatch,variant,reason):
    m,d=model_data()
    if variant=='sparse':d.qLD.shape=(2,118)
    elif variant=='discrete':m.opt.integrator=types.IntegratorType.DISCRETE
    elif variant=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif variant=='scalar':m.M_tiles[1].size=5
    elif variant=='tile':m.M_tiles[2].size=12
    elif variant=='threads':m.block_dim.cholesky_solve=32
    else:m.M_tiles=[]
    monkeypatch.setattr(acceleration,'MassSolvePlan',lambda *a:pytest.fail('unexpected CUDA load'))
    calls=[]
    monkeypatch.setattr(acceleration.forward,'fwd_acceleration',lambda *a:calls.append(a))
    driver=acceleration.Acceleration(m,d,None,implementation='cuda')
    driver.enqueue()
    assert driver.identity==dict(requested='cuda',selected='upstream',fallback=reason)
    assert calls==[(m,d)]


def test_default_keeps_complete_upstream_caller(monkeypatch):
    m,d=model_data();calls=[]
    monkeypatch.setattr(acceleration,'MassSolvePlan',lambda *a:pytest.fail('unexpected CUDA load'))
    monkeypatch.setattr(acceleration.forward,'fwd_acceleration',lambda *a:calls.append(a))
    driver=acceleration.Acceleration(m,d,None)
    driver.enqueue()
    assert driver.identity==dict(requested='upstream',selected='upstream',fallback=None)
    assert calls==[(m,d)]


def test_invalid_selector_rejected():
    with pytest.raises(ValueError,match='implementation'):
        acceleration.Acceleration(*model_data(),None,implementation='typo')


def test_selected_caller_assembles_forces_once_then_only_selected_solve(monkeypatch):
    m,d=model_data();calls=[]
    m.body_treeid='treeid';m.dof_bodyid='bodyid'
    for key in ('qfrc_applied','tree_awake','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_smooth','qacc_smooth'):
        setattr(d,key,key)
    def bind(x,y):
        assert (x,y)==('qacc_smooth','qfrc_smooth')
        return NS(enqueue=lambda:calls.append('selected-solve'))
    monkeypatch.setattr(acceleration,'MassSolvePlan',lambda *a:NS(bind=bind,identity={'test':'plan'}))
    monkeypatch.setattr(acceleration.forward,'fwd_acceleration',lambda *a:pytest.fail('duplicate upstream solve'))
    monkeypatch.setattr(acceleration.forward,'_qfrc_smooth',lambda sleep:'smooth-prefix' if not sleep else pytest.fail('sleep'))
    monkeypatch.setattr(acceleration.wp,'launch',lambda kernel,**kw:calls.append((kernel,kw)))
    monkeypatch.setattr(acceleration.forward,'xfrc_accumulate',lambda *a:calls.append(('xfrc',a)))
    driver=acceleration.Acceleration(m,d,None,implementation='cuda',library='admitted')
    driver.enqueue()
    assert calls==[('smooth-prefix',dict(dim=(2,16),inputs=['treeid','bodyid','qfrc_applied',
        'tree_awake','qfrc_bias','qfrc_passive','qfrc_actuator'],outputs=['qfrc_smooth'])),
        ('xfrc',(m,d,'qfrc_smooth')),'selected-solve']
    assert driver.identity==dict(requested='cuda',selected='cuda',fallback=None,test='plan')


def test_complete_acceleration_uses_only_selected_force_and_mass(monkeypatch):
    from libero.gpu.host.runtime.dynamics import mjwarp_smooth_force
    m,d=model_data();d.qacc_smooth='acc';d.qfrc_smooth='force';calls=[]
    monkeypatch.setattr(acceleration,'MassSolvePlan',lambda *a:NS(identity={},
        bind=lambda x,y:NS(enqueue=lambda:calls.append('mass'))))
    monkeypatch.setattr(mjwarp_smooth_force,'SmoothForce',lambda *a:NS(identity={'source':'force'},
        enqueue=lambda:calls.append('force')))
    monkeypatch.setattr(acceleration.wp,'launch',lambda *a,**kw:pytest.fail('upstream numerical prefix'))
    monkeypatch.setattr(acceleration.forward,'xfrc_accumulate',lambda *a:pytest.fail('upstream external force'))
    driver=acceleration.Acceleration(m,d,None,implementation='cuda',force_implementation='cuda',force_library='lib')
    driver.enqueue()
    assert calls==['force','mass']
    assert driver.identity['forces']==dict(requested='cuda',selected='cuda',fallback=None,source='force')


def test_complete_acceleration_falls_back_as_one_caller(monkeypatch):
    from libero.gpu.host.runtime.dynamics import mjwarp_smooth_force
    m,d=model_data();m.opt.integrator=types.IntegratorType.DISCRETE;calls=[]
    monkeypatch.setattr(acceleration,'MassSolvePlan',lambda *a:pytest.fail('CUDA mass load'))
    monkeypatch.setattr(mjwarp_smooth_force,'SmoothForce',lambda *a:pytest.fail('CUDA force load'))
    monkeypatch.setattr(acceleration.forward,'fwd_acceleration',lambda *a:calls.append('upstream'))
    driver=acceleration.Acceleration(m,d,None,implementation='cuda',force_implementation='cuda')
    driver.enqueue()
    assert calls==['upstream']
    assert driver.identity['forces']==dict(requested='cuda',selected='upstream',fallback='discrete effective metric')


@pytest.mark.parametrize('mass,force',[('cuda','typo'),('upstream','cuda')])
def test_invalid_complete_selector_does_not_load(mass,force):
    with pytest.raises(ValueError,match='force implementation|requires CUDA mass'):
        acceleration.Acceleration(*model_data(),None,implementation=mass,force_implementation=force)
