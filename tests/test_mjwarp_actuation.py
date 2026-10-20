"""Stateless actuation admission, callback order, public clear and history owner."""
from types import SimpleNamespace as NS
import numpy as np
import pytest


def model_data():
    from mujoco_warp._src import types
    arr=lambda x:NS(numpy=lambda:np.asarray(x))
    m=NS(nv=3,nu=3,nactuator=3,njnt=3,na=0,ntendon=0,nhistory=0,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0),
        callback=NS(act_dyn=None,act_gain=None,act_bias=None),
        actuator_dyntype=arr([0,0,0]),actuator_gaintype=arr([0,1,0]),actuator_biastype=arr([0,0,1]),
        actuator_trntype=arr([0,0,0]),actuator_trnid=arr([[0,-1],[1,-1],[2,-1]]),
        actuator_ctrladr=arr([0,1,2]),actuator_ctrlnum=arr([1,1,1]),actuator_actadr=arr([-1,-1,-1]),jnt_type=arr([2,3,2]))
    d=NS(nworld=2,ctrl=NS(capacity=24),actuator_moment=NS(shape=(2,3)))
    return m,d


def test_qualified_scalar_affine_and_fixed_domain():
    from libero.gpu.host.runtime.dynamics.mjwarp_actuation import qualify
    m,d=model_data();a=qualify(m,d)
    assert a['reason'] is None and a['model']==id(m) and a['data']==id(d)


@pytest.mark.parametrize('fault',('implicit','sleep','stateful','tendon','empty','empty-moments','dyn','gain','bias','transmission','ball','multi-control'))
def test_unsupported_domain_has_visible_fallback_and_no_load(monkeypatch,fault):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();calls=[];arr=lambda x:NS(numpy=lambda:np.asarray(x))
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='stateful':m.na=1
    elif fault=='tendon':m.ntendon=1
    elif fault=='empty':m.nactuator=0
    elif fault=='empty-moments':d.actuator_moment.shape=(2,0)
    elif fault=='dyn':m.actuator_dyntype=arr([1,0,0])
    elif fault=='gain':m.actuator_gaintype=arr([2,0,0])
    elif fault=='bias':m.actuator_biastype=arr([0,2,0])
    elif fault=='transmission':m.actuator_trntype=arr([3,0,0])
    elif fault=='ball':m.jnt_type=arr([1,3,2])
    elif fault=='multi-control':m.actuator_ctrlnum=arr([2,1,1])
    monkeypatch.setattr(a,'ActuatorForces',lambda *x,**kw:pytest.fail('optional library loading'))
    monkeypatch.setattr(a.forward,'fwd_actuation',lambda *x:calls.append('upstream'))
    driver=a.Actuation(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.fallback and calls==['upstream']


def test_bad_control_and_joint_addresses_rejected():
    from libero.gpu.host.runtime.dynamics.mjwarp_actuation import qualify
    m,d=model_data();m.actuator_ctrladr=NS(numpy=lambda:np.array([0,1,3]))
    with pytest.raises(ValueError,match='control address'):qualify(m,d)
    m,d=model_data();m.actuator_trnid=NS(numpy=lambda:np.array([[3,-1],[1,-1],[2,-1]]))
    with pytest.raises(ValueError,match='transmission joint'):qualify(m,d)


def test_default_does_not_read_metadata_or_load_optional_binary(monkeypatch):
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();calls=[]
    monkeypatch.setattr(a,'qualify',lambda *x:pytest.fail('metadata download'))
    monkeypatch.setattr(a,'ActuatorForces',lambda *x,**kw:pytest.fail('optional library load'))
    monkeypatch.setattr(a.forward,'fwd_actuation',lambda *x:calls.append('whole-upstream'))
    driver=a.Actuation(m,d,None);driver.enqueue();assert calls==['whole-upstream']


def test_callbacks_keep_order_between_output_force_and_projection(monkeypatch):
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();calls=[]
    for key in ('act_dyn','act_gain','act_bias'):setattr(m.callback,key,lambda *x,key=key:calls.append(key))
    monkeypatch.setattr(a,'ActuatorForces',lambda *x,**kw:NS(enqueue=lambda phase:calls.append(phase)))
    a.Actuation(m,d,None,implementation='cuda').enqueue()
    assert calls==[0,'act_dyn','act_gain','act_bias',1,2]


def test_disabled_clears_all_public_outputs_and_skips_callbacks(monkeypatch):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();m.opt.disableflags=types.DisableBit.ACTUATION;calls=[]
    for key in ('act_dot','qfrc_actuator','actuator_force'):setattr(d,key,NS(zero_=lambda key=key:calls.append(key)))
    monkeypatch.setattr(a,'ActuatorForces',lambda *x,**kw:NS(enqueue=lambda phase:pytest.fail('disabled launch')))
    driver=a.Actuation(m,d,None,implementation='cuda');driver.enqueue()
    assert calls==['act_dot','qfrc_actuator','actuator_force']


def test_history_read_precedes_force_and_workspace_is_retained(monkeypatch):
    import warp as wp
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();m.nhistory=8;calls=[];workspace=NS(capacity=24)
    monkeypatch.setattr(wp,'empty_like',lambda ctrl:workspace)
    def binding(*x,**kw):
        assert kw['control'] is workspace
        return NS(enqueue=lambda phase:calls.append(phase),identity={})
    monkeypatch.setattr(a,'ActuatorForces',binding)
    def history(model,data,ctrl):
        assert model is m and data is d and ctrl is workspace;calls.append('read-history')
    monkeypatch.setattr(a.history,'read_ctrl_delayed',history)
    driver=a.Actuation(m,d,None,implementation='cuda');driver.enqueue();driver.enqueue()
    assert driver.control is workspace and driver.identity['control_workspace_bytes']==24
    assert calls==['read-history',0,1,2]*2


def test_admission_is_not_repeated_and_cannot_cross_owners(monkeypatch):
    from libero.gpu.host.backends.mjwarp import actuation as a
    m,d=model_data();admission=a.qualify(m,d)
    monkeypatch.setattr(a,'qualify',lambda *x:pytest.fail('duplicate admission'))
    monkeypatch.setattr(a,'ActuatorForces',lambda *x,**kw:NS())
    assert a.Actuation(m,d,None,implementation='cuda',admission=admission).selected=='cuda'
    with pytest.raises(ValueError,match='owner'):a.Actuation(*model_data(),None,implementation='cuda',admission=admission)


def test_stage_uses_exactly_one_selected_actuation(monkeypatch):
    from libero.gpu.host.backends.mjwarp import stages
    m,d=model_data();calls=[]
    monkeypatch.setattr(stages.forward,'fwd_actuation',lambda *x:pytest.fail('duplicate actuation'))
    p=stages.PhysicsStages(m,d,actuation=NS(enqueue=lambda:calls.append('cuda-actuation')));p.actuation()
    assert calls==['cuda-actuation']
