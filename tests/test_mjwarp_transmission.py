"""Scalar-domain admission, independent selectors and original stage ordering."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import transmission as module


def model_data():
    def arr(a):return NS(numpy=lambda:np.asarray(a))
    m=NS(nv=4,nq=4,njnt=4,nactuator=3,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0),
        actuator_trntype=arr([0,1,0]),actuator_trnid=arr([[1,-1],[3,-1],[1,-1]]),
        jnt_type=arr([2,3,2,3]),jnt_qposadr=arr([0,1,2,3]),jnt_dofadr=arr([0,1,2,3]))
    d=NS(nworld=2,actuator_moment=NS(shape=(2,8)))
    return m,d


def test_scalar_domain_and_duplicate_target_joint():
    a=module.qualify(*model_data(),transmission='cuda',actuator_velocity='cuda')
    assert a['selected']==dict(transmission='cuda',actuator_velocity='cuda') and not any(a['reasons'].values())


@pytest.mark.parametrize('fault',('implicit','sleep','empty','storage'))
def test_whole_caller_fallback_precedes_optional_load(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='empty':m.nactuator=0
    else:d.actuator_moment.shape=(2,0)
    monkeypatch.setattr(module,'TransmissionNumerics',lambda *a,**k:pytest.fail('optional load'))
    p=module.Transmission(m,d,None,transmission='cuda',actuator_velocity='cuda')
    assert set(p.selected.values())=={'upstream'} and all(p.admission['reasons'].values())


@pytest.mark.parametrize('fault',('site','tendon','free','ball'))
def test_transmission_fallback_keeps_generic_cuda_velocity(monkeypatch,fault):
    m,d=model_data()
    if fault in ('site','tendon'):m.actuator_trntype=NS(numpy=lambda:np.array([4 if fault=='site' else 3,0,0]))
    else:m.jnt_type=NS(numpy=lambda:np.array([2,0 if fault=='free' else 1,2,3]))
    calls=[]
    def binding(*a,**kw):
        assert kw['admission']['selected']==dict(transmission='upstream',actuator_velocity='cuda')
        return NS(enqueue=lambda k:calls.append(k))
    monkeypatch.setattr(module,'TransmissionNumerics',binding)
    monkeypatch.setattr(module.smooth,'transmission',lambda *a:calls.append('upstream transmission'))
    p=module.Transmission(m,d,None,transmission='cuda',actuator_velocity='cuda');p.enqueue_transmission();p.enqueue_velocity()
    assert calls==['upstream transmission',1]


def test_default_no_metadata_read_load_or_allocation(monkeypatch):
    m,d=model_data()
    for field in ('actuator_trntype','actuator_trnid','jnt_type','jnt_qposadr','jnt_dofadr'):
        setattr(m,field,NS(numpy=lambda:pytest.fail('metadata download')))
    monkeypatch.setattr(module,'TransmissionNumerics',lambda *a,**k:pytest.fail('optional load'))
    p=module.Transmission(m,d,None);assert p.binding is None and not p.admission['values']


def test_velocity_alone_needs_no_transmission_metadata_download(monkeypatch):
    m,d=model_data()
    for field in ('actuator_trntype','actuator_trnid','jnt_type','jnt_qposadr','jnt_dofadr'):
        setattr(m,field,NS(numpy=lambda:pytest.fail('metadata download')))
    monkeypatch.setattr(module,'TransmissionNumerics',lambda *a,**k:NS())
    p=module.Transmission(m,d,None,actuator_velocity='cuda')
    assert p.selected==dict(transmission='upstream',actuator_velocity='cuda')


@pytest.mark.parametrize('fault',('joint-id','qpos','dof','capacity','shape'))
def test_invalid_domain_storage_rejected(fault):
    m,d=model_data()
    if fault=='joint-id':m.actuator_trnid=NS(numpy=lambda:np.array([[4,-1],[3,-1],[1,-1]]))
    elif fault=='qpos':m.jnt_qposadr=NS(numpy=lambda:np.array([0,4,2,3]))
    elif fault=='dof':m.jnt_dofadr=NS(numpy=lambda:np.array([0,1,2,-1]))
    elif fault=='capacity':d.actuator_moment.shape=(2,2)
    else:m.actuator_trnid=NS(numpy=lambda:np.array([1,3,1]))
    with pytest.raises(ValueError):module.qualify(m,d,transmission='cuda')


def test_admission_reused_with_exact_owner_and_selector(monkeypatch):
    m,d=model_data();a=module.qualify(m,d,transmission='cuda');calls=[]
    monkeypatch.setattr(module,'qualify',lambda *a,**k:pytest.fail('duplicate metadata download'))
    monkeypatch.setattr(module,'TransmissionNumerics',lambda *args,**kw:NS(enqueue=lambda k:calls.append(k)))
    p=module.Transmission(m,d,None,transmission='cuda',admission=a);p.enqueue_transmission();assert calls==[0]
    with pytest.raises(ValueError,match='owner'):module.Transmission(*model_data(),None,transmission='cuda',admission=a)
    with pytest.raises(ValueError,match='selectors'):module.Transmission(m,d,None,actuator_velocity='cuda',admission=a)


def test_counter_reset_only_for_transmission_and_phase_selection():
    import ctypes as ct
    from libero.gpu.host.runtime.dynamics.mjwarp_transmission import TransmissionNumerics,TransmissionInputs
    p=TransmissionNumerics.__new__(TransmissionNumerics);p.selected=dict(transmission='cuda',actuator_velocity='cuda');calls=[]
    p.arrays={'total':NS(zero_=lambda:calls.append('zero counter'))};p.arguments=TransmissionInputs();p.stream=NS(cuda_stream=17)
    p.launch=lambda a,phase,stream:calls.append((phase,stream)) or 0
    p.enqueue(0);p.enqueue(1);p.enqueue(0)
    assert calls==['zero counter',(0,17),(1,17),'zero counter',(0,17)]
    p.selected['transmission']='upstream'
    with pytest.raises(ValueError,match='not selected'):p.enqueue(0)
    with pytest.raises(ValueError):p.enqueue(2)
    p.launch=lambda *a:7
    with pytest.raises(RuntimeError,match='CUDA error 7'):p.enqueue(1)


def test_transmission_only_splits_position_once_and_keeps_original_order(monkeypatch):
    from libero.gpu.host.backends.mjwarp import position as pm
    m=NS(opt=NS(run_collision_detection=True));d=NS();calls=[]
    for owner,name in ((pm.forward,'fwd_kinematics'),(pm.smooth,'crb'),(pm.smooth,'tendon_armature'),(pm.smooth,'factor_m'),
            (pm.forward.collision_driver,'collision'),(pm.forward.constraint,'make_constraint')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    monkeypatch.setattr(pm.forward,'fwd_position',lambda *a:pytest.fail('duplicate position'))
    monkeypatch.setattr(pm.smooth,'transmission',lambda *a:pytest.fail('duplicate transmission'))
    p=pm.Position(m,d,transmission=NS(selected=dict(transmission='cuda'),enqueue_transmission=lambda:calls.append('cuda transmission')));p.position()
    assert calls==['fwd_kinematics','crb','tendon_armature','factor_m','collision','make_constraint','cuda transmission']


def test_velocity_only_splits_stage_and_preserves_other_calls(monkeypatch):
    from libero.gpu.host.backends.mjwarp import velocity as vm
    m=NS(nactuator=3,ntendon=0);d=NS(nworld=2,qvel='qvel',ten_velocity='tv',qfrc_bias='bias');calls=[]
    for k in ('ten_J_rownnz','ten_J_rowadr','ten_J_colind'):setattr(m,k,k)
    d.ten_J='J'
    monkeypatch.setattr(vm.forward,'fwd_velocity',lambda *a:pytest.fail('duplicate stage'))
    monkeypatch.setattr(vm.wp,'launch',lambda kernel,**kw:calls.append(kernel.key))
    for owner,name in ((vm.smooth,'com_vel'),(vm.passive,'passive'),(vm.smooth,'rne'),(vm.smooth,'tendon_bias')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    p=vm.Velocity(m,d,None,transmission=NS(selected=dict(actuator_velocity='cuda'),enqueue_velocity=lambda:calls.append('cuda velocity')));p.enqueue()
    assert calls==['cuda velocity','_tendon_velocity','com_vel','passive','rne','tendon_bias']


def test_bad_selector():
    with pytest.raises(ValueError,match='must be'):module.Transmission(*model_data(),None,transmission='typo')
