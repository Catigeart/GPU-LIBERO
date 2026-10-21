"""Cold CoM ownership/admission and independent position-boundary dispatch."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import com_pos as c, kinematics as k, position as p
from libero.gpu.host.runtime.dynamics.mjwarp_com_pos import validate_indices


def model_data():
    return NS(nv=2,nq=2,njnt=2,nbody=4,nbranch=2,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0)),NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','nv','njnt','nbody','worlds'))
def test_fallback_before_optional_load(monkeypatch,fault):
    m,d=model_data();calls=[]
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='worlds':d.nworld=0
    else:setattr(m,fault,0)
    monkeypatch.setattr(c,'CenteredFrames',lambda *a:pytest.fail('optional load'))
    monkeypatch.setattr(c.smooth,'com_pos',lambda *a:calls.append('upstream'))
    driver=c.ComPos(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.fallback and calls==['upstream']


def test_default_no_admission_or_load(monkeypatch):
    monkeypatch.setattr(c,'caller_unsupported_reason',lambda *a:pytest.fail('admission'))
    monkeypatch.setattr(c,'CenteredFrames',lambda *a:pytest.fail('optional load'))
    assert c.ComPos(*model_data(),None).selected=='upstream'


def test_bad_selector():
    with pytest.raises(ValueError,match='com_pos must'):c.ComPos(*model_data(),None,implementation='typo')


def indices():
    return [2,np.array([0,0,1,0]),np.array([0,1,1,3]),np.array([3,2]),np.array([0,1]),np.array([1,2]),
            [np.array([0]),np.array([1,3]),np.array([2])]]


def test_valid_tree_and_disjoint_joint_writes():validate_indices(*indices())


@pytest.mark.parametrize('fault',('parent-cycle','root','kind','body','span','overlap','depth','duplicate','missing'))
def test_unsafe_indices_rejected(fault):
    a=indices()
    if fault=='parent-cycle':a[1][2]=2
    elif fault=='root':a[2][2]=2
    elif fault=='kind':a[3][0]=4
    elif fault=='body':a[5][0]=4
    elif fault=='span':a[4][1]=2
    elif fault=='overlap':a[4][1]=0
    elif fault=='depth':a[6][1]=np.array([1,2,3])
    elif fault=='duplicate':a[6][1]=np.array([1,1,3])
    else:a[6][1]=np.array([1])
    with pytest.raises(ValueError,match='CoM'):validate_indices(*a)


@pytest.mark.parametrize('kin_cuda',(False,True))
@pytest.mark.parametrize('com_cuda',(False,True))
def test_independent_kinematics_com_selection(monkeypatch,kin_cuda,com_cuda):
    m,d=model_data();calls=[]
    monkeypatch.setattr(k,'KinematicFrames',lambda *a,**kw:NS(enqueue=lambda phase:calls.append('kin'+str(phase))))
    monkeypatch.setattr(k.forward,'fwd_kinematics',lambda *a:calls.append('whole'))
    for name in ('kinematics','com_pos','camlight','flex','tendon'):
        monkeypatch.setattr(k.smooth,name,lambda *a,name=name:calls.append(name))
    driver=k.Kinematics(m,d,None,implementation='cuda' if kin_cuda else 'upstream')
    com=NS(selected='cuda' if com_cuda else 'upstream',enqueue=lambda:calls.append('CUDA com'))
    driver.forward(com_pos=com)
    expected=['kin'+str(i) for i in range(5)] if kin_cuda else ['kinematics']
    expected += ['CUDA com' if com_cuda else 'com_pos','camlight','flex','tendon']
    assert calls==(expected if kin_cuda or com_cuda else ['whole'])


@pytest.mark.parametrize('with_kin_driver',(False,True))
def test_only_com_cuda_keeps_full_position_order(monkeypatch,with_kin_driver):
    m,d=model_data();m.opt.run_collision_detection=True;calls=[]
    for name in ('fwd_position','fwd_kinematics'):
        monkeypatch.setattr(p.forward,name,lambda *a:pytest.fail('duplicate whole caller'))
    for owner,name in ((p.smooth,'kinematics'),(p.smooth,'camlight'),(p.smooth,'flex'),(p.smooth,'tendon'),
                       (p.smooth,'crb'),(p.smooth,'tendon_armature'),(p.smooth,'factor_m'),
                       (p.forward.collision_driver,'collision'),(p.forward.constraint,'make_constraint'),(p.smooth,'transmission')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    monkeypatch.setattr(p.smooth,'com_pos',lambda *a:pytest.fail('duplicate com'))
    kin=k.Kinematics(m,d,None) if with_kin_driver else None
    com=NS(selected='cuda',enqueue=lambda:calls.append('CUDA com'))
    p.Position(m,d,kinematics=kin,com_pos=com).position()
    assert calls==['kinematics','CUDA com','camlight','flex','tendon','crb','tendon_armature','factor_m','collision','make_constraint','transmission']


def test_fallback_com_preserves_whole_default_position(monkeypatch):
    calls=[];m,d=model_data()
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:calls.append('whole'))
    p.Position(m,d,com_pos=NS(selected='upstream',enqueue=lambda:pytest.fail('duplicate'))).position()
    assert calls==['whole']
