"""Rigid-domain admission, index ownership, upstream tail and position ordering."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import kinematics as k
from libero.gpu.host.runtime.dynamics.mjwarp_kinematics import validate_indices


def model_data():
    return NS(nv=2,nq=2,njnt=2,nbody=3,nbranch=1,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0)),NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','nv','nq','njnt','nbody','nbranch','worlds'))
def test_fallback_before_optional_load(monkeypatch,fault):
    m,d=model_data();calls=[]
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='worlds':d.nworld=0
    else:setattr(m,fault,0)
    monkeypatch.setattr(k,'KinematicFrames',lambda *a,**kw:pytest.fail('optional CUDA load'))
    monkeypatch.setattr(k.forward,'fwd_kinematics',lambda *a:calls.append('whole upstream'))
    p=k.Kinematics(m,d,None,implementation='cuda');p.forward()
    assert p.selected=='upstream' and p.fallback and calls==['whole upstream']


def test_default_no_admission_or_optional_load(monkeypatch):
    monkeypatch.setattr(k,'caller_unsupported_reason',lambda *a:pytest.fail('cold admission'))
    monkeypatch.setattr(k,'KinematicFrames',lambda *a,**kw:pytest.fail('optional load'))
    p=k.Kinematics(*model_data(),None);assert p.binding is None and p.selected=='upstream'


def test_shared_topology_and_five_phases_then_original_tail(monkeypatch):
    m,d=model_data();topology=NS();calls=[]
    def bind(mm,dd,lib,stream,*,topology):
        assert mm is m and dd is d and topology is shared
        return NS(enqueue=lambda phase:calls.append(phase),identity={'test':True})
    shared=topology;monkeypatch.setattr(k,'KinematicFrames',bind)
    for name in ('com_pos','camlight','flex','tendon'):monkeypatch.setattr(k.smooth,name,lambda *a,name=name:calls.append(name))
    monkeypatch.setattr(k.forward,'fwd_kinematics',lambda *a:pytest.fail('duplicated forward'))
    monkeypatch.setattr(k.smooth,'kinematics',lambda *a:pytest.fail('duplicated kinematics'))
    p=k.Kinematics(m,d,None,implementation='cuda',topology=topology);p.forward()
    assert calls==[0,1,2,3,4,'com_pos','camlight','flex','tendon']


def indices():
    # HINGE on body1, SLIDE on body2; body3 is fixed mocap under world.
    return [2,1,np.array([0,1]),np.array([3,2]),np.array([0,1,1,0]),np.array([-1,0,1,-1]),np.array([-1,-1,-1,0]),np.array([0,1,1,3]),np.array([0,1,2,3]),np.array([0,2,3]),np.array([1,3])]


def test_valid_geometry_indices():validate_indices(*indices())


@pytest.mark.parametrize('fault',('mocap','root','weld','geom','site','qpos-span','qpos-overlap','body-span','joint-type','free-multiple'))
def test_bad_geometry_indices_rejected(fault):
    a=indices()
    if fault=='mocap':a[6][3]=1
    elif fault=='root':a[7][2]=4
    elif fault=='weld':a[8][2]=-1
    elif fault=='geom':a[9][0]=4
    elif fault=='site':a[10][0]=-1
    elif fault=='qpos-span':a[2][1]=2
    elif fault=='qpos-overlap':a[2][1]=0
    elif fault=='body-span':a[5][2]=2
    elif fault=='joint-type':a[3][0]=4
    else:
        a[0]=8;a[2]=np.array([0,7]);a[3]=np.array([0,2]);a[4]=np.array([0,2,0,0]);a[5]=np.array([-1,0,-1,-1])
    with pytest.raises(ValueError,match='Kinematics'):validate_indices(*a)


def test_position_only_kinematics_selection_preserves_sequence(monkeypatch):
    from libero.gpu.host.backends.mjwarp import position as p
    m=NS(opt=NS(run_collision_detection=True));d=NS();calls=[]
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:pytest.fail('duplicate position'))
    monkeypatch.setattr(p.forward,'fwd_kinematics',lambda *a:pytest.fail('duplicate kinematics'))
    for owner,name in ((p.smooth,'crb'),(p.smooth,'tendon_armature'),(p.smooth,'factor_m'),(p.forward.collision_driver,'collision'),(p.forward.constraint,'make_constraint'),(p.smooth,'transmission')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    p.Position(m,d,kinematics=NS(selected='cuda',forward=lambda:calls.append('CUDA kinematics plus upstream tail'))).position()
    assert calls==['CUDA kinematics plus upstream tail','crb','tendon_armature','factor_m','collision','make_constraint','transmission']


def test_bad_selector():
    with pytest.raises(ValueError,match='kinematics must'):k.Kinematics(*model_data(),None,implementation='typo')
