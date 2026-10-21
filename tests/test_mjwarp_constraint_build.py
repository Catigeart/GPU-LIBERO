"""Dense rigid row construction admission and independent position selection."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import constraint_build as c,position as p
from libero.gpu.host.runtime.dynamics.mjwarp_constraint_build import validate_metadata


def model_data():
    m=NS(nv=3,nq=3,njnt=3,nbody=4,neq=0,ntendon=0,nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,is_sparse=False,
        flg_surfacevel=False,nmaxcondim=4,jnt_limited_ball_adr=NS(size=0),opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,cone=types.ConeType.ELLIPTIC))
    return m,NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','sparse','empty','equality','tendon','flex','ball','surface','cone','dimension'))
def test_fallback_precedes_optional_loading(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='sparse':m.is_sparse=True
    elif fault=='empty':m.nv=0
    elif fault=='equality':m.neq=1
    elif fault=='tendon':m.ntendon=1
    elif fault=='flex':m.nflex=1
    elif fault=='ball':m.jnt_limited_ball_adr.size=1
    elif fault=='surface':m.flg_surfacevel=True
    elif fault=='cone':m.opt.cone=9
    else:m.nmaxcondim=2
    monkeypatch.setattr(c,'ConstraintConstruction',lambda *a,**kw:pytest.fail('optional CUDA load'))
    calls=[];monkeypatch.setattr(c.constraint,'make_constraint',lambda *a:calls.append('upstream'))
    driver=c.ConstraintBuild(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.identity['fallback'] and calls==['upstream']


def test_default_skips_cuda_domain_and_bad_selector():
    assert c.ConstraintBuild(NS(),NS(),None).selected=='upstream'
    with pytest.raises(ValueError,match='constraint_build must'):c.ConstraintBuild(NS(),NS(),None,implementation='invalid')


def test_admission_owner_and_selector():
    m,d=model_data();admission=c.qualify(m,d,implementation='cuda')
    with pytest.raises(ValueError,match='owner'):c.ConstraintBuild(m,NS(nworld=2),None,implementation='cuda',admission=admission)
    with pytest.raises(ValueError,match='selector'):c.ConstraintBuild(m,d,None,admission=admission)


def metadata():
    return {k:np.array(v,np.int32) for k,v in dict(parent=[0,0,1,0],root=[0,1,1,3],geom_body=[0,1,2,3],joint_type=[3,2,3],
        joint_qpos=[0,1,2],joint_dof=[0,1,2],dof_body=[1,2,3],limited_joint=[0,1,2],ancestor=[[0,0,0,0],[1,0,0,0],[1,1,0,0],[0,0,1,0]]).items()}


def test_valid_ancestor_and_joint_metadata():validate_metadata(metadata(),3,3,4)


@pytest.mark.parametrize('fault',('parent','root','ancestor','padding','qpos','dof','limited','limited-type','dof-body'))
def test_bad_index_or_dependency_is_rejected(fault):
    a=metadata()
    if fault=='parent':a['parent'][2]=2
    elif fault=='root':a['root'][2]=2
    elif fault=='ancestor':a['ancestor'][2,0]=0
    elif fault=='padding':a['ancestor'][1,3]=1
    elif fault=='qpos':a['joint_qpos'][0]=3
    elif fault=='dof':a['joint_dof'][0]=3
    elif fault=='limited':a['limited_joint'][0]=1
    elif fault=='limited-type':a['joint_type'][0]=0
    else:a['dof_body'][0]=0
    with pytest.raises(ValueError):validate_metadata(a,3,3,4)


def test_constraint_only_selection_cannot_delegate_whole_position(monkeypatch):
    m,d=model_data();m.opt.run_collision_detection=True;calls=[]
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:pytest.fail('selected construction bypassed'))
    for owner,name,label in ((p.forward,'fwd_kinematics','kinematics'),(p.smooth,'crb','crb'),(p.smooth,'tendon_armature','tendon'),
            (p.smooth,'factor_m','factor'),(p.forward.collision_driver,'collision','collision'),(p.smooth,'transmission','transmission')):
        monkeypatch.setattr(owner,name,lambda *a,label=label:calls.append(label))
    monkeypatch.setattr(p.forward.constraint,'make_constraint',lambda *a:pytest.fail('duplicate upstream constraints'))
    p.Position(m,d,constraint_build=NS(selected='cuda',enqueue=lambda:calls.append('constraints'))).position()
    assert calls==['kinematics','crb','tendon','factor','collision','constraints','transmission']


def test_all_upstream_keeps_original_whole_boundary(monkeypatch):
    calls=[];monkeypatch.setattr(p.forward,'fwd_position',lambda *a:calls.append('whole'))
    p.Position(*model_data(),constraint_build=NS(selected='upstream',enqueue=lambda:pytest.fail('duplicate construction'))).position()
    assert calls==['whole']
