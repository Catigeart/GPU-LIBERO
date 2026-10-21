"""Fluid admission, DFS projection domain and independent passive dispatch."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import fluid as f,passive as p
from libero.gpu.host.runtime.dynamics.mjwarp_fluid import validate_topology


def model_data():
    m=NS(nv=2,nq=2,njnt=2,nbody=4,ntendon=0,nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,has_fluid=True,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0),body_fluid_ellipsoid=NS(numpy=lambda:np.zeros(4,bool)))
    return m,NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','worlds','nv','nq','njnt','nbody','tendon','flex','disabled','ellipsoid'))
def test_visible_fallback_before_optional_load(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='worlds':d.nworld=0
    elif fault=='tendon':m.ntendon=1
    elif fault=='flex':m.nflex=1
    elif fault=='disabled':m.has_fluid=False
    elif fault=='ellipsoid':m.body_fluid_ellipsoid=NS(numpy=lambda:np.ones(4,bool))
    else:setattr(m,fault,0)
    monkeypatch.setattr(f,'FluidNumerics',lambda *a,**kw:pytest.fail('optional CUDA load'))
    calls=[];monkeypatch.setattr(f.passive,'_fluid',lambda *a:calls.append('upstream'))
    driver=f.Fluid(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.identity['fallback'] and calls==['upstream']


def test_default_skips_admission_reads(monkeypatch):
    m,d=model_data();m.body_fluid_ellipsoid=NS(numpy=lambda:pytest.fail('cold read'))
    monkeypatch.setattr(f,'FluidNumerics',lambda *a,**kw:pytest.fail('optional load'))
    assert f.Fluid(m,d,None).selected=='upstream'


def test_bad_selector_and_owner():
    m,d=model_data()
    with pytest.raises(ValueError,match='fluid must'):f.Fluid(m,d,None,implementation='typo')
    a=f.qualify(m,d,implementation='cuda')
    with pytest.raises(ValueError,match='owner'):f.Fluid(m,NS(nworld=2),None,implementation='cuda',admission=a)


def topology():return [np.array(v,np.int32) for v in ([0,0,1,1,0],[0,1,1,1,4],[5,3,1,1,1],[1,2,3,4])]

def test_valid_dfs_projection_topology():validate_topology(*topology())


@pytest.mark.parametrize('fault',('cycle','root','subtree','dof','interleaved'))
def test_bad_topology_rejected(fault):
    a=topology()
    if fault=='cycle':a[0][2]=2
    elif fault=='root':a[1][2]=2
    elif fault=='subtree':a[2][1]=2
    elif fault=='dof':a[3][0]=5
    else:a=[np.array(v,np.int32) for v in ([0,0,0,1],[0,1,2,1],[4,2,1,1],[1,2,3])]
    with pytest.raises(ValueError,match='Fluid'):validate_topology(*a)


def test_fluid_only_preserves_upstream_joint_sum_and_callback(monkeypatch):
    m,d=model_data();calls=[];m.callback=NS(passive=lambda *a:calls.append('callback'));m.flg_adhesion=False
    for key in ('qpos_spring','jnt_type','jnt_qposadr','jnt_dofadr','jnt_stiffness','jnt_stiffnesspoly','dof_damping','dof_dampingpoly','jnt_actgravcomp','dof_jntid'):setattr(m,key,None)
    for key in ('qpos','qvel','qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_adhesion','qfrc_passive'):setattr(d,key,None)
    monkeypatch.setattr(p.passive,'passive',lambda *a:pytest.fail('duplicate whole passive'))
    monkeypatch.setattr(p.passive,'_qfrc_passive_kernel',lambda *a:'sum')
    monkeypatch.setattr(p.wp,'launch',lambda kernel,*a,**kw:calls.append('sum' if kernel=='sum' else 'joint'))
    fluid=NS(selected='cuda',enqueue=lambda:calls.append('fluid'))
    def other(m,d,*,fluid,gravity=None):calls.append('other-prefix');fluid.enqueue();calls.append('other-tail')
    monkeypatch.setattr(p,'other_forces',other)
    driver=p.Passive(m,d,None,fluid=fluid);assert driver.selected=='upstream' and driver.uses_cuda
    driver.enqueue();assert calls==['joint','other-prefix','fluid','other-tail','sum','callback']


def test_fluid_only_both_disabled_keeps_adhesion_and_workspace(monkeypatch):
    m,d=model_data();m.opt.disableflags=types.DisableBit.SPRING|types.DisableBit.DAMPER;m.callback=NS(passive=lambda *a:pytest.fail('callback'))
    calls=[]
    for key in ('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive'):setattr(d,key,NS(zero_=lambda key=key:calls.append(key)))
    d.qfrc_adhesion=NS(zero_=lambda:pytest.fail('adhesion hold'))
    fluid=NS(selected='cuda',enqueue=lambda:pytest.fail('workspace must be held'))
    p.Passive(m,d,None,fluid=fluid).enqueue()
    assert calls==['qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive']


def test_fluid_only_prevents_whole_velocity_delegation(monkeypatch):
    from libero.gpu.host.backends.mjwarp import velocity as v
    from test_mjwarp_motion import model_data as motion_model
    m,d=motion_model();calls=[]
    monkeypatch.setattr(v.forward,'fwd_velocity',lambda *a:pytest.fail('whole velocity'))
    monkeypatch.setattr(v.wp,'launch',lambda *a,**kw:None)
    for name in ('com_vel','rne','tendon_bias'):monkeypatch.setattr(v.smooth,name,lambda *a,name=name:calls.append(name))
    driver=NS(selected='upstream',uses_cuda=True,enqueue=lambda:calls.append('fluid passive'),identity={})
    v.Velocity(m,d,None,passive_driver=driver).enqueue()
    assert calls==['com_vel','fluid passive','rne','tendon_bias']
