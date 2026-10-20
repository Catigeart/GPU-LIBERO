"""CPU-only mass-factor admission, scheduling and workspace lifetime contracts."""
from types import SimpleNamespace as NS
import pytest

pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import mass_factor as module


def model_data():
    m=NS(nv=16,nC=68,qLD_block_total=117,M_tiles=[NS(size=1,elemid=NS(size=0)),
        NS(size=6,elemid=NS(size=0)),NS(size=9,elemid=NS(size=81))],
        block_dim=NS(small_cholesky=64,cholesky_factorize=32,cholesky_factorize_solve=32),
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,run_collision_detection=True,timestep='h'))
    d=NS(nworld=2,M='M',qLD=NS(ndim=2,shape=(2,117)),qLDiagInv='D',qacc='public-qacc',qvel='velocity',efc=NS(Ma='Ma'))
    return m,d


@pytest.mark.parametrize('variant,reason',[
    ('sparse','unported sparse LDL mass region or factor layout'),('sleep','active-DOF compaction'),
    ('discrete','mass-factor caller requires Euler'),('implicit','mass-factor caller requires Euler'),
    ('scalar','unqualified scalar mass block size'),('tile','unqualified tile mass block size'),
    ('threads','unqualified factor block dimension'),('empty','no qualified mass blocks')])
def test_whole_caller_fallback_precedes_cuda_load(monkeypatch,variant,reason):
    m,d=model_data()
    if variant=='sparse':d.qLD.shape=(2,118)
    elif variant=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif variant=='discrete':m.opt.integrator=types.IntegratorType.DISCRETE
    elif variant=='implicit':m.opt.integrator=types.IntegratorType.IMPLICITFAST
    elif variant=='scalar':m.M_tiles[1].size=4
    elif variant=='tile':m.M_tiles[2].size=12
    elif variant=='threads':m.block_dim.cholesky_factorize=64
    else:m.M_tiles=[]
    monkeypatch.setattr(module,'MassFactorPlan',lambda *a:pytest.fail('CUDA load'))
    monkeypatch.setattr(module.wp,'empty_like',lambda *a:pytest.fail('GPU allocation'))
    calls=[]
    monkeypatch.setattr(module.forward,'fwd_position',lambda *a:calls.append('position'))
    for name in ('euler','implicit','discrete'):
        monkeypatch.setattr(module.forward,name,lambda *a,name=name:calls.append(name))
    driver=module.MassFactors(m,d,None,implementation='cuda');driver.position();driver.integrate()
    assert driver.identity['selected']=='upstream' and driver.identity['fallback']==reason
    assert calls==['position','discrete' if variant=='discrete' else 'implicit' if variant=='implicit' else 'euler']


def test_default_does_not_load_optional_product(monkeypatch):
    m,d=model_data();calls=[]
    monkeypatch.setattr(module,'MassFactorPlan',lambda *a:pytest.fail('CUDA load'))
    monkeypatch.setattr(module.forward,'fwd_position',lambda *a:calls.append('position'))
    monkeypatch.setattr(module.forward,'euler',lambda *a:calls.append('euler'))
    driver=module.MassFactors(m,d,None);driver.position();driver.integrate()
    assert calls==['position','euler'] and driver.identity['requested']=='upstream'


def fake_plan(monkeypatch,m,d,calls):
    bindings=[]
    def bind(matrix,factor,diagonal,solution=None,rhs=None):
        bindings.append((matrix,factor,diagonal,solution,rhs))
        return NS(enqueue=lambda:calls.append('factor-solve' if rhs is not None else 'factor'))
    monkeypatch.setattr(module,'MassFactorPlan',lambda *a:NS(bind=bind,identity={'test':'plan'}))
    count=[]
    def empty(a):
        item=NS(capacity=16,name=len(count));count.append(item);return item
    monkeypatch.setattr(module.wp,'empty_like',empty)
    return bindings,count


@pytest.mark.parametrize('collision',[False,True])
def test_position_order_preserves_factor_boundary(monkeypatch,collision):
    m,d=model_data();m.opt.run_collision_detection=collision;calls=[]
    fake_plan(monkeypatch,m,d,calls)
    for owner,name,label in ((module.forward,'fwd_kinematics','kinematics'),(module.smooth,'crb','crb'),
            (module.smooth,'tendon_armature','armature'),(module.forward.collision_driver,'collision','collision'),
            (module.forward.constraint,'make_constraint','constraints'),(module.smooth,'transmission','transmission')):
        monkeypatch.setattr(owner,name,lambda *a,label=label:calls.append(label))
    monkeypatch.setattr(module.forward,'fwd_position',lambda *a:pytest.fail('duplicate position'))
    monkeypatch.setattr(module.smooth,'factor_m',lambda *a:pytest.fail('upstream factor'))
    driver=module.MassFactors(m,d,None,implementation='cuda');driver.position()
    assert calls==['kinematics','crb','armature','factor']+(['collision'] if collision else [])+['constraints','transmission']


def test_euler_reuses_private_workspace_and_preserves_public_acceleration(monkeypatch):
    m,d=model_data();m.dof_damping='damping';m.dof_dampingpoly='poly';m.M_rownnz='nnz';m.M_rowadr='row'
    calls=[];bindings,allocated=fake_plan(monkeypatch,m,d,calls)
    monkeypatch.setattr(module.wp,'launch',lambda kernel,**kw:calls.append(('launch',kernel,kw)))
    monkeypatch.setattr(module.wp,'copy',lambda a,b:calls.append(('copy',a,b)))
    monkeypatch.setattr(module.forward,'_advance',lambda a,b,c:calls.append(('advance',c,b.qacc)))
    monkeypatch.setattr(module.forward,'euler',lambda *a:pytest.fail('duplicate integration'))
    monkeypatch.setattr(module.smooth,'factor_solve_i',lambda *a:pytest.fail('upstream factor solve'))
    driver=module.MassFactors(m,d,None,implementation='cuda');w=driver.euler_workspace
    assert len(allocated)==5 and bindings[0][:3]==(d.M,d.qLD,d.qLDiagInv)
    assert bindings[1]==(w.matrix,w.factor,w.diagonal,w.acceleration,d.efc.Ma)
    driver.integrate();driver.integrate()
    assert len(allocated)==5
    for iteration in (calls[:5],calls[5:]):
        assert iteration==[
            ('launch',module.forward._compute_damping_deriv,dict(dim=(2,16),inputs=['damping','poly','velocity'],outputs=[w.damping])),
            ('copy',w.matrix,'M'),
            ('launch',module.forward._euler_damp_qfrc,dict(dim=(2,16),inputs=['h','nnz','row',w.damping],outputs=[w.matrix])),
            'factor-solve',('advance',w.acceleration,'public-qacc')]
    assert d.qacc=='public-qacc' and driver.identity['euler_workspace_bytes']==80


@pytest.mark.parametrize('bit',[types.DisableBit.EULERDAMP,types.DisableBit.DAMPER])
def test_disabled_damping_uses_original_advance_without_factor_solve(monkeypatch,bit):
    m,d=model_data();m.opt.disableflags=bit;calls=[];bindings,allocated=fake_plan(monkeypatch,m,d,calls)
    monkeypatch.setattr(module.forward,'_advance',lambda a,b,c:calls.append(c))
    driver=module.MassFactors(m,d,None,implementation='cuda');driver.integrate()
    assert calls==['public-qacc'] and len(bindings)==1 and allocated==[]


def test_invalid_selector_rejected():
    with pytest.raises(ValueError,match='implementation'):module.MassFactors(*model_data(),None,implementation='typo')
