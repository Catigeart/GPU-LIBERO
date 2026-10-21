"""Euler whole-caller fallback, private workspace and publication contracts."""
import ast,inspect
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import euler as e


def model_data():
    m=NS(nq=8,nv=7,njnt=2,nC=22,na=0,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0))
    d=NS(nworld=2,M='M',qLD='L',qLDiagInv='D',qacc='public acceleration',qvel='velocity',efc=NS(Ma='Ma'))
    f=NS(selected='upstream',euler_workspace=None,euler_binding=None,integrate=lambda:None)
    return m,d,f


@pytest.mark.parametrize('fault',('implicit','discrete','sleep','activation','empty'))
def test_fallback_before_loading_or_allocating(monkeypatch,fault):
    m,d,f=model_data();calls=[];f.integrate=lambda:calls.append('upstream')
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='discrete':m.opt.integrator=types.IntegratorType.DISCRETE
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='activation':m.na=1
    else:m.nv=0
    monkeypatch.setattr(e,'EulerNumerics',lambda *a,**k:pytest.fail('optional load'))
    monkeypatch.setattr(e.wp,'empty_like',lambda *a:pytest.fail('workspace allocation'))
    p=e.Euler(m,d,None,mass_factor=f,implementation='cuda');p.integrate()
    assert p.selected=='upstream' and p.fallback and calls==['upstream']


def test_default_preserves_existing_factor_integrator(monkeypatch):
    m,d,f=model_data();calls=[];f.integrate=lambda:calls.append('existing integrator')
    monkeypatch.setattr(e,'caller_unsupported_reason',lambda *a:pytest.fail('metadata read'))
    monkeypatch.setattr(e,'EulerNumerics',lambda *a,**k:pytest.fail('optional binary'))
    p=e.Euler(m,d,None,mass_factor=f);p.integrate()
    assert calls==['existing integrator'] and p.identity['owned_workspace_bytes']==0


def hooks(monkeypatch,calls):
    monkeypatch.setattr(e,'EulerNumerics',lambda *a,**k:NS(enqueue=lambda phase:calls.append(phase),identity={'qualified':True}))
    monkeypatch.setattr(e.wp,'copy',lambda dst,src:calls.append(('copy',dst,src)))
    monkeypatch.setattr(e,'advance_activation',lambda *a:calls.append('activation'))
    monkeypatch.setattr(e,'publish_advance',lambda *a:calls.append(('publish public',a[1].qacc)))


def test_existing_factor_workspace_is_shared_and_not_reallocated(monkeypatch):
    m,d,f=model_data();calls=[];hooks(monkeypatch,calls)
    f.selected='cuda';f.euler_workspace=NS(matrix='private M',factor='private L',diagonal='private D',acceleration='private acceleration',damping='derivative')
    f.euler_binding=NS(enqueue=lambda:calls.append('existing factor solve'))
    monkeypatch.setattr(e.wp,'empty_like',lambda *a:pytest.fail('duplicate workspace'))
    monkeypatch.setattr(e.smooth,'factor_solve_i',lambda *a:pytest.fail('duplicate factor solve'))
    p=e.Euler(m,d,None,mass_factor=f,implementation='cuda');p.integrate();p.integrate()
    assert p.workspace is f.euler_workspace and p.identity['workspace']=='mass-factor-shared'
    assert p.identity['owned_workspace_bytes']==0
    assert calls==[0,('copy','private M','M'),1,'existing factor solve','activation',2,3,('publish public','public acceleration')]*2


def test_upstream_factor_uses_one_retained_workspace(monkeypatch):
    m,d,f=model_data();calls=[];hooks(monkeypatch,calls);allocated=[]
    def allocate(a):
        result=NS(capacity=16);allocated.append(result);return result
    monkeypatch.setattr(e.wp,'empty_like',allocate)
    def solve(mm,dd,M,L,D,x,b):
        assert mm is m and dd is d and b=='Ma';calls.append(('solve',M,L,D,x))
    monkeypatch.setattr(e.smooth,'factor_solve_i',solve)
    p=e.Euler(m,d,None,mass_factor=f,implementation='cuda');w=p.workspace;p.integrate();p.integrate()
    assert len(allocated)==5 and p.identity['owned_workspace_bytes']==80 and d.qacc=='public acceleration'
    assert calls==[0,('copy',w.matrix,'M'),1,('solve',w.matrix,w.factor,w.diagonal,w.acceleration),'activation',2,3,('publish public','public acceleration')]*2


@pytest.mark.parametrize('bit',(types.DisableBit.EULERDAMP,types.DisableBit.DAMPER))
def test_disabled_damping_bypasses_factor_and_uses_public_acceleration(monkeypatch,bit):
    m,d,f=model_data();m.opt.disableflags=bit;calls=[];hooks(monkeypatch,calls)
    monkeypatch.setattr(e.wp,'empty_like',lambda *a:pytest.fail('disabled damping workspace'))
    def binding(mm,dd,lib,stream,*,workspace):
        assert dd.qacc=='public acceleration' and workspace is None
        return NS(enqueue=lambda phase:calls.append(phase))
    monkeypatch.setattr(e,'EulerNumerics',binding)
    p=e.Euler(m,d,None,mass_factor=f,implementation='cuda');p.integrate()
    assert calls==['activation',2,3,('publish public','public acceleration')]


def test_retained_activation_and_tail_match_pinned_ast():
    from mujoco_warp._src import forward
    from libero.gpu.host.backends.mjwarp import euler_other
    original=ast.parse(inspect.getsource(forward._advance)).body[0]
    for fn,body in ((euler_other.advance_activation,original.body[1:2]),(euler_other.publish_advance,original.body[5:])):
        candidate=ast.parse(inspect.getsource(fn)).body[0].body
        assert [ast.dump(n) for n in candidate]==[ast.dump(n) for n in body]


def test_stage_selects_only_one_integrator():
    from libero.gpu.host.backends.mjwarp.stages import PhysicsStages
    m,d,f=model_data();calls=[];f.integrate=lambda:pytest.fail('duplicate integration')
    p=PhysicsStages(m,d,mass_factor=f,euler=NS(integrate=lambda:calls.append('Euler')));p.integrate()
    assert calls==['Euler']


def test_selector_rejects_typo():
    m,d,f=model_data()
    with pytest.raises(ValueError,match='euler must'):e.Euler(m,d,None,mass_factor=f,implementation='typo')
