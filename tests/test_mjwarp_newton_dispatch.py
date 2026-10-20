"""CPU-only admission, fallback and graph-owner lifetime contracts."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

mjw = pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import newton


def model():
    return NS(nv=39, nv_pad=48, is_sparse=False,
        opt=NS(solver=types.SolverType.NEWTON, cone=types.ConeType.ELLIPTIC,
               integrator=types.IntegratorType.EULER, enableflags=0))


@pytest.mark.parametrize('implementation', ['cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton'])
@pytest.mark.parametrize('field,value,reason', [
    ('solver', types.SolverType.CG, 'solver is not Newton'),
    ('cone', types.ConeType.PYRAMIDAL, 'requires dense elliptic constraints'),
    ('integrator', types.IntegratorType.DISCRETE, 'discrete effective metric'),
    ('enableflags', types.EnableBit.SLEEP, 'active DOF compaction')])
def test_unsupported_domain_keeps_full_upstream_without_loading_cuda(monkeypatch, field, value, reason, implementation):
    m, d = model(), NS(njmax=64)
    setattr(m.opt, field, value)
    monkeypatch.setattr(newton, 'HessianPlan', lambda *a, **k: pytest.fail('unexpected CUDA load'))
    monkeypatch.setattr(newton, 'DirectionPlan', lambda *a, **k: pytest.fail('unexpected direction load'))
    monkeypatch.setattr(newton, 'ConstraintPlan', lambda *a, **k: pytest.fail('unexpected constraint load'))
    calls = []
    monkeypatch.setattr(newton.wp, 'ScopedStream', lambda s: nullcontext())
    monkeypatch.setattr(newton.solver, 'solve', lambda a, b: calls.append((a, b)))
    driver = newton.Newton(m, d, None, implementation=implementation)
    assert driver.identity == dict(requested=implementation, selected='upstream', fallback=reason)
    driver.solve()
    assert calls == [(m, d)]


def test_capture_owners_survive_scope_and_exception_without_cross_capture_leak():
    driver = newton.Newton(model(), NS(njmax=64), None)
    token = object()
    with pytest.raises(RuntimeError, match='capture failed'):
        with driver.capture_owners() as first:
            first.append(token)
            raise RuntimeError('capture failed')
    assert first == [token]
    with driver.capture_owners() as second:
        assert second == []
        with pytest.raises(RuntimeError, match='nested'):
            with driver.capture_owners():
                pass
    assert first == [token] and second == []


def test_noncanonical_or_compact_layout_has_visible_fallback():
    m, d = model(), NS(njmax=64)
    assert newton.unsupported_reason(m, d) is None
    m.nv_pad = 64
    assert newton.unsupported_reason(m, d) == 'unqualified padded NV'
    m.nv_pad, m.is_sparse = 48, True
    assert newton.unsupported_reason(m, d) == 'requires dense elliptic constraints'
    m.is_sparse, d.njmax = False, 0
    assert newton.unsupported_reason(m, d) == 'empty solver dimension'


def test_invalid_selector_is_not_silently_treated_as_upstream():
    with pytest.raises(ValueError, match='implementation'):
        newton.Newton(model(), NS(njmax=64), None, implementation='typo')


@pytest.mark.parametrize('implementation', ['cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton'])
def test_direction_unqualified_shape_or_threads_fall_back_before_loading(monkeypatch, implementation):
    m, d = model(), NS(njmax=64)
    m.block_dim = NS(update_gradient_cholesky=64, update_gradient_cholesky_blocked=64)
    monkeypatch.setattr(newton, 'HessianPlan', lambda *a, **k: pytest.fail('unexpected CUDA load'))
    driver = newton.Newton(m, d, None, implementation=implementation)
    assert driver.identity['fallback'] == 'unqualified direction block dimension'
    m.nv = 35
    driver = newton.Newton(m, d, None, implementation=implementation)
    assert driver.identity['fallback'] == 'unqualified direction NV'


@pytest.mark.parametrize('nv,parts', [(39,1),(57,3)])
def test_initialization_uses_selected_constraint_once_after_original_prefix(monkeypatch,nv,parts):
    calls=[]
    m=model();m.nv=nv
    d=NS(nworld=2,njmax=64,nefc='nefc',qacc='qacc',dof_cdof='cdof',solver_niter='niter',
         efc=NS(J_rownnz='rownnz',J_rowadr='rowadr',J_colind='colind',J='J',aref='aref',Ma='Ma'))
    ctx=NS(search_dot='search_dot',done='done',Jaref=NS(zero_=lambda:calls.append('zero')))
    driver=newton.Newton(m,d,None)
    monkeypatch.setattr(newton.solver,'init_context',lambda *a,**k:pytest.fail('duplicate upstream init'))
    monkeypatch.setattr(newton.solver,'_update_constraint',lambda *a,**k:pytest.fail('upstream constraint'))
    monkeypatch.setattr(newton.wp,'launch',lambda kernel,**k:calls.append(('launch',k['dim'])))
    monkeypatch.setattr(newton.solver,'_solve_init_jaref_kernel',lambda *a:calls.append(('jaref',a)) or 'kernel')
    monkeypatch.setattr(newton.solver,'_mul_m_compact_aware',lambda *a:calls.append('mass'))
    driver._initialize_context(ctx,NS(enqueue=lambda:calls.append('constraint')))
    assert calls==[('launch',2)]+(['zero'] if parts>1 else [])+[
        ('jaref',(False,nv,20 if parts>1 else 50,False)),('launch',(2,64,parts)),'mass','constraint']


@pytest.mark.parametrize('implementation', ['cuda_linesearch', 'cuda_newton'])
def test_linesearch_unqualified_threads_fall_back_before_loading(monkeypatch, implementation):
    m,d=model(),NS(njmax=64)
    m.block_dim=NS(update_gradient_cholesky=64,update_gradient_cholesky_blocked=32,linesearch_iterative=64)
    monkeypatch.setattr(newton,'HessianPlan',lambda *a,**k:pytest.fail('unexpected CUDA load'))
    driver=newton.Newton(m,d,None,implementation=implementation)
    assert driver.identity==dict(requested=implementation,selected='upstream',
                                fallback='unqualified linesearch block dimension')


def test_iteration_uses_selected_linesearch_before_constraint_and_gradient(monkeypatch):
    driver=newton.Newton(model(),NS(nworld=2),None)
    driver.model.opt.warn_overflow=0
    driver.model.opt.tolerance='tolerance';driver.model.opt.iterations=100
    driver.model.stat=NS(meaninertia='inertia')
    driver.data.solver_niter='niter';driver.data.overflow='overflow'
    ctx=NS(**{k:k for k in ('grad_dot','newton_decrement','improvement','alpha','done')})
    calls=[]
    monkeypatch.setattr(newton.solver,'_linesearch',lambda *a:pytest.fail('duplicate upstream linesearch'))
    monkeypatch.setattr(driver,'_constraint',lambda *a:calls.append('constraint'))
    monkeypatch.setattr(driver,'_gradient',lambda *a:calls.append('gradient'))
    monkeypatch.setattr(newton.solver,'_solve_done',lambda *a:'stopping')
    monkeypatch.setattr(newton.wp,'launch',lambda *a,**k:calls.append('stopping'))
    driver._iteration(ctx,None,'nsolving',linesearch=NS(enqueue=lambda:calls.append('linesearch')))
    assert calls==['linesearch','constraint','gradient','stopping']


def test_complete_iteration_has_no_upstream_numerical_stopping_call(monkeypatch):
    driver=newton.Newton(model(),NS(nworld=2),None)
    calls=[]
    monkeypatch.setattr(driver,'_constraint',lambda *a:calls.append('constraint'))
    monkeypatch.setattr(driver,'_gradient',lambda *a:calls.append('gradient'))
    monkeypatch.setattr(newton.solver,'_solve_done',lambda *a:pytest.fail('unexpected upstream stopping'))
    monkeypatch.setattr(newton.solver,'_linesearch',lambda *a:pytest.fail('unexpected upstream line search'))
    driver._iteration(NS(),None,None,linesearch=NS(enqueue=lambda:calls.append('linesearch')),
                      solver_state=NS(stop=lambda:calls.append('stopping')))
    assert calls==['linesearch','constraint','gradient','stopping']


def test_complete_solve_initializes_once_without_upstream_prefix(monkeypatch):
    m,d=model(),NS(njmax=64,nworld=2)
    m.opt.iterations=0;m.opt.noslip_iterations=0;m.opt.graph_conditional=False
    driver=newton.Newton(m,d,None)
    driver.selected='cuda_newton'
    calls=[];ctx=object();counter=object()
    state=NS(initialize=lambda:calls.append('initialize'),
             bind_stopping=lambda c:calls.append(('bind-counter',c)))
    token=object()
    driver.plan=driver.direction_plan=driver.constraint_plan=driver.linesearch_plan=NS(bind=lambda c:token)
    driver.state_plan=NS(bind=lambda c:state)
    monkeypatch.setattr(newton.wp,'ScopedStream',lambda s:nullcontext())
    monkeypatch.setattr(newton.wp,'launch',lambda *a,**k:pytest.fail('upstream numerical initialization'))
    monkeypatch.setattr(newton.wp,'full',lambda **k:counter)
    monkeypatch.setattr(newton.solver,'_create_solver_context',lambda *a:ctx)
    monkeypatch.setattr(driver,'_initialize_context',lambda *a:pytest.fail('upstream prefix'))
    monkeypatch.setattr(driver,'_constraint',lambda *a:calls.append('constraint'))
    monkeypatch.setattr(driver,'_gradient',lambda *a:calls.append('gradient'))
    with driver.capture_owners() as owners:
        owner=driver.solve()
    assert calls==['initialize','constraint','gradient',('bind-counter',counter)]
    assert owner==(ctx,token,counter,token,token,token,state)
    assert owners==[owner]
