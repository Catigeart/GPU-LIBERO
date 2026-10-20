"""Postconstraint replacement boundary, upstream force schedule and sensors."""
import ast
import copy
import inspect
from types import SimpleNamespace as NS
import pytest


def model_data():
    from mujoco_warp._src import types
    m=NS(nv=5,nbody=4,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    return m,NS(nworld=2)


@pytest.mark.parametrize('selected',('upstream','cuda'))
def test_postconstraint_keeps_external_before_suffix(monkeypatch,selected):
    from libero.gpu.host.backends.mjwarp import postconstraint as p
    m,d=model_data();calls=[];topology=NS()
    monkeypatch.setattr(p,'MotionTopology',lambda *a:topology)
    def binding(*a):
        assert a[0] is m and a[1] is d and a[-1] is topology
        return NS(enqueue=lambda:calls.append('cuda-rne'),identity={})
    monkeypatch.setattr(p,'PostconstraintRNE',binding)
    monkeypatch.setattr(p,'assemble_external',lambda a,b:calls.append('external'))
    monkeypatch.setattr(p.smooth,'rne_postconstraint',lambda a,b:calls.append('whole-upstream'))
    driver=p.Postconstraint(m,d,None,implementation=selected);driver.enqueue()
    assert calls==(['external','cuda-rne'] if selected=='cuda' else ['whole-upstream'])


def test_postconstraint_reuses_velocity_topology(monkeypatch):
    from libero.gpu.host.backends.mjwarp import postconstraint as p
    topology=NS();monkeypatch.setattr(p,'MotionTopology',lambda *a:pytest.fail('duplicate topology allocation'))
    monkeypatch.setattr(p,'PostconstraintRNE',lambda *a:NS(identity={},topology=a[-1]))
    driver=p.Postconstraint(*model_data(),None,implementation='cuda',topology=topology)
    assert driver.binding.topology is topology and driver.topology is topology


@pytest.mark.parametrize('fault',('sleep','implicit','empty','default'))
def test_fallback_and_default_do_not_load_optional_product(monkeypatch,fault):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import postconstraint as p
    m,d=model_data();calls=[]
    if fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='empty':m.nv=0
    monkeypatch.setattr(p,'MotionTopology',lambda *a:pytest.fail('optional topology allocation'))
    monkeypatch.setattr(p,'PostconstraintRNE',lambda *a:pytest.fail('optional binary loading'))
    monkeypatch.setattr(p.smooth,'rne_postconstraint',lambda *a:calls.append('upstream'))
    driver=p.Postconstraint(m,d,None,implementation='upstream' if fault=='default' else 'cuda');driver.enqueue()
    assert driver.selected=='upstream' and calls==['upstream']
    if fault!='default':assert driver.fallback


def test_external_prefix_is_the_pinned_upstream_schedule():
    from mujoco_warp._src import smooth
    from libero.gpu.host.backends.mjwarp.postconstraint_external import assemble_external
    upstream=ast.parse(inspect.getsource(smooth.rne_postconstraint)).body[0]
    # The suffix starts at the first top-level _rne_cacc_world call.
    cut=next(i for i,n in enumerate(upstream.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_rne_cacc_world')
    original=ast.Module(body=copy.deepcopy(upstream.body[1:cut]),type_ignores=[])
    adapted=ast.Module(body=ast.parse(inspect.getsource(assemble_external)).body[0].body,type_ignores=[])
    class Unqualify(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='smooth':return ast.Name(id=node.attr,ctx=node.ctx)
            if isinstance(node.value,ast.Name) and node.value.id=='types' and node.attr=='DisableBit':
                return ast.Name(id='DisableBit',ctx=node.ctx)
            return self.generic_visit(node)
    assert ast.dump(original)==ast.dump(Unqualify().visit(adapted))


@pytest.mark.parametrize('run,sensors_disabled,sensor_needs',[(a,b,c) for a in (False,True) for b in (False,True) for c in (False,True)])
def test_postconstraint_predicate_and_sensor_order(monkeypatch,run,sensors_disabled,sensor_needs):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import stages
    m,d=model_data();m.opt.run_rne_postconstraint=run;m.sensor_rne_postconstraint=sensor_needs
    m.opt.disableflags=int(types.DisableBit.SENSOR) if sensors_disabled else 0
    calls=[];p=stages.PhysicsStages(m,d,postconstraint=NS(enqueue=lambda:calls.append('post')))
    for method in ('actuation','acceleration','newton','integrate'):
        monkeypatch.setattr(p,method,lambda method=method:calls.append(method))
    def acc(a,b,*,skip_rne_postconstraint):
        assert a is m and b is d and skip_rne_postconstraint
        calls.append('sensor')
    monkeypatch.setattr(stages.sensor,'sensor_acc',acc)
    monkeypatch.setattr(stages.smooth,'rne_postconstraint',lambda *a:pytest.fail('duplicate upstream RNE'))
    p.step2()
    expected=['actuation','acceleration','newton']+(['post'] if run or (not sensors_disabled and sensor_needs) else [])+['sensor','integrate']
    assert calls==expected
