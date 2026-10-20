"""Cold joint ownership and exact passive caller/remaining-force boundaries."""
import ast,copy,inspect
from types import SimpleNamespace as NS
import numpy as np
import pytest
from libero.gpu.host.runtime.dynamics.mjwarp_passive import validate_joints


def test_all_joint_coordinate_spans_have_one_owner():
    validate_joints(13,11,np.array([0,1,2,3]),np.array([0,7,11,12]),np.array([0,6,9,10]),np.array([0]*6+[1]*3+[2,3]))


@pytest.mark.parametrize('fault',('type','q-span','d-span','owner','overlap','missing'))
def test_invalid_joint_metadata_rejected(fault):
    a=[np.array(v) for v in ([0,1,2,3],[0,7,11,12],[0,6,9,10],[0]*6+[1]*3+[2,3])]
    if fault=='type':a[0][0]=4
    elif fault=='q-span':a[1][2]=14
    elif fault=='d-span':a[2][2]=12
    elif fault=='owner':a[3][1]=1
    elif fault=='overlap':a[1][3]=11
    else:a[1][0]=1
    with pytest.raises(ValueError,match='Passive'):validate_joints(13,11,*a)


def model_data():
    from mujoco_warp._src import types
    m=NS(nv=11,nq=13,njnt=4,ntendon=0,nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0),callback=NS(passive=None))
    return m,NS(nworld=2)


@pytest.mark.parametrize('selected',('upstream','cuda'))
def test_passive_order_and_callback(monkeypatch,selected):
    from libero.gpu.host.backends.mjwarp import passive as p
    m,d=model_data();calls=[];m.callback.passive=lambda *a:calls.append('callback')
    binding=NS(flags=dict(disable_spring=False,disable_damper=False),enqueue=lambda phase:calls.append(('cuda',phase)),identity={})
    monkeypatch.setattr(p,'PassiveForces',lambda *a:binding)
    monkeypatch.setattr(p,'other_forces',lambda *a:calls.append('other'))
    monkeypatch.setattr(p.passive,'passive',lambda *a:calls.append('whole-upstream'))
    p.Passive(m,d,None,implementation=selected).enqueue()
    assert calls==([('cuda',0),'other',('cuda',1),'callback'] if selected=='cuda' else ['whole-upstream'])


def test_both_disabled_keeps_pinned_clear_and_early_return(monkeypatch):
    from libero.gpu.host.backends.mjwarp import passive as p
    m,d=model_data();calls=[];m.callback.passive=lambda *a:pytest.fail('callback after early return')
    for key in ('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive'):
        setattr(d,key,NS(zero_=lambda key=key:calls.append(key)))
    d.qfrc_adhesion=NS(zero_=lambda:pytest.fail('adhesion must be held'))
    monkeypatch.setattr(p,'PassiveForces',lambda *a:NS(flags=dict(disable_spring=True,disable_damper=True)))
    monkeypatch.setattr(p,'other_forces',lambda *a:pytest.fail('other after early return'))
    p.Passive(m,d,None,implementation='cuda').enqueue()
    assert calls==['qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive']


@pytest.mark.parametrize('fault',('tendon','flex','sleep','implicit','empty','default'))
def test_optional_binary_not_loaded_for_default_or_fallback(monkeypatch,fault):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import passive as p
    m,d=model_data();calls=[]
    if fault=='tendon':m.ntendon=1
    elif fault=='flex':m.nflex=1
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='empty':m.nv=0
    monkeypatch.setattr(p,'PassiveForces',lambda *a:pytest.fail('optional binary load'))
    monkeypatch.setattr(p.passive,'passive',lambda *a:calls.append('upstream'))
    driver=p.Passive(m,d,None,implementation='upstream' if fault=='default' else 'cuda');driver.enqueue()
    assert driver.selected=='upstream' and calls==['upstream']
    if fault!='default':assert driver.fallback


def test_retained_rigid_force_schedule_is_upstream_ast():
    from mujoco_warp._src import passive
    from libero.gpu.host.backends.mjwarp.passive_other import other_forces
    tree=ast.parse(inspect.getsource(passive.passive)).body[0].body
    start=next(i for i,n in enumerate(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gravity_enabled' for t in n.targets))
    fluid=next(i for i,n in enumerate(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='has_fluid')
    end=next(i for i,n in enumerate(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args and isinstance(n.value.args[0],ast.Call) and isinstance(n.value.args[0].func,ast.Name) and n.value.args[0].func.id=='_qfrc_passive_kernel')
    expected=ast.Module(body=copy.deepcopy(tree[start:start+3]+tree[fluid:end]),type_ignores=[])
    actual=ast.Module(body=ast.parse(inspect.getsource(other_forces)).body[0].body,type_ignores=[])
    class Names(ast.NodeTransformer):
        def visit_Attribute(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='passive':return ast.Name(id=n.attr,ctx=n.ctx)
            if isinstance(n.value,ast.Name) and n.value.id=='types' and n.attr=='DisableBit':return ast.Name(id='DisableBit',ctx=n.ctx)
            return self.generic_visit(n)
    assert ast.dump(expected)==ast.dump(Names().visit(actual))


def test_passive_only_selection_prevents_whole_velocity_duplication(monkeypatch):
    from libero.gpu.host.backends.mjwarp import velocity as v
    from test_mjwarp_motion import model_data as velocity_model_data
    m,d=velocity_model_data();calls=[]
    monkeypatch.setattr(v.forward,'fwd_velocity',lambda *a:pytest.fail('whole upstream velocity duplicates passive'))
    monkeypatch.setattr(v.wp,'launch',lambda *a,**kw:None)
    monkeypatch.setattr(v.smooth,'com_vel',lambda *a:calls.append('com'))
    monkeypatch.setattr(v.smooth,'rne',lambda *a:calls.append('bias'))
    monkeypatch.setattr(v.smooth,'tendon_bias',lambda *a:calls.append('tendon-bias'))
    monkeypatch.setattr(v.passive,'passive',lambda *a:pytest.fail('duplicate passive'))
    driver=NS(selected='cuda',enqueue=lambda:calls.append('cuda-passive'),identity={})
    velocity=v.Velocity(m,d,None,passive_driver=driver);velocity.enqueue()
    assert calls==['com','cuda-passive','bias','tendon-bias'] and velocity.topology is None
