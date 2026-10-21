"""Independent external-wrench selection, admission and postconstraint order."""
import ast,copy,inspect
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import external_wrench as e,postconstraint as p
from libero.gpu.host.runtime.dynamics.mjwarp_external_wrench import validate_topology


def model_data():
    m=NS(nv=3,nbody=4,ngeom=4,neq=0,ntendon=0,nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,nmaxcondim=6,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,cone=types.ConeType.ELLIPTIC))
    return m,NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','worlds','bodies','equality','tendon','flex','cone','dimension'))
def test_fallback_is_visible_before_optional_loading(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='worlds':d.nworld=0
    elif fault=='bodies':m.nbody=0
    elif fault=='equality':m.neq=1
    elif fault=='tendon':m.ntendon=1
    elif fault=='flex':m.nflexedge=1
    elif fault=='cone':m.opt.cone=19
    else:m.nmaxcondim=2
    calls=[];monkeypatch.setattr(e,'ExternalWrenchNumerics',lambda *a,**kw:pytest.fail('optional CUDA loading'))
    monkeypatch.setattr(e,'assemble_external',lambda *a:calls.append('upstream'))
    driver=e.ExternalWrench(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.identity['fallback'] and calls==['upstream']


def test_default_does_not_read_cuda_domain(monkeypatch):
    monkeypatch.setattr(e,'ExternalWrenchNumerics',lambda *a,**kw:pytest.fail('optional loading'))
    assert e.ExternalWrench(NS(),NS(),None).selected=='upstream'


def test_bad_selector_and_admission_owner():
    m,d=model_data()
    with pytest.raises(ValueError,match='external_wrench must'):e.ExternalWrench(m,d,None,implementation='typo')
    a=e.qualify(m,d,implementation='cuda')
    with pytest.raises(ValueError,match='owner'):e.ExternalWrench(m,NS(nworld=2),None,implementation='cuda',admission=a)
    with pytest.raises(ValueError,match='selector'):e.ExternalWrench(m,d,None,admission=a)


@pytest.mark.parametrize('cone',(types.ConeType.PYRAMIDAL,types.ConeType.ELLIPTIC))
@pytest.mark.parametrize('dimension',(1,3,4,6))
def test_both_cones_and_all_rigid_contact_dimensions_admitted(cone,dimension):
    m,d=model_data();m.opt.cone=cone;m.nmaxcondim=dimension
    assert e.qualify(m,d,implementation='cuda')['selected']=='cuda'


@pytest.mark.parametrize('fault',('parent','root','geom'))
def test_invalid_immutable_mapping_rejected(fault):
    parent=np.array([0,0,1,0]);root=np.array([0,1,1,3]);geom=np.array([0,1,2,3])
    if fault=='parent':parent[2]=2
    elif fault=='root':root[2]=2
    else:geom[0]=4
    with pytest.raises(ValueError,match='External wrench'):validate_topology(parent,root,geom)


def test_world_only_mapping_is_valid():validate_topology(np.array([0]),np.array([0]),np.array([0,0]))


@pytest.mark.parametrize('external',('upstream','cuda'))
@pytest.mark.parametrize('rne',('upstream','cuda'))
def test_independent_leaves_keep_one_prefix_and_one_suffix(monkeypatch,external,rne):
    m,d=model_data();calls=[];topology=NS()
    monkeypatch.setattr(p,'MotionTopology',lambda *a:topology)
    monkeypatch.setattr(p,'PostconstraintRNE',lambda *a:NS(identity={},enqueue=lambda:calls.append('cuda-rne')))
    monkeypatch.setattr(p.smooth,'rne_postconstraint',lambda *a:calls.append('whole-upstream'))
    monkeypatch.setattr(p,'upstream_suffix',lambda *a:calls.append('upstream-rne'))
    driver=p.Postconstraint(m,d,None,implementation=rne,external_wrench=NS(selected=external,identity={},enqueue=lambda:calls.append(external+'-external')))
    driver.enqueue()
    assert calls==(['whole-upstream'] if rne==external=='upstream' else [external+'-external',rne+'-rne'])
    assert driver.identity['external_wrenches']==external


def test_retained_suffix_matches_pinned_schedule():
    upstream=ast.parse(inspect.getsource(p.smooth.rne_postconstraint)).body[0]
    cut=next(i for i,n in enumerate(upstream.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_rne_cacc_world')
    original=ast.Module(body=copy.deepcopy(upstream.body[cut:]),type_ignores=[])
    adapted=ast.Module(body=ast.parse(inspect.getsource(p.upstream_suffix)).body[0].body[1:],type_ignores=[])
    class Unqualify(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='smooth':return ast.Name(id=node.attr,ctx=node.ctx)
            return self.generic_visit(node)
    assert ast.dump(original)==ast.dump(Unqualify().visit(adapted))
