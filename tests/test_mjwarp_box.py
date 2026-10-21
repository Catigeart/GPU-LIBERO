"""Box-family schedule and retained native CCD/primitive split."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import collision as c
from libero.gpu.host.runtime.collision.mjwarp_box import qualify


def model_data():
    return NS(geom_pair_type_count=(1,)*66,ngeom=4,nflex=0,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,broadphase=types.BroadphaseType.NXN),callback=NS(contactfilter=None)),NS(nworld=2,naconmax=8)


@pytest.mark.parametrize('fault',('implicit','sleep','flex','empty','worlds'))
def test_fallback_before_box_loading(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='flex':m.nflex=1
    elif fault=='empty':m.ngeom=0
    else:d.nworld=0
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:pytest.fail('optional context'))
    monkeypatch.setattr(c,'BoxContacts',lambda *a,**k:pytest.fail('optional load'))
    driver=c.Collision(m,d,None,boxes='cuda');assert driver.selected=='upstream' and driver.box_identity['fallback']


def test_bad_selector():
    with pytest.raises(ValueError,match='box_collision'):qualify(*model_data(),implementation='bad')


@pytest.mark.parametrize('primitive',('no','margin','flag'))
@pytest.mark.parametrize('planes',('upstream','cuda'))
def test_independent_box_route_no_duplicate_families(monkeypatch,primitive,planes):
    m,d=model_data();m.has_boxbox_margin=primitive=='margin';m.has_sdf_geom=False
    if primitive=='flag':m.opt.disableflags|=types.DisableBit.NATIVECCD
    d.nacon=NS(zero_=lambda:None);d.ncollision=NS(zero_=lambda:None);ctx=NS(collision_pair=NS(capacity=64),collision_pairid=NS(capacity=64),collision_worldid=NS(capacity=32));calls=[]
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:ctx)
    monkeypatch.setattr(c,'BoxContacts',lambda *a,**k:NS(enqueue=lambda:calls.append('boxes'),identity={}))
    monkeypatch.setattr(c,'PlaneContacts',lambda *a,**k:NS(enqueue=lambda:calls.append('planes'),identity={}))
    monkeypatch.setattr(c.collision_driver,'nxn_broadphase',lambda *a:calls.append('candidates'))
    monkeypatch.setattr(c.collision_driver,'_narrowphase',lambda *a:pytest.fail('whole narrowphase'))
    def convex(m,d,owner,table):
        assert owner is ctx and (((types.GeomType.BOX,types.GeomType.BOX) in table)==(primitive=='no'));calls.append('convex')
    def residual(m,d,owner,table):
        assert owner is ctx and not any(tuple(map(int,p)) in ((2,6),(3,6),(6,6)) for p in table)
        assert any(int(p[0])==0 for p in table)==(planes=='upstream');calls.append('residual')
    monkeypatch.setattr(c.collision_convex,'convex_narrowphase',convex);monkeypatch.setattr(c.collision_primitive,'primitive_narrowphase',residual)
    driver=c.Collision(m,d,None,planes=planes,boxes='cuda');driver.enqueue()
    assert calls==['candidates','convex']+(['planes'] if planes=='cuda' else [])+['boxes','residual']
    assert driver.box_identity['selected']=='cuda' and driver.plane_identity['selected']==planes
