"""Plane selector boundaries, retained composition and cold mesh span checks."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import collision as c
from libero.gpu.host.runtime.collision.mjwarp_plane import qualify,validate_mesh


def model_data():
    return NS(geom_pair_type_count=(1,)*66,ngeom=4,nflex=0,npair=0,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,broadphase=types.BroadphaseType.NXN),callback=NS(contactfilter=None)),NS(nworld=2,naconmax=8)


@pytest.mark.parametrize('fault',('implicit','sleep','flex','empty','worlds'))
def test_fallback_before_plane_loading(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='flex':m.nflex=1
    elif fault=='empty':m.ngeom=0
    else:d.nworld=0
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:pytest.fail('optional allocation'))
    monkeypatch.setattr(c,'PlaneContacts',lambda *a,**kw:pytest.fail('optional loading'))
    driver=c.Collision(m,d,None,planes='cuda');assert driver.selected=='upstream' and driver.plane_identity['fallback']


def test_bad_selector():
    with pytest.raises(ValueError,match='plane_collision'):qualify(*model_data(),implementation='bad')


@pytest.mark.parametrize('broadphase',('upstream','cuda'))
def test_independent_plane_schedule_and_no_duplicate_family(monkeypatch,broadphase):
    m,d=model_data();calls=[];ctx=NS(collision_pair=NS(capacity=64),collision_pairid=NS(capacity=64),collision_worldid=NS(capacity=32))
    m.has_boxbox_margin=True;m.has_sdf_geom=False;d.nacon=NS(zero_=lambda:calls.append('contacts-zero'));d.ncollision=NS(zero_=lambda:calls.append('candidates-zero'))
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:ctx)
    monkeypatch.setattr(c,'BroadphaseNumerics',lambda *a,**k:NS(enqueue=lambda:calls.append('cuda-candidates'),identity={}))
    monkeypatch.setattr(c,'PlaneContacts',lambda *a,**k:NS(enqueue=lambda:calls.append('planes'),identity={}))
    monkeypatch.setattr(c.collision_driver,'nxn_broadphase',lambda *a:calls.append('upstream-candidates'))
    monkeypatch.setattr(c.collision_driver,'_narrowphase',lambda *a:pytest.fail('whole narrowphase duplicate'))
    monkeypatch.setattr(c.collision_driver,'collision',lambda *a:pytest.fail('whole collision duplicate'))
    def convex(m,d,owner,table):
        assert owner is ctx and (types.GeomType.BOX,types.GeomType.BOX) not in table;calls.append('convex')
    def primitive(m,d,owner,table):
        assert owner is ctx and all(a!=types.GeomType.PLANE for a,b in table) and (types.GeomType.BOX,types.GeomType.BOX) in table;calls.append('residual')
    monkeypatch.setattr(c.collision_convex,'convex_narrowphase',convex);monkeypatch.setattr(c.collision_primitive,'primitive_narrowphase',primitive)
    driver=c.Collision(m,d,None,broadphase=broadphase,planes='cuda');driver.enqueue()
    assert driver.selected=='cuda' and driver.plane_identity['selected']=='cuda' and driver.identity['selected']==broadphase
    assert calls==['candidates-zero','contacts-zero',broadphase+'-candidates','convex','planes','residual']


def mesh():
    a=dict(mesh_vertadr=np.array([0]),mesh_vertnum=np.array([3]),mesh_graphadr=np.array([-1]),mesh_graph=np.array([],int),mesh_polyadr=np.array([0]),mesh_polynum=np.array([1]),mesh_polyvertadr=np.array([0]),mesh_polyvertnum=np.array([3]),mesh_polyvert=np.array([0,1,2]),mesh_polymapadr=np.array([0,1,2]),mesh_polymapnum=np.array([1,1,1]),mesh_polymap=np.array([0,0,0]),mesh_vert=np.ones((3,3)),mesh_polynormal=np.ones((1,3)))
    return np.array([[-1,0]]),np.array([0,7]),a


def test_valid_mesh_empty_and_terminated_graph():
    d,t,a=mesh();validate_mesh(d,t,a)
    a['mesh_graphadr'][:]=0
    # Header[1] and trailing face data are not consumed by support traversal.
    a['mesh_graph']=np.array([3,777,0,3,6,0,1,2,1,2,-1,0,2,-1,0,1,-1,800,900]);validate_mesh(d,t,a)


@pytest.mark.parametrize('fault',('id','span','polygon','map','graph-header','graph-neighbor','graph-terminator'))
def test_malformed_mesh_rejected(fault):
    d,t,a=mesh()
    if fault=='id':d[0,1]=1
    elif fault=='span':a['mesh_vertnum'][0]=4
    elif fault=='polygon':a['mesh_polyvert'][0]=3
    elif fault=='map':a['mesh_polymap'][0]=1
    else:
        a['mesh_graphadr'][:]=0;a['mesh_graph']=np.array([3,777,0,3,6,0,1,2,1,2,-1,0,2,-1,0,1,-1])
        if fault=='graph-header':a['mesh_graph'][0]=30
        elif fault=='graph-neighbor':a['mesh_graph'][8]=5
        else:a['mesh_graph'][-1]=2
    with pytest.raises(ValueError,match='Plane mesh'):validate_mesh(d,t,a)
