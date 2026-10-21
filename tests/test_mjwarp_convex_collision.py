"""Production pair-domain boundaries, original box route and scratch spans."""
import numpy as np
import pytest
from libero.gpu.host.runtime.collision.mjwarp_convex_collision import select_families,validate_workspace,FAMILIES


def counts(pairs):
    c=np.zeros(66,np.int32)
    for a,b in pairs:c[a*(22-a-1)//2+b]=1
    return c


def test_all_actual130_families_and_private_type_indices():
    active,indices,reason=select_families(counts(FAMILIES),FAMILIES)
    assert active==list(FAMILIES) and reason is None and len(set(indices))==7 and indices[4]==51


def test_primitive_box_route_excludes_convex_duplicate():
    active,indices,reason=select_families(counts(FAMILIES),FAMILIES,primitive_boxbox=True)
    assert (6,6) not in active and 51 not in indices and len(active)==6 and reason is None


def test_cylinder_cylinder_is_explicit_whole_boundary_fallback():
    active,_,reason=select_families(counts([(5,5)]),FAMILIES+((5,5),))
    assert active==[(5,5)] and 'additional convex pair' in reason


def test_general_multiccd_only_box_domain_is_independently_supported():
    assert select_families(counts([(6,6)]),FAMILIES,general_multiccd=True)[2] is None
    assert 'MULTICCD' in select_families(counts([(6,7)]),FAMILIES,general_multiccd=True)[2]


def test_absent_unqualified_pairs_do_not_force_fallback():
    assert select_families(counts(FAMILIES),FAMILIES+((5,5),))[2] is None


@pytest.mark.parametrize('table,ntypes',[(np.zeros(45),11),(np.zeros(66),10)])
def test_pair_count_layout_rejected(table,ntypes):
    with pytest.raises(ValueError,match='pair-count table'):select_families(table,FAMILIES,ntypes=ntypes)


@pytest.mark.parametrize('slots,iterations',[(0,0),(512,16),(1024,50)])
def test_dynamic_workspace_matches_original_capacity(slots,iterations):
    assert validate_workspace(slots,iterations)==(10+2*iterations,6+5*iterations)


@pytest.mark.parametrize('values',[(-1,0),(1,-1),(2**30,50)])
def test_workspace_index_overflow_rejected(values):
    with pytest.raises(ValueError,match='Convex workspace'):validate_workspace(*values)


def test_empty_primitive_residue_does_not_capture_upstream_noop(monkeypatch):
    from types import SimpleNamespace as NS
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import collision as c
    calls=[];ctx=NS();m=NS(ngeom=4,nflex=0,geom_pair_type_count=counts([(6,7)]),has_boxbox_margin=False,has_sdf_geom=False,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=types.DisableBit.MULTICCD,broadphase=types.BroadphaseType.NXN),callback=NS(contactfilter=None));d=NS(nworld=2,naconmax=8,ncollision=NS(zero_=lambda:None),nacon=NS(zero_=lambda:None))
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:ctx)
    monkeypatch.setattr(c,'ConvexContacts',lambda *a,**k:NS(enqueue=lambda:calls.append('cuda-convex'),identity={}))
    monkeypatch.setattr(c.collision_driver,'nxn_broadphase',lambda *a:calls.append('candidates'))
    monkeypatch.setattr(c.collision_convex,'convex_narrowphase',lambda *a:pytest.fail('duplicate convex'))
    monkeypatch.setattr(c.collision_primitive,'primitive_narrowphase',lambda *a:pytest.fail('empty residue'))
    driver=c.Collision(m,d,None,convex='cuda');driver.enqueue()
    assert calls==['candidates','cuda-convex'] and driver.convex_identity['selected']=='cuda'
