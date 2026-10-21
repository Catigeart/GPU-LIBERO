"""Convex request ownership and distinct exhaustive/graph cache formats."""
import numpy as np
import pytest
from libero.gpu.host.runtime.collision.mjwarp_convex import validate_requests


def metadata():
    types=np.array([2,3,4,5,6,7,7],np.int32);ids=np.array([[-1,-1,-1,-1,-1,0,1]],np.int32)
    return types,ids,np.array([13,30]),np.array([4,12]),np.array([-1,0]),np.array([8,777])


def test_valid_cache_modes_and_ignored_primitive_cache():
    validate_requests(np.array([0,1,0,1]),np.array([0,5,6,6]),np.array([999,14,7,-1]),*metadata(),2)


@pytest.mark.parametrize('fault',('world','geom','shape','type','negative-cache','missing-mesh','exhaustive-local','exhaustive-end','graph-absolute','graph-end'))
def test_invalid_requests_rejected(fault):
    world=np.array([0]);geom=np.array([5]);cached=np.array([14]);types,ids,adr,num,gadr,graph=metadata()
    if fault=='world':world[0]=2
    elif fault=='geom':geom[0]=7
    elif fault=='shape':cached=np.array([],int)
    elif fault=='type':types[5]=0
    elif fault=='negative-cache':cached[0]=-2
    elif fault=='missing-mesh':ids[0,5]=-1
    elif fault=='exhaustive-local':cached[0]=1
    elif fault=='exhaustive-end':cached[0]=17
    elif fault=='graph-absolute':geom[0]=6;cached[0]=30
    else:geom[0]=6;cached[0]=8
    with pytest.raises(ValueError,match='Convex'):validate_requests(world,geom,cached,types,ids,adr,num,gadr,graph,2)


def test_independent_world_broadcast_selects_current_mesh_cache_format():
    types,ids,adr,num,gadr,graph=metadata();ids=np.repeat(ids,2,axis=0);ids[1,6]=0
    validate_requests(np.array([1]),np.array([6]),np.array([14]),types,ids,adr,num,gadr,graph,2)
    with pytest.raises(ValueError,match='absolute'):validate_requests(np.array([1]),np.array([6]),np.array([7]),types,ids,adr,num,gadr,graph,2)


def test_empty_requests_valid():
    empty=np.empty(0,np.int32);validate_requests(empty,empty,empty,*metadata(),2)


def test_zero_support_components_still_require_an_indexed_corner():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from validation.mjwarp_convex_reference import support,check
    c=dict(world=np.array([0]),geom=np.array([0]),geom_type=np.array([6]),
           size=np.array([[[.5,.6,.025]]]),xpos=np.zeros((2,1,3)),
           xmat=np.broadcast_to(np.eye(3),(2,1,3,3)),direction=np.array([[0.,0.,1.]]),request_margin=np.zeros(1))
    ref=support(c,0)
    # Both points maximize the support projection. Only the corner matches
    # vertex4 (-x,-y,+z), which downstream EPA uses to recover edges/faces.
    interior=np.array([0.,0.,.025]);corner=np.array([-.5,-.6,.025])
    assert all(v['passed'] for v in check(c,0,interior,-1,4,ref))
    rows=check(c,0,interior,-1,4,ref,require_indexed_corner=True)
    assert [v['name'] for v in rows if not v['passed']]==['box support equals its indexed corner']
    assert all(v['passed'] for v in check(c,0,corner,-1,4,ref,require_indexed_corner=True))
