"""Cold GJK controls, exact cache formats and independent geometric checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
from libero.gpu.host.runtime.collision.mjwarp_convex import validate_gjk_queries
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation.mjwarp_convex_gjk import check


@pytest.mark.parametrize('iterations,discrete,tolerance,cutoff',[(0,0,1.e-6,0),(1,1,1.e-8,.2),(50,0,1.e-6,1.e32)])
def test_valid_controls(iterations,discrete,tolerance,cutoff):
    validate_gjk_queries(np.array([0]),np.array([[0,1]]),*(np.array([v]) for v in (iterations,discrete,tolerance,cutoff)))


@pytest.mark.parametrize('fault',('pair-shape','vector-shape','negative-iterations','invalid-discrete','zero-tolerance','nan-tolerance','negative-cutoff','infinite-cutoff'))
def test_invalid_controls(fault):
    world=np.array([0]);pair=np.array([[0,1]]);iterations=np.array([50]);discrete=np.array([0]);tolerance=np.array([1.e-6]);cutoff=np.array([.1])
    if fault=='pair-shape':pair=pair.ravel()
    elif fault=='vector-shape':iterations=np.array([],int)
    elif fault=='negative-iterations':iterations[0]=-1
    elif fault=='invalid-discrete':discrete[0]=2
    elif fault=='zero-tolerance':tolerance[0]=0
    elif fault=='nan-tolerance':tolerance[0]=np.nan
    elif fault=='negative-cutoff':cutoff[0]=-.1
    else:cutoff[0]=np.inf
    with pytest.raises(ValueError,match='GJK query'):validate_gjk_queries(world,pair,iterations,discrete,tolerance,cutoff)


def case():
    return dict(world=np.array([0]),pair=np.array([[0,1]]),size=np.array([[[1,1,1],[1,1,1]]],float),geom_type=np.array([2,2]),geom_dataid=np.array([[-1,-1]]),mesh_vertadr=np.array([],int),mesh_vertnum=np.array([],int),mesh_graphadr=np.array([],int),mesh_graph=np.array([],int),position1=np.array([[0,0,0.]]),position2=np.array([[3,0,0.]]),xpos=np.zeros((2,2,3)),xmat=np.broadcast_to(np.eye(3),(2,2,3,3)),initial1=np.array([[0,0,0.]]),initial2=np.array([[3,0,0.]]),margin1=np.array([0.]),margin2=np.array([0.]),cutoff=np.array([0.]))


def test_cutoff_and_immediate_witness_contracts_are_distinct():
    c=case();got=dict(dim_out=0,dist_out=np.float32(1.e30),x1_out=np.zeros(3),x2_out=np.zeros(3),separated_out=1,index1_out=-1,index2_out=-1)
    assert all(v['passed'] for v in check(c,0,got,'upstream')[0])
    got['x1_out']=np.ones(3);assert not all(v['passed'] for v in check(c,0,got,'upstream')[0])
    got.update(dist_out=3.,x1_out=c['initial1'][0],x2_out=c['initial2'][0]);assert all(v['passed'] for v in check(c,0,got,'upstream')[0])
    got['x2_out']=np.array([2.,0,0]);assert not all(v['passed'] for v in check(c,0,got,'upstream')[0])


def test_real_tuned_direction_counterexample_is_not_a_separating_plane():
    import json
    from validation.mjwarp_box_reference import box_separation
    f=json.loads((Path(__file__).parent/'data/gjk_tuned_direction.json').read_text())
    a,b=np.asarray(f['position1']),np.asarray(f['position2'])
    rotations,sizes=np.asarray(f['rotations']),np.asarray(f['sizes'])
    assert box_separation(a,b,*rotations,*sizes)<-f['tolerance']
    direction=np.asarray(f['previous_invalid_direction'])
    actual=np.dot(direction,a-b)-sum(np.dot(abs(r.T@direction),s) for r,s in zip(rotations,sizes))
    assert f['previous_claimed_lower']>0
    assert actual < -f['tolerance']
