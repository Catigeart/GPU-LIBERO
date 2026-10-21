"""Global EPA scratch spans and packed face/witness residual qualification."""
from pathlib import Path
import sys
import numpy as np
import pytest
from libero.gpu.host.runtime.collision.mjwarp_convex import validate_epa_capacity
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation.mjwarp_convex_ccd import check,current_case
from validation.mjwarp_box_reference import box_separation


def test_box_sat_uses_actual_edge_face_normals():
    # Skew makes the true face normal (1,-.3,0), not column (1,0,0).
    R=np.array([[1.,.3,0.],[0.,1.,0.],[0.,0.,1.]])
    gap=box_separation(np.zeros(3),np.array([1.15,0.,0.]),R,np.eye(3),np.ones(3),np.full(3,.1))
    assert gap==pytest.approx(.02/np.sqrt(1.09))


def test_box_sat_distinguishes_small_overlap_from_separation():
    for shift in (-1.e-5,1.e-5):
        gap=box_separation(np.zeros(3),np.array([2.+shift,0.,0.]),np.eye(3),np.eye(3),np.ones(3),np.ones(3))
        assert gap==pytest.approx(shift,abs=1.e-14)


@pytest.mark.parametrize('requests,iterations,vertices,faces,horizon',[(0,0,10,6,1),(2,16,42,86,24),(4096,50,110,256,24)])
def test_admitted_workspace_domains(requests,iterations,vertices,faces,horizon):
    validate_epa_capacity(requests,iterations,vertices,faces,horizon)


@pytest.mark.parametrize('values',[(-1,0,10,6,1),(1,-1,10,6,1),(1,1,10,6,1),(1,0,10,5,1),(1,0,10,6,0),(2**30,0,10,6,24)])
def test_bad_workspace_spans(values):
    with pytest.raises(ValueError,match='EPA scratch'):validate_epa_capacity(*values)


def test_affine_witness_check_detects_bad_publication():
    c=dict(world=np.array([0]),pair=np.array([[0,1]]),size=np.ones((1,2,3)),mesh_vert=np.zeros((0,3)),geom_type=np.array([5,5]),geom_dataid=np.array([[-1,-1]]),mesh_vertadr=np.array([],int),mesh_vertnum=np.array([],int),mesh_graphadr=np.array([],int),mesh_graph=np.array([],int),position1=np.zeros((1,3)),position2=np.zeros((1,3)),epa_iterations=0,face_capacity=6,xmat=np.broadcast_to(np.eye(3),(2,2,3,3)),margin1=np.array([0.]),margin2=np.array([0.]))
    # Populate one known face: Minkowski triangle in the x=.2 plane.
    one=np.array([[.2,1,0],[.2,0,1],[.2,-1,-1]]);vertices=np.zeros((1,10,3));vertices[0,[0,2,4]]=one
    workspace=dict(vertices=vertices,face=np.array([[(1<<10)+(2<<20),0,0,0,0,0]]),projection=np.array([[[.2,0,0]]*6]),norm2=np.array([[.04]*6]))
    got=dict(dist_out=-.2,x1_out=np.array([.2,0,0]),x2_out=np.zeros(3),ncontact_out=1,needs_epa_out=1,face_index_out=0,index1_out=-1,index2_out=-1)
    # Avoid imposing unrelated mesh coordinates on this analytic box case.
    c['mesh_vert']=np.zeros((1,3))
    assert all(v['passed'] for v in check(c,0,got,workspace,'upstream')[0])
    got['x1_out']=np.array([.2,.1,0]);assert not all(v['passed'] for v in check(c,0,got,workspace,'upstream')[0])


def test_frozen_simplex_update_translates_witness_hulls_without_resizing():
    c=dict(use_frozen=True,size=np.ones((1,2,3)),position1=np.zeros((2,3),np.float32),position2=np.ones((2,3),np.float32),initial1=np.zeros((2,3),np.float32),initial2=np.ones((2,3),np.float32),frozen_simplex=np.ones((2,4,3),np.float32))
    for key in ('simplex1','simplex2'):c['frozen_'+key]=np.zeros((2,4,3),np.float32)
    for key in ('x1','x2'):c['frozen_'+key]=np.zeros((2,3),np.float32)
    result=current_case(c,True)
    np.testing.assert_array_equal(result['size'],c['size']);np.testing.assert_array_equal(result['frozen_simplex'],c['frozen_simplex'])
    np.testing.assert_allclose(result['frozen_simplex1'][0,0],[.013,-.017,.019],atol=1.e-9)
    assert not np.any(c['frozen_simplex1'])
