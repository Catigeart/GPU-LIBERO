"""Cold box-only EPA face admission and independent witness surface checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
from libero.gpu.host.runtime.collision.mjwarp_convex import validate_box_manifold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation.mjwarp_box_manifold import check


def arrays():
    return [np.array([0]),np.array([[0,1]]),np.array([6,6]),np.zeros(1),np.zeros(1),np.array([[(1<<10)+(2<<20)]]),np.array([0]),np.array([[0,1,2,3,4,5]])]


def test_valid_packed_features_and_empty_request():
    validate_box_manifold(*arrays())
    a=arrays()
    for i in (0,1,3,4,5,6,7):a[i]=a[i][:0]
    validate_box_manifold(*a)


@pytest.mark.parametrize('fault',('type','margin','face-negative','face-end','vertex-end','negative-feature','feature-end','odd-storage'))
def test_bad_packed_face_admission(fault):
    a=arrays()
    if fault=='type':a[2][0]=5
    elif fault=='margin':a[3][0]=.01
    elif fault=='face-negative':a[6][0]=-1
    elif fault=='face-end':a[6][0]=1
    elif fault=='vertex-end':a[5][0,0]=3+(1<<10)+(2<<20)
    elif fault=='negative-feature':a[7][0,0]=-1
    elif fault=='feature-end':a[7][0,0]=8
    else:a[7]=a[7][:,:5]
    with pytest.raises(ValueError,match='Box manifold'):validate_box_manifold(*a)


def test_surface_residual_detects_bad_witness():
    c=dict(size=np.ones((1,2,3)),world=np.array([0]),pair=np.array([[0,1]]),xmat=np.broadcast_to(np.eye(3),(2,2,3,3)),position1=np.zeros((1,3)),position2=np.array([[0.,0,1.5]]),frozen_x1=np.array([[0.,0,1.]]),frozen_x2=np.array([[0.,0,.5]]))
    got=dict(count=1,witness1=np.array([[0.,0,1.],[0,0,0],[0,0,0],[0,0,0]]),witness2=np.array([[0.,0,.5],[0,0,0],[0,0,0],[0,0,0]]),distance=np.array([-.5,0,0,0]))
    assert all(v['passed'] for v in check(c,0,got)[0])
    got['witness1'][0,0]=1.1;assert not all(v['passed'] for v in check(c,0,got)[0])
