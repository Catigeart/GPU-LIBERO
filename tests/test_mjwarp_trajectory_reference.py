from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from validation.mjwarp_trajectory_reference import variation_checks


def test_quiet_reference_cannot_excuse_candidate_drift():
    a=dict(qvel=np.array([[.1],[0]],np.float32));b=dict(qvel=np.zeros((2,1),np.float32))
    rows=variation_checks(a,b,b)
    assert not rows[0]['within_original_repeat_order'] and rows[1]['within_original_repeat_order']


def test_original_repeat_order_is_per_world_and_candidate_independent():
    b=dict(qvel=np.zeros((2,1),np.float32));c=dict(qvel=np.array([[.0015],[.00002]],np.float32))
    a=dict(qvel=np.array([[.002],[.002]],np.float32));rows=variation_checks(a,b,c)
    assert rows[0]['within_original_repeat_order'] and not rows[1]['within_original_repeat_order']
    assert rows[0]['original_repeat_decade']==.01
    a['qvel'][0]=1.;changed=variation_checks(a,b,c)
    assert changed[0]['original_repeat_decade']==rows[0]['original_repeat_decade']
    assert not changed[0]['within_original_repeat_order']


def test_nonfinite_candidate_and_shape_mismatch_are_visible():
    b=dict(qvel=np.zeros((2,1),np.float32));a=dict(qvel=np.array([[np.nan],[0]],np.float32))
    assert not variation_checks(a,b,b)[0]['within_original_repeat_order']
    with pytest.raises(ValueError,match='shapes differ'):variation_checks(a,dict(qvel=np.zeros((1,1))),b)
