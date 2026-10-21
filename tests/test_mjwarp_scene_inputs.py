"""Fixture parameters come from original prepared input, never resampling."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from libero.gpu.host.backends.mjwarp.scene_inputs import apply


def values(monkeypatch):
    names=['world','fixture','object'];monkeypatch.setattr(mujoco,'mj_id2name',lambda m,k,i:names[i]);monkeypatch.setattr(mujoco,'mj_name2id',lambda m,k,name:names.index(name) if name in names else -1)
    m=NS(nbody=3,body_pos=np.zeros((3,3)),body_quat=np.tile([1.,0,0,0],(3,1)))
    data=dict(body_names=np.array(names),body_pos=np.array([[0,0,0],[.1,.2,.905],[0,0,0]]),body_quat=np.array([[1.,0,0,0],[0,0,0,1.],[1,0,0,0]]))
    for key,v in dict(body_parentid=[0,0,0],jnt_type=[0],jnt_bodyid=[2],jnt_qposadr=[0],jnt_dofadr=[0],dof_bodyid=[2]*6).items():setattr(m,key,np.array(v));data[key]=np.array(v)
    shell=NS(fixtures_dict={'fixture':NS(root_body='fixture')},placement_initializer=NS(sample=lambda:pytest.fail('resampling original fixed parameters')))
    return shell,m,data


def test_only_declared_fixed_root_is_written_exactly(monkeypatch):
    shell,m,data=values(monkeypatch);data['body_pos'][2]=[9,9,9];apply(shell,m,data)
    assert np.array_equal(m.body_pos[1],data['body_pos'][1]) and np.array_equal(m.body_quat[1],data['body_quat'][1])
    assert np.array_equal(m.body_pos[2],[0,0,0])


@pytest.mark.parametrize('key',['body_parentid','jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','dof_bodyid'])
def test_index_space_changes_fail_before_pose_writes(monkeypatch,key):
    shell,m,data=values(monkeypatch);data[key]=data[key]+1
    with pytest.raises(ValueError,match='mapping'):apply(shell,m,data)
    assert not m.body_pos.any()


def test_body_name_reordering_is_not_silently_reinterpreted(monkeypatch):
    shell,m,data=values(monkeypatch);data['body_names']=data['body_names'][[0,2,1]]
    with pytest.raises(ValueError,match='names'):apply(shell,m,data)
