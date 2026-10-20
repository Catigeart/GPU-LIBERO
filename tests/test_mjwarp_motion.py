"""Cold motion topology safety; common ancestors have one output owner."""
import numpy as np
import pytest
from libero.gpu.host.runtime.dynamics.mjwarp_motion import validate_motion


def metadata():
    a=[np.array(v,np.int32) for v in ([0,0,1,1], [0,1,1,1], [-1,0,1,2], [0,1,3,1],
        [-1,0,1,4], [3,1,2], [0,1,4], [1,2,2,2,3], [1,2,1,3], [0,2,4])]
    a.append([np.array(v,np.int32) for v in ([0],[1],[2,3])])
    return a


def test_shared_prefix_has_one_owner_and_both_complete_paths():
    np.testing.assert_array_equal(validate_motion(*metadata()),[-1,0,0,1])


@pytest.mark.parametrize('fault',('parent-cycle','invalid-joint','dof-span','overlapping-joints',
    'wrong-body','partial-path','repeated-leaf','missing-leaf','layer-dependency','duplicate-layer'))
def test_invalid_branch_or_joint_ownership_rejected(fault):
    a=metadata()
    if fault=='parent-cycle':a[0][1]=1
    elif fault=='invalid-joint':a[5][0]=4
    elif fault=='dof-span':a[6][1]=2
    elif fault=='overlapping-joints':a[2][3]=1
    elif fault=='wrong-body':a[7][1]=3
    elif fault=='partial-path':a[8][0]=2
    elif fault=='repeated-leaf':a[8][-1]=2
    elif fault=='missing-leaf':a[8]=a[8][:2];a[9]=a[9][:2]
    elif fault=='layer-dependency':a[10][1][0]=2
    else:a[10][2]=np.array([2,2,3],np.int32)
    with pytest.raises(ValueError,match='Motion'):
        validate_motion(*a)


def model_data():
    from types import SimpleNamespace as NS
    from mujoco_warp._src import types
    m=NS(nv=5,nbody=4,nactuator=1,ntendon=0,ten_J_rownnz='nnz',ten_J_rowadr='adr',ten_J_colind='col',
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0))
    d=NS(nworld=2,qvel='qvel',moment_rownnz='moment-nnz',moment_rowadr='moment-adr',moment_colind='moment-col',
        actuator_moment='moment',actuator_velocity='act-vel',ten_J='ten-J',ten_velocity='ten-vel',qfrc_bias='bias')
    return m,d


@pytest.mark.parametrize('com,bias',(('upstream','upstream'),('cuda','upstream'),('upstream','cuda'),('cuda','cuda')))
def test_independent_velocity_leaves_keep_one_ordered_caller(monkeypatch,com,bias):
    from types import SimpleNamespace as NS
    from libero.gpu.host.backends.mjwarp import velocity as v
    m,d=model_data();calls=[];topology=NS()
    monkeypatch.setattr(v,'MotionTopology',lambda *a:topology)
    def leaf(name):
        def create(*a):
            assert a[-1] is topology
            return NS(enqueue=lambda:calls.append(name),identity={'leaf':name})
        return create
    monkeypatch.setattr(v,'ComVelocity',leaf('cuda-com'))
    monkeypatch.setattr(v,'Bias',leaf('cuda-bias'))
    monkeypatch.setattr(v.wp,'launch',lambda kernel,**kw:calls.append('act-vel' if kernel is v.forward._actuator_velocity else 'ten-vel'))
    for owner,name,label in ((v.forward,'fwd_velocity','whole-velocity'),(v.smooth,'com_vel','up-com'),
            (v.passive,'passive','passive'),(v.smooth,'rne','up-bias')):
        monkeypatch.setattr(owner,name,lambda *a,label=label:calls.append(label))
    def tendon(model,data,out):
        assert model is m and data is d and out is d.qfrc_bias
        calls.append('tendon-bias')
    monkeypatch.setattr(v.smooth,'tendon_bias',tendon)
    driver=v.Velocity(m,d,None,com_vel=com,bias=bias);driver.enqueue()
    expected=['whole-velocity'] if (com,bias)==('upstream','upstream') else [
        'act-vel','ten-vel','cuda-com' if com=='cuda' else 'up-com','passive',
        'cuda-bias' if bias=='cuda' else 'up-bias','tendon-bias']
    assert calls==expected
    assert driver.identity['com_vel']['selected']==com and driver.identity['bias']['selected']==bias


@pytest.mark.parametrize('fault',('sleep','implicit','empty'))
def test_motion_caller_fallback_loads_no_optional_binary(monkeypatch,fault):
    from mujoco_warp._src import types
    from libero.gpu.host.backends.mjwarp import velocity as v
    m,d=model_data();calls=[]
    if fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    else:m.nv=0
    monkeypatch.setattr(v,'MotionTopology',lambda *a:pytest.fail('optional topology allocation'))
    monkeypatch.setattr(v.forward,'fwd_velocity',lambda *a:calls.append('whole-velocity'))
    driver=v.Velocity(m,d,None,com_vel='cuda',bias='cuda');driver.enqueue()
    assert calls==['whole-velocity'] and all(r['selected']=='upstream' and r['fallback'] for r in driver.identity.values())


def test_default_does_not_prepare_motion_metadata(monkeypatch):
    from libero.gpu.host.backends.mjwarp import velocity as v
    monkeypatch.setattr(v,'MotionTopology',lambda *a:pytest.fail('optional topology allocation'))
    assert v.Velocity(*model_data(),None).topology is None
