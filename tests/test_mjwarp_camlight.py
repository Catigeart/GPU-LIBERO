"""FIXED admission/no-load fallback and independent position-prefix ordering."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import camlight as c,kinematics as k,position as p
from libero.gpu.host.runtime.dynamics.mjwarp_camlight import validate_indices


def model_data():
    m=NS(nv=2,nq=2,nbody=3,njnt=2,nbranch=1,ncam=2,nlight=1,
         opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,run_collision_detection=True))
    for key,width in (('cam_mode',2),('light_mode',1)):setattr(m,key,NS(numpy=lambda width=width:np.zeros(width,np.int32)))
    return m,NS(nworld=2)


@pytest.mark.parametrize('fault',('implicit','sleep','worlds','bodies','empty'))
def test_fallback_before_mode_read_and_optional_load(monkeypatch,fault):
    m,d=model_data();calls=[]
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='worlds':d.nworld=0
    elif fault=='bodies':m.nbody=0
    else:m.ncam=m.nlight=0
    m.cam_mode=m.light_mode=NS(numpy=lambda:pytest.fail('unnecessary mode read'))
    monkeypatch.setattr(c,'CamLightFrames',lambda *a,**kw:pytest.fail('optional load'))
    monkeypatch.setattr(c.smooth,'camlight',lambda *a:calls.append('upstream'))
    driver=c.CamLight(m,d,None,implementation='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.identity['fallback'] and calls==['upstream']


@pytest.mark.parametrize('field',('cam_mode','light_mode'))
@pytest.mark.parametrize('mode',(1,2,3,4))
def test_tracking_modes_visibly_fall_back_before_optional_load(monkeypatch,field,mode):
    m,d=model_data();width=m.ncam if field=='cam_mode' else m.nlight
    setattr(m,field,NS(numpy=lambda:np.full(width,mode,np.int32)))
    monkeypatch.setattr(c,'CamLightFrames',lambda *a,**kw:pytest.fail('optional load'))
    driver=c.CamLight(m,d,None,implementation='cuda')
    assert driver.selected=='upstream' and 'non-FIXED' in driver.identity['fallback']


def test_default_skips_mode_read_and_optional_load(monkeypatch):
    m,d=model_data();m.cam_mode=m.light_mode=NS(numpy=lambda:pytest.fail('mode read'))
    monkeypatch.setattr(c,'CamLightFrames',lambda *a,**kw:pytest.fail('optional load'))
    assert c.CamLight(m,d,None).selected=='upstream'


def test_bad_selector():
    with pytest.raises(ValueError,match='camlight must'):c.CamLight(*model_data(),None,implementation='typo')


def test_admission_owner_and_selector_checked(monkeypatch):
    m,d=model_data();a=c.qualify(m,d,implementation='cuda')
    with pytest.raises(ValueError,match='owner'):c.CamLight(m,NS(nworld=2),None,implementation='cuda',admission=a)
    with pytest.raises(ValueError,match='selector'):c.CamLight(m,d,None,admission=a)


@pytest.mark.parametrize('cam,light',(([],[0,2]),([0,1],[]),([0,2],[1])))
def test_empty_half_and_body_zero_indices(cam,light):validate_indices(3,np.array(cam,int),np.array(light,int))


@pytest.mark.parametrize('cam,light',(([-1],[0]),([3],[0]),([0],[3])))
def test_bad_attachment_indices(cam,light):
    with pytest.raises(ValueError,match='body'):validate_indices(3,np.array(cam),np.array(light))


@pytest.mark.parametrize('kin_cuda',(False,True))
@pytest.mark.parametrize('com_cuda',(False,True))
@pytest.mark.parametrize('cam_cuda',(False,True))
def test_independent_prefix_selection_preserves_original_order(monkeypatch,kin_cuda,com_cuda,cam_cuda):
    m,d=model_data();calls=[]
    monkeypatch.setattr(k,'KinematicFrames',lambda *a,**kw:NS(enqueue=lambda phase:calls.append('kin'+str(phase))))
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:calls.append('whole position'))
    monkeypatch.setattr(k.forward,'fwd_kinematics',lambda *a:calls.append('whole kinematics'))
    for owner,name in ((p.smooth,'kinematics'),(p.smooth,'com_pos'),(p.smooth,'camlight'),(p.smooth,'flex'),(p.smooth,'tendon'),
                       (p.smooth,'crb'),(p.smooth,'tendon_armature'),(p.smooth,'factor_m'),
                       (p.forward.collision_driver,'collision'),(p.forward.constraint,'make_constraint'),(p.smooth,'transmission')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    kin=k.Kinematics(m,d,None,implementation='cuda' if kin_cuda else 'upstream')
    com=NS(selected='cuda' if com_cuda else 'upstream',enqueue=lambda:calls.append('CUDA com'))
    cam=NS(selected='cuda' if cam_cuda else 'upstream',enqueue=lambda:calls.append('CUDA cam'))
    p.Position(m,d,kinematics=kin,com_pos=com,camlight=cam).position()
    if not any((kin_cuda,com_cuda,cam_cuda)):assert calls==['whole position'];return
    expected=['kin'+str(i) for i in range(5)] if kin_cuda else ['kinematics']
    expected+=['CUDA com' if com_cuda else 'com_pos','CUDA cam' if cam_cuda else 'camlight','flex','tendon',
               'crb','tendon_armature','factor_m','collision','make_constraint','transmission']
    assert calls==expected


def test_camera_only_without_kinematics_driver(monkeypatch):
    m,d=model_data();calls=[]
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:pytest.fail('whole caller'))
    for owner,name in ((p.smooth,'kinematics'),(p.smooth,'com_pos'),(p.smooth,'flex'),(p.smooth,'tendon'),(p.smooth,'crb'),
                       (p.smooth,'tendon_armature'),(p.smooth,'factor_m'),(p.forward.collision_driver,'collision'),
                       (p.forward.constraint,'make_constraint'),(p.smooth,'transmission')):
        monkeypatch.setattr(owner,name,lambda *a,name=name:calls.append(name))
    monkeypatch.setattr(p.smooth,'camlight',lambda *a:pytest.fail('duplicate upstream'))
    p.Position(m,d,camlight=NS(selected='cuda',enqueue=lambda:calls.append('CUDA cam'))).position()
    assert calls==['kinematics','com_pos','CUDA cam','flex','tendon','crb','tendon_armature','factor_m','collision','make_constraint','transmission']
