"""Broadphase domain, pair ownership and explicit rigid collision scheduling."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import collision as c,position as p
from libero.gpu.host.runtime.collision.mjwarp_broadphase import validate_pairs


def model_data():
    m=NS(ngeom=4,nflex=0,opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,broadphase=types.BroadphaseType.NXN),callback=NS(contactfilter=None))
    return m,NS(nworld=2,naconmax=8)


@pytest.mark.parametrize('fault',('implicit','sleep','sap','flex','empty','worlds'))
def test_fallback_before_context_and_binary_load(monkeypatch,fault):
    m,d=model_data()
    if fault=='implicit':m.opt.integrator=types.IntegratorType.IMPLICIT
    elif fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    elif fault=='sap':m.opt.broadphase=types.BroadphaseType.SAP_TILE
    elif fault=='flex':m.nflex=1
    elif fault=='empty':m.ngeom=0
    else:d.nworld=0
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:pytest.fail('optional context allocation'))
    monkeypatch.setattr(c,'BroadphaseNumerics',lambda *a,**kw:pytest.fail('optional CUDA loading'))
    calls=[];monkeypatch.setattr(c.collision_driver,'collision',lambda *a:calls.append('whole'))
    driver=c.Collision(m,d,None,broadphase='cuda');driver.enqueue()
    assert driver.selected=='upstream' and driver.identity['fallback'] and calls==['whole']


def test_default_no_optional_context_and_bad_selector(monkeypatch):
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:pytest.fail('optional context'))
    assert c.Collision(*model_data(),None).selected=='upstream'
    with pytest.raises(ValueError,match='broadphase must'):c.Collision(*model_data(),None,broadphase='invalid')


def pair_metadata():return np.array([0,6,7]),np.array([[0,1],[0,2],[1,2]]),np.array([[-1,-1],[0,-1],[-2,0]])


def test_valid_filtered_pairs():validate_pairs(*pair_metadata(),1,1)


@pytest.mark.parametrize('fault',('type','range','order','duplicate','pair','sensor','excluded'))
def test_bad_pair_metadata_rejected(fault):
    g,pairs,ids=pair_metadata()
    if fault=='type':g[2]=9
    elif fault=='range':pairs[0,1]=3
    elif fault=='order':pairs[0]=[1,0]
    elif fault=='duplicate':pairs[1]=pairs[0]
    elif fault=='pair':ids[0,0]=1
    elif fault=='sensor':ids[0,1]=1
    else:ids[0]=[-2,-1]
    with pytest.raises(ValueError,match='Broadphase'):validate_pairs(g,pairs,ids,1,1)


@pytest.mark.parametrize('disabled',('no','constraint','contact','capacity'))
def test_cuda_collision_resets_and_callback_order(monkeypatch,disabled):
    m,d=model_data();calls=[]
    if disabled=='constraint':m.opt.disableflags=types.DisableBit.CONSTRAINT
    elif disabled=='contact':m.opt.disableflags=types.DisableBit.CONTACT
    elif disabled=='capacity':d.naconmax=0
    d.nacon=NS(zero_=lambda:calls.append('clear-contacts'));d.ncollision=NS(zero_=lambda:calls.append('clear-pairs'))
    m.callback.contactfilter=lambda *a:calls.append('callback')
    ctx=NS(collision_pair=NS(capacity=64),collision_pairid=NS(capacity=64),collision_worldid=NS(capacity=32))
    monkeypatch.setattr(c.collision_core,'create_collision_context',lambda *a:ctx)
    monkeypatch.setattr(c,'BroadphaseNumerics',lambda *a,**kw:NS(enqueue=lambda:calls.append('broadphase'),identity={}))
    monkeypatch.setattr(c.collision_driver,'_narrowphase',lambda a,b,context:calls.append('narrowphase') if context is ctx else pytest.fail('owner'))
    monkeypatch.setattr(c.collision_driver,'collision',lambda *a:pytest.fail('whole upstream duplicates prefix'))
    driver=c.Collision(m,d,None,broadphase='cuda');driver.enqueue()
    assert calls==(['clear-pairs','clear-contacts','broadphase','narrowphase','callback'] if disabled=='no' else ['clear-contacts'])
    assert driver.identity['context_bytes']==160


def test_broadphase_only_selection_prevents_whole_position_bypass(monkeypatch):
    m,d=model_data();m.opt.run_collision_detection=True;calls=[]
    monkeypatch.setattr(p.forward,'fwd_position',lambda *a:pytest.fail('position bypass'))
    monkeypatch.setattr(p.forward,'fwd_kinematics',lambda *a:calls.append('kinematics'))
    for name in ('crb','tendon_armature','factor_m','transmission'):monkeypatch.setattr(p.smooth,name,lambda *a,name=name:calls.append(name))
    monkeypatch.setattr(p.forward.collision_driver,'collision',lambda *a:pytest.fail('upstream collision bypass'))
    monkeypatch.setattr(p.forward.constraint,'make_constraint',lambda *a:calls.append('constraints'))
    p.Position(m,d,collision=NS(selected='cuda',enqueue=lambda:calls.append('collision'))).position()
    assert calls==['kinematics','crb','tendon_armature','factor_m','collision','constraints','transmission']
