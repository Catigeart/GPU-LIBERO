"""Empty-state admission and complete fallback for the integration suffix."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import publication as p


def model():return NS(na=0,nhistory=0,opt=NS(enableflags=0,disableflags=0))


@pytest.mark.parametrize('fault',['activation','history','sleep'])
def test_fallback_before_load(monkeypatch,fault):
    m=model();calls=[]
    if fault=='activation':m.na=1
    if fault=='history':m.nhistory=1
    if fault=='sleep':m.opt.enableflags=types.EnableBit.SLEEP
    monkeypatch.setattr(p,'PublicationState',lambda *a:pytest.fail('optional load'))
    monkeypatch.setattr(p,'advance_activation',lambda *a:calls.append('activation'))
    monkeypatch.setattr(p,'publish_advance',lambda *a:calls.append('publish'))
    d=p.Publication(m,NS(),None,implementation='cuda');d.activation();d.publish()
    assert d.selected=='upstream' and d.fallback and calls==['activation','publish']


def test_stateless_suffix_skips_noops_and_copies_public_acceleration(monkeypatch):
    calls=[];m=model();d=NS(qacc=object(),qacc_warmstart=object())
    monkeypatch.setattr(p,'PublicationState',lambda *a:NS(enqueue=lambda:calls.append('time'),identity={}))
    monkeypatch.setattr(p,'advance_activation',lambda *a:pytest.fail('empty activation'))
    monkeypatch.setattr(p,'publish_advance',lambda *a:pytest.fail('duplicate original suffix'))
    monkeypatch.setattr(p.wp,'copy',lambda dst,src:calls.append((dst,src)))
    driver=p.Publication(m,d,None,implementation='cuda');driver.activation();driver.publish()
    assert calls==['time',(d.qacc_warmstart,d.qacc)]


def test_sleep_flag_disabled_by_island():
    m=model();m.opt.enableflags=types.EnableBit.SLEEP;m.opt.disableflags=types.DisableBit.ISLAND
    assert p.unsupported_reason(m) is None
