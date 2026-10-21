"""Public refresh factorization order and independent Newton ownership."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import stages,acceleration,mass_factor


def test_public_schedule_preserves_later_factorization_and_no_integration(monkeypatch):
    calls=[];m=NS(opt=NS(disableflags=0,run_rne_postconstraint=True),callback=NS(control=lambda *a:calls.append('control')));d=NS(sensordata=NS(zero_=lambda:calls.append('clear-sensor')))
    p=stages.PhysicsStages.__new__(stages.PhysicsStages);p.model,p.data=m,d;p.queries=NS(**{name:(lambda name=name:calls.append(name)) for name in ('sensor_pos','energy_pos','sensor_vel','energy_vel','sensor_acc')});p.postconstraint_driver=NS(enqueue=lambda:calls.append('post'))
    p.position=lambda **kw:calls.append(('position',kw));p.velocity=lambda:calls.append('velocity');p.actuation=lambda:calls.append('actuation');p.acceleration=lambda **kw:calls.append(('acceleration',kw));p.newton=lambda:calls.append('newton');p.integrate=lambda:pytest.fail('public forward integrated')
    monkeypatch.setattr(stages.forward,'check_discrete',lambda *a:None)
    p.forward()
    assert calls==[('position',dict(factorize=False)),'clear-sensor','sensor_pos','energy_pos','velocity','sensor_vel','energy_vel','control','actuation',('acceleration',dict(factorize=True)),'newton','post','sensor_acc']


def test_selected_public_factor_solve_replaces_plain_solve(monkeypatch):
    calls=[];p=acceleration.Acceleration.__new__(acceleration.Acceleration);p.model=NS();p.data=NS();p.binding=NS(enqueue=lambda:calls.append('plain'))
    f=NS(public_binding=NS(enqueue=lambda:calls.append('factor-solve')))
    p._solve(True,f);p._solve(False,f)
    assert calls==['factor-solve','plain']


def test_public_mass_binding_is_prepared_once_without_affecting_euler():
    calls=[];p=mass_factor.MassFactors.__new__(mass_factor.MassFactors);p.data=NS(M='M',qLD='L',qLDiagInv='D',qacc_smooth='a',qfrc_smooth='f');p.public_binding=None;p.plan=NS(bind=lambda *a:calls.append(a) or object())
    p.prepare_public();owner=p.public_binding;p.prepare_public()
    assert p.public_binding is owner and calls==[('M','L','D','a','f')]
