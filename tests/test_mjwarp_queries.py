"""Query admission, metadata and sensor callbacks without a GPU."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
from mujoco_warp._src import types
from libero.gpu.host.backends.mjwarp import queries as q
from libero.gpu.host.runtime.dynamics.mjwarp_queries import validate_geometry,validate_force_sensors


def test_geometry_admits_empty_and_rejects_nonbinary():
    validate_geometry(np.array([0]),np.empty((1,0),int),1,0)
    with pytest.raises(ValueError):validate_geometry(np.array([0,1]),np.array([[0],[2]]),2,1)


def sensor_args():return [np.array([0,1]),np.array([1]),np.array([4,5]),np.array([0,0]),np.array([2,5]),np.array([0,1]),np.array([3,3]),np.array([6,6]),np.array([1,0]),2,9]


def test_sensor_permutation_and_gaps():validate_force_sensors(*sensor_args())


@pytest.mark.parametrize('field,value',[(2,[4,6]),(3,[0,1]),(4,[2,4]),(4,[2,8]),(5,[0,4]),(6,[3,4]),(7,[6,1]),(8,[0,0])])
def test_invalid_sensor_metadata(field,value):
    a=sensor_args();a[field]=np.array(value)
    with pytest.raises(ValueError):validate_force_sensors(*a)


def test_unselected_queries_never_load_or_read_metadata(monkeypatch):
    monkeypatch.setattr(q,'ForceSensors',lambda *a:pytest.fail('optional load'))
    driver=q.Queries(NS(),NS(),None)
    assert driver.gravity is None and driver.sensors is None and driver.jacobian(None,None,None,None) is None


@pytest.mark.parametrize('disabled',[False,True])
def test_sensor_callbacks_and_no_duplicate_numerics(monkeypatch,disabled):
    calls=[];m=NS(nhistory=0,sensor_type=NS(numpy=lambda:np.array([4,5])),opt=NS(disableflags=int(types.DisableBit.SENSOR) if disabled else 0),callback=NS(sensor=lambda m,d,s:calls.append(s)))
    monkeypatch.setattr(q,'ForceSensors',lambda *a:NS(enqueue=lambda:calls.append('force')))
    monkeypatch.setattr(q,'GravityProjection',lambda *a:NS(identity={}))
    for key in ('sensor_pos','sensor_vel','sensor_acc'):monkeypatch.setattr(q.sensor,key,lambda *a,**kw:pytest.fail('duplicate raw math'))
    driver=q.Queries(m,NS(),None,implementation='cuda');driver.sensor_pos();driver.sensor_vel();driver.sensor_acc()
    assert calls==(['force'] if disabled else [types.Stage.POS,types.Stage.VEL,'force',types.Stage.ACC])


def test_history_retains_whole_sensor_boundary(monkeypatch):
    m=NS(nhistory=1);monkeypatch.setattr(q,'ForceSensors',lambda *a:pytest.fail('sensor optional load'));monkeypatch.setattr(q,'GravityProjection',lambda *a:NS(identity={}))
    driver=q.Queries(m,NS(),None,implementation='cuda');assert driver.sensors is None and 'history' in driver.identity['sensor_fallback']
