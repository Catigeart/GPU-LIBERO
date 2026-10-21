"""Full-path endpoint screen includes roots and exact public warmstart copies."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
pytest.importorskip('mujoco_warp')
mujoco=pytest.importorskip('mujoco')
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from validation.mjwarp_complete import root_state_checks,ROOTS
from validation.mjwarp_public_forward import upstream_numerical_kernel


def data():
    fields={name:np.zeros((2,1),np.float32) for name in ROOTS};fields['M']=np.ones((2,1),np.float32)
    return NS(**{name:NS(numpy=lambda value=value:value) for name,value in fields.items()})


def model():return mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')


def test_unique_warp_module_does_not_hide_original_numerical_kernel():
    row=dict(module='_next_time_builder__locals___next_time_6a85bd47',python_module='mujoco_warp._src.forward')
    assert upstream_numerical_kernel(row)
    assert upstream_numerical_kernel(dict(module='unique',source='/env/site-packages/mujoco_warp/_src/smooth.py'))
    assert not upstream_numerical_kernel(dict(module='libero.gpu.host.backends.mjwarp.kernels',python_module='libero.gpu.host.backends.mjwarp.kernels'))
    assert not upstream_numerical_kernel(dict(module='warp._src.utils',python_module='warp._src.utils'))


def test_root_velocity_error_cannot_hide_behind_derived_pose():
    a,b=data(),data();a.qvel.numpy()[0]=1
    checks=root_state_checks(model(),a,b)
    assert any(v['field']=='root qvel' and not v['passed'] for v in checks)


def test_warmstart_copy_is_exact_even_with_sensitive_solve():
    a,b=data(),data();a.qacc_warmstart.numpy()[0]=1e-7
    checks=root_state_checks(model(),a,b)
    assert any(v['field']=='cuda public warmstart copy' and not v['passed'] for v in checks)


def test_well_conditioned_acceleration_is_not_given_a_large_envelope():
    a,b=data(),data();a.qacc.numpy()[:]=.005;a.qacc_warmstart.numpy()[:]=.005
    checks=root_state_checks(model(),a,b)
    assert any(v['field']=='qacc frozen sensitivity' and not v['passed'] for v in checks)


@pytest.mark.parametrize('initial_ok',(True,False))
def test_measurement_report_persists_both_modes_without_device_execution(tmp_path,monkeypatch,initial_ok):
    from validation import mjwarp_complete as c
    arrays={name:NS(numpy=lambda:np.zeros((2,1),np.float32)) for name in c.ROOTS+c.OUTPUTS}
    arrays['overflow']=NS(numpy=lambda:np.zeros(2,np.int32));arrays['nacon']=NS(numpy=lambda:np.zeros(1,np.int32));arrays['nefc']=NS(numpy=lambda:np.zeros(2,np.int32));d=NS(naconmax=32,njmax=32,**arrays)
    runtime=NS(data=d,substeps=25,integration_count=NS(numpy=lambda:np.full(2,625,np.int64)))
    env=NS(identity={'stage':{'selected':'cuda'}},built=NS(native_model=object()),data=d,runtime=runtime,stream=None)
    reference=NS(data=d,runtime=runtime)
    comparisons=iter((initial_ok,True))
    monkeypatch.setattr(c,'compare',lambda *a:[dict(field='test',passed=next(comparisons))])
    monkeypatch.setattr(c,'root_state_checks',lambda *a:[])
    monkeypatch.setattr(c,'endpoint_leaf_checks',lambda *a:dict(passed=True,checks=[]))
    monkeypatch.setattr(c,'initial_runtime_contracts',lambda *a:[dict(field='structural',passed=True)])
    monkeypatch.setattr(c,'gpu_state',lambda:{})
    monkeypatch.setattr(c,'measure',lambda *a:dict(total_gpu_seconds=1.,mean_control_ms=50.))
    assert c.run_prepared(env,reference,{}, {},{},dict(events=[]),{},dict(suite='suite',task_id=0),tmp_path)==(0 if initial_ok else 2)
    import json
    row=json.loads((tmp_path/'summary.json').read_text());partial=json.loads((tmp_path/'measurement.json').read_text())
    assert row['passed']==initial_ok and row['initial_numerical_passed']==initial_ok
    assert row['selected']==[dict(path='.stage',selected='cuda')]
    assert partial['completed_modes']==['upstream','cuda'] and (tmp_path/'endpoints.npz').is_file()
    assert (tmp_path/'initial-failure.npz').exists()!=initial_ok


def test_initial_runtime_contracts_reject_nonfinite_and_changed_inputs():
    from validation import mjwarp_complete as c
    def full_data():
        arrays={name:NS(numpy=lambda:np.zeros((2,1),np.float32)) for name in c.ROOTS+c.OUTPUTS}
        arrays['nacon']=NS(numpy=lambda:np.zeros(1,np.int32));arrays['nefc']=NS(numpy=lambda:np.zeros(2,np.int32))
        return NS(naconmax=4,njmax=8,**arrays)
    a,b=full_data(),full_data()
    assert all(v['passed'] for v in c.initial_runtime_contracts(a,b))
    a.qacc=NS(numpy=lambda:np.full((2,1),np.nan,np.float32))
    assert any(v['field']=='cuda finite qacc' and not v['passed'] for v in c.initial_runtime_contracts(a,b))
    a,b=full_data(),full_data();a.qvel=NS(numpy=lambda:np.ones((2,1),np.float32))
    assert any(v['field']=='common initial input qvel' and not v['passed'] for v in c.initial_runtime_contracts(a,b))


def test_saved_initial_failure_cannot_be_readmitted_by_endpoint_reassessment():
    from validation.mjwarp_complete import initial_numerical_agreement
    assert initial_numerical_agreement(dict(initial_checks=[dict(passed=True)]))
    assert not initial_numerical_agreement(dict(initial_checks=[dict(passed=False)]))
    assert not initial_numerical_agreement(dict(initial_numerical_passed=False))


def test_correctness_only_never_enters_measurement(tmp_path,monkeypatch):
    from validation import mjwarp_complete as c
    arrays={name:NS(numpy=lambda:np.zeros((2,1),np.float32)) for name in c.ROOTS+c.OUTPUTS}
    arrays['overflow']=NS(numpy=lambda:np.zeros(2,np.int32))
    arrays['nacon']=NS(numpy=lambda:np.zeros(1,np.int32))
    arrays['nefc']=NS(numpy=lambda:np.zeros(2,np.int32))
    data=NS(naconmax=32,njmax=32,**arrays)
    runtime=NS(data=data,substeps=25,integration_count=NS(numpy=lambda:np.full(2,625,np.int64)))
    env=NS(identity={},built=NS(native_model=object()),data=data,runtime=runtime,stream=None)
    reference=NS(data=data,runtime=runtime)
    monkeypatch.setattr(c,'compare',lambda *a:[])
    monkeypatch.setattr(c,'root_state_checks',lambda *a:[])
    monkeypatch.setattr(c,'initial_runtime_contracts',lambda *a:[])
    monkeypatch.setattr(c,'endpoint_leaf_checks',lambda *a:dict(passed=True,checks=[]))
    monkeypatch.setattr(c,'gpu_state',lambda:{})
    def forbidden(*args):raise AssertionError('Correctness requested but timing was attempted')
    monkeypatch.setattr(c,'measure',forbidden)
    calls=[]
    monkeypatch.setattr(c,'run_controls',lambda *a:calls.append(a))
    assert c.run_prepared(env,reference,{}, {},{},dict(events=[]),{},dict(suite='suite',task_id=0),tmp_path,correctness_only=True)==0
    import json
    row=json.loads((tmp_path/'summary.json').read_text())
    assert len(calls)==2 and row['timings']=={} and row['speedup'] is None
    assert row['execution']['completed_modes']==['upstream','cuda']
    assert (tmp_path/'execution.json').is_file() and not (tmp_path/'measurement.json').exists()
