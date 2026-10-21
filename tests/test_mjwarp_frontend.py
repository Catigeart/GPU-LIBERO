"""CPU-only routing and independent packed-mass validation contracts."""
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np

from libero.gpu.host.adapters.environment.batch_env import GpuBatchEnv


def test_mjwarp_routes_before_native_construction(monkeypatch):
    import libero.gpu.precision as scalar
    class Alternate:
        pass
    selected = []
    monkeypatch.setattr(scalar, 'select', lambda value: selected.append(value))
    monkeypatch.setitem(sys.modules, 'libero.gpu.host.backends.mjwarp.environment',
                        SimpleNamespace(MjWarpBatchEnv=Alternate))
    result = GpuBatchEnv.__new__(GpuBatchEnv, 'unused.bddl', physics_engine='mjwarp')
    assert isinstance(result, Alternate)
    assert selected == ['fp32']
    assert type(GpuBatchEnv.__new__(GpuBatchEnv, 'unused.bddl', precision='fp64')) is GpuBatchEnv


def test_precision_defaults_and_explicit_native_override(monkeypatch):
    from libero.gpu.host.backends.selection import resolve_engine
    import libero.gpu.precision as scalar
    import pytest
    monkeypatch.setattr(scalar, 'bits', 64)
    assert resolve_engine() == 'native'
    assert resolve_engine('auto', 'fp32') == 'cuda_fp32'
    assert resolve_engine('auto', 'float64') == 'native'
    assert resolve_engine('native', 'fp32') == 'native'
    monkeypatch.setattr(scalar, 'bits', 32)
    assert resolve_engine() == 'cuda_fp32'
    with pytest.raises(ValueError, match='requires FP32'):
        resolve_engine('mjwarp', 'fp64')
    with pytest.raises(ValueError):
        resolve_engine('unavailable', 'fp32')
    with pytest.raises(ValueError):
        resolve_engine('auto', 'fp16')


def test_auto_fp32_uses_independent_cuda_and_explicit_native_stays_native(monkeypatch):
    import libero.gpu.precision as scalar
    class Alternate:
        pass
    monkeypatch.setattr(scalar, 'select', lambda value: value)
    monkeypatch.setattr(scalar, 'bits', 32)
    monkeypatch.setitem(sys.modules, 'libero.gpu.host.backends.cuda_fp32.environment',
                        SimpleNamespace(CudaFP32BatchEnv=Alternate))
    assert isinstance(GpuBatchEnv.__new__(GpuBatchEnv, 'unused'), Alternate)
    assert isinstance(GpuBatchEnv.__new__(GpuBatchEnv, 'unused', precision='fp32'), Alternate)
    assert type(GpuBatchEnv.__new__(GpuBatchEnv, 'unused', precision='fp32',
                                  physics_engine='native')) is GpuBatchEnv


def test_portable_mass_expansion_matches_cpu_mujoco():
    if mujoco.__version__ != '2.3.7':
        import pytest
        pytest.skip('independent layout reference is MuJoCo 2.3.7')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from validation.precision import dense_mass
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint axis="0 0 1"/><geom type="sphere" size=".1"/>
      <body pos=".2 0 0"><joint axis="0 1 0"/><geom type="sphere" size=".1"/>
      <body pos=".2 0 0"><joint axis="1 0 0"/><geom type="sphere" size=".1"/>
      </body></body></body></worldbody></mujoco>''')
    indices = np.full((model.nv, model.nv), -1, np.int32)
    for k in range(model.nM):
        basis = np.zeros(model.nM)
        basis[k] = 1
        matrix = dense_mass(model, basis)
        indices[matrix == 1] = k
    portable = dict(indices=indices, nv=model.nv, nM=model.nM)
    for seed in range(3):
        packed = np.random.default_rng(seed).normal(size=model.nM)
        np.testing.assert_array_equal(dense_mass(portable, packed), dense_mass(model, packed))


def test_optional_engine_has_separate_source_admission():
    from libero.gpu.host.build.aot import _source_identity_files
    native = _source_identity_files()
    guarded = _source_identity_files(include_excluded=True)
    optional = {name for name in guarded if '/host/backends/mjwarp/' in name or
                '/host/backends/cuda_fp32/' in name or '/cuda/runtime/' in name or
                name.endswith('/cpu_scene_input_source_pin.json') or
                name.endswith('/mjwarp_source_pin.json') or name.endswith('/mjwarp_controller.cu')}
    assert optional
    assert optional.isdisjoint(native)


def test_profile_path_binds_sources_and_controller_binary(tmp_path, monkeypatch):
    from libero.gpu.host.backends.mjwarp import build
    library = tmp_path / 'controller.so'
    library.write_bytes(b'controller one')
    env = SimpleNamespace(identity=dict(worlds=2, precision='fp32'),
                          build_root=tmp_path, library_path=library)
    monkeypatch.setattr(build, 'source_files', lambda: {'runtime.py': 'version one'})
    first = build.profile_path(env)
    assert first == build.profile_path(env)
    monkeypatch.setattr(build, 'source_files', lambda: {'runtime.py': 'version two'})
    second = build.profile_path(env)
    assert second != first
    library.write_bytes(b'controller two')
    assert build.profile_path(env) not in (first, second)


def test_backend_joint_addresses_use_compiled_spans():
    from libero.gpu.host.backends.mjwarp.facade import bind_model_compatibility
    model = SimpleNamespace(joint_name2id=lambda name: {'free': 0, 'hinge': 1}[name],
        jnt_qposadr=np.array([0, 7], np.int32), jnt_dofadr=np.array([0, 6], np.int32),
        _metadata=dict(joint_qpos_slices={'free': (0, 7), 'hinge': (7, 8)},
                       joint_qvel_slices={'free': (0, 6), 'hinge': (6, 7)}))
    bind_model_compatibility(model)
    assert model.get_joint_qpos_addr('free') == (0, 7)
    assert model.get_joint_qvel_addr('free') == (0, 6)
    assert model.get_joint_qpos_addr('hinge') == np.int32(7)
    assert model.get_joint_qvel_addr('hinge') == np.int32(6)
