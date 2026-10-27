"""Process precision, actual ABI layouts and typed reset defaults; no GPU work."""
import os
import hashlib
import json
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PrecisionPolicyTests(unittest.TestCase):
    def test_default_and_explicit_backend_selection(self):
        source = '''
import sys
from libero.gpu import precision as scalar
from libero.gpu.host.backends.selection import resolve_engine
assert scalar.bits == int(sys.argv[1])
assert resolve_engine() == sys.argv[2]
scalar.select(sys.argv[3])
assert scalar.bits == int(sys.argv[3][2:])
assert resolve_engine() == sys.argv[4]
assert 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
'''
        for configured, bits, engine, explicit, selected in (
                (None, 32, 'cuda_fp32', 'fp32', 'cuda_fp32'),
                (None, 32, 'cuda_fp32', 'fp64', 'native'),
                ('fp64', 64, 'native', 'fp64', 'native'),
                ('float64', 64, 'native', 'fp32', 'cuda_fp32')):
            with self.subTest(configured=configured, explicit=explicit):
                environment = dict(os.environ)
                environment.pop('LIBERO_GPU_PRECISION', None)
                if configured is not None:
                    environment['LIBERO_GPU_PRECISION'] = configured
                result = subprocess.run([sys.executable, '-c', source, str(bits), engine,
                                         explicit, selected], capture_output=True, text=True,
                                        env=environment, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_specialization_index_excludes_obsolete_templates(self):
        source = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/runtime/core/specialization_cache.py'
        index = runpy.run_path(str(source))['index_packages']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / 'template.py'
            template.write_text('current')
            for name, digest in (('shape_s_aaa', hashlib.sha256(b'current').hexdigest()),
                                 ('shape_s_zzz', hashlib.sha256(b'obsolete').hexdigest())):
                package = root / name
                package.mkdir()
                (package / 'MODEL_MANIFEST.json').write_text(json.dumps({'model': 'same'}))
                (package / 'SOURCE_GENERATION.json').write_text(json.dumps(dict(
                    template_pins=dict(kernel=dict(path=str(template), sha256=digest)))))
            entries = index(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(next(iter(entries.values()))['package'], 'shape_s_aaa')
            with self.assertRaisesRegex(ValueError, 'stale source pins'):
                index(root, [root / 'shape_s_zzz'])

    def check_policy(self, bits):
        source = r'''
import ctypes as ct
import numpy as np
import libero.gpu.precision as scalar
scalar.select('fpBITS')
from libero.gpu.host.runtime.step.substep_types import SubstepArguments
from libero.gpu.host.runtime.step.fused_substep import FusedSubstep
from libero.gpu.host.runtime.step.substep_binding import _validate_smooth_reuse
from types import SimpleNamespace
from libero.gpu.host.runtime.dynamics.recovery import model_root_defaults
assert scalar.itemsize == BITS // 8
assert ct.sizeof(scalar.ct_real) == scalar.itemsize
assert scalar.wp_real.__name__ == 'floatBITS'
assert scalar.wp_bits.__name__ == 'uintBITS'
assert ct.sizeof(SubstepArguments) == (6288 if BITS == 32 else 6296)
dimensions = dict(nq=1, nv=1, nu=0, na=0, nbody=1, nmocap=0,
                  nuserdata=0, nsensordata=0, nplugin=0, npluginstate=0)
q0=np.array([0.1],dtype=np.float64)
defaults=model_root_defaults(dimensions,qpos0=q0,body_mocapid=np.array([-1],np.int32),
                            body_pos=np.zeros((1,3)),body_quat=np.array([[1.,0.,0.,0.]]))
assert all(a.dtype == np.dtype('floatBITS') for a in defaults.values())
np.testing.assert_array_equal(defaults['qpos'], q0.astype('floatBITS'))
assert defaults['qpos'].nbytes == scalar.itemsize
assert scalar.solver_tolerance(1e-8) == (1e-6 if BITS == 32 else 1e-8)
# Path resolution is read-only and independent of a user's directory spelling.
import tempfile
from pathlib import Path
from libero.gpu.host.build.project_paths import project_paths, prepare_precision_project, cmake_root
with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary) / 'fp32'
    paths = project_paths(base)
    assert not base.exists()
    assert paths.base == base
    assert paths.precision == (base / 'fp32' if BITS == 32 else base)
    prepared = prepare_precision_project(base)
    assert project_paths(prepared.precision) == prepared
    assert cmake_root(prepared.precision / 'cmake-control') == base / 'cmake-control'
    if BITS == 32:
        import json
        from libero.gpu.host.build.task_catalog import load_catalog
        source = base / 'TASK_INSTANCES.json'
        scene = base / 'scenes/task'
        scene.mkdir(parents=True)
        (scene / 'model.mjb').write_bytes(b'fixture')
        record = dict(tasks=[dict(key='task', seed=0, directory='scenes/task',
                                  constraint_package='constraints/task')])
        source.write_text(json.dumps(record))
        prepare_precision_project(base)
        assert load_catalog(prepared.precision)[0]['directory'] == str(prepared.precision / '../scenes/task')
        record['tasks'][0]['seed'] = 1
        source.write_text(json.dumps(record))
        prepare_precision_project(base)
        moved = base.with_name('renamed')
        base.rename(moved)
        row = load_catalog(moved / 'fp32')[0]
        assert row['seed'] == 1
        assert (Path(row['directory']) / 'model.mjb').read_bytes() == b'fixture'
# Adjacent real allocations must remain disjoint in the byte-span proof.
arena=(scalar.ct_real*4)();applied=(scalar.ct_real*2)();external=(scalar.ct_real*6)()
arguments=SubstepArguments()
arguments.state.nv=arguments.smooth_acceleration.nv=arguments.smooth_commit.width=2
arguments.fluid_passive.nbody=1
arguments.smooth_commit.pf=ct.addressof(arena)
arguments.smooth_commit.pa=ct.addressof(arena)+2*scalar.itemsize
arguments.smooth_acceleration.applied=ct.addressof(applied)
arguments.smooth_acceleration.xfrc=ct.addressof(external)
class EmptyScratch:
    def __getattr__(self, name): return 0
core=SimpleNamespace(params=EmptyScratch(),owners=[])
assert _validate_smooth_reuse(arguments,core,1)['passed']
arguments.smooth_commit.pa=ct.addressof(arena)+scalar.itemsize
try:
    _validate_smooth_reuse(arguments,core,1)
except ValueError:
    pass
else:
    raise AssertionError('overlapping smooth publications were accepted')
assert scalar.select('fpBITS') == 'fpBITS'
try:
    scalar.select('fpOTHER')
except RuntimeError:
    pass
else:
    raise AssertionError('mixed process precision was accepted')
'''.replace('BITS', str(bits)).replace('OTHER', str(96 - bits))
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True,
                                env=dict(os.environ, LIBERO_GPU_PRECISION='fp64'), timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fp32(self):
        self.check_policy(32)

    def test_fp64(self):
        self.check_policy(64)


if __name__ == '__main__':
    unittest.main()
