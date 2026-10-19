"""Geometric regressions against the maintained CUDA BoxBox body on the CPU.

Only the device entry is adapted to a small host ABI. MuJoCo supplies the
double-precision vector helpers. GPU subgroup execution is checked separately.
"""
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'libero/libero/gpu/cuda/collision/native/native_source.cuh'
ORIGINAL = ROOT / 'libero/libero/gpu/host/runtime/collision/native_originals/engine_collision_box.c.txt'
FIXTURE = Path(__file__).parent / 'fixtures/boxbox_task5.json'


def function_body(source, name):
    start = source.index('int ' + name + '(')
    brace = source.index('{', start)
    level, end = 1, brace + 1
    while level:
        level += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def host_probe_source():
    header = HEADER.read_text()
    prefix = header[header.index('struct NativeBoxAxis'):header.index('__device__ int mjc_BoxBox(')]
    body = function_body(header, 'mjc_BoxBox')
    reference = function_body(ORIGINAL.read_text(), 'mjc_BoxBox')
    reference = reference.replace('int mjc_BoxBox(', 'int reference_BoxBox(', 1)
    reference = reference.replace('if (code == -1)', 'M->adapter_diag[6] = code;\n  if (code == -1)', 1)
    code = prefix + body + '\n' + reference
    for old, new in (('mjModel', 'ProbeModel'), ('mjData', 'ProbeData'), ('mjContact', 'ProbeContact')):
        code = re.sub(r'\b' + old + r'\b', new, code)
    return r'''
#include <mujoco/mujoco.h>
#include <cmath>
#define __device__
#define __forceinline__ inline
#define mju_abs fabs
struct { int x; } threadIdx = {0};
static unsigned native_query_mask(int) { return ~0u; }
template<class T> T __shfl_sync(unsigned, T x, int, int) { return x; }
struct ProbeModel {
  const double* geom_size; int* adapter_diag; int* adapter_frame0;
  int adapter_subgroup_width;
};
struct ProbeData { const double* geom_xpos; const double* geom_xmat; };
struct ProbeContact { double dist, pos[3], frame[9]; };
''' + code + r'''
extern "C" int run(const double* size, const double* pos, const double* mat,
                   double margin, double* raw, int* sat, int reference) {
  int diag[16]{}, frame = 0;
  ProbeModel m{size, diag, &frame, 0}; ProbeData d{pos, mat};
  ProbeContact con[50]{};
  int n = reference ? reference_BoxBox(&m, &d, con, 0, 1, margin)
                    : mjc_BoxBox(&m, &d, con, 0, 1, margin);
  *sat = diag[6];
  for (int j = 0; j < n && j < 50; ++j) {
    raw[10*j] = con[j].dist;
    for (int k = 0; k < 3; ++k) raw[10*j+1+k] = con[j].pos[k];
    for (int k = 0; k < 6; ++k) raw[10*j+4+k] = con[j].frame[k];
  }
  return n;
}
'''


class BoxContactGenerationTests(unittest.TestCase):
    def test_derivation_retains_the_maintained_contact_correction(self):
        path = ROOT / 'libero/libero/gpu/host/build/box_contact_source.py'
        spec = importlib.util.spec_from_file_location('derive_box_contacts', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generated = module.correct_boxbox_edge_contacts(
            function_body(ORIGINAL.read_text(), 'mjc_BoxBox')).split('edgeedge:', 1)[1]
        maintained = function_body(HEADER.read_text(), 'mjc_BoxBox').split('edgeedge:', 1)[1]
        maintained = maintained.replace('  *M->adapter_frame0=1;\n', '')
        self.assertEqual(generated, maintained)


class NativeBoxContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('g++'):
            raise unittest.SkipTest('host contact regression requires g++')
        try:
            import mujoco
        except ImportError:
            raise unittest.SkipTest('host contact regression requires MuJoCo')
        cls.mujoco = mujoco
        cls.temp = tempfile.TemporaryDirectory(prefix='boxbox_regression_')
        cls.addClassCleanup(cls.temp.cleanup)
        package = Path(mujoco.__file__).parent
        library = next(package.glob('libmujoco.so.*'))
        source = Path(cls.temp.name) / 'probe.cpp'
        source.write_text(host_probe_source())
        target = source.with_suffix('.so')
        subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                        '-I' + str(package / 'include'), str(source), str(library),
                        '-Wl,-rpath,' + str(package), '-o', str(target)],
                       check=True, capture_output=True, text=True)
        cls.library = ct.CDLL(str(target))
        cls.run_probe = cls.library.run
        cls.run_probe.argtypes = [ct.c_void_p] * 3 + [ct.c_double, ct.c_void_p,
                                                   ct.c_void_p, ct.c_int]
        cls.run_probe.restype = ct.c_int

    def probe(self, size, pos, mat, margin=0., reference=False):
        arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in (size, pos, mat)]
        raw = np.zeros((50, 10))
        sat = ct.c_int()
        count = self.run_probe(*(a.ctypes.data for a in arrays), margin, raw.ctypes.data,
                               ct.byref(sat), int(reference))
        self.assertTrue(0 <= count <= 50)
        self.assertTrue(np.isfinite(raw[:count]).all())
        return raw[:count], sat.value

    def test_captured_book_contact_has_no_lateral_negative_depth(self):
        f = json.loads(FIXTURE.read_text())
        old, old_axis = self.probe(f['size'], f['pos'], f['mat'], reference=True)
        new, new_axis = self.probe(f['size'], f['pos'], f['mat'])
        self.assertEqual(old_axis, 18)
        self.assertLess(old[:, 0].min(), -1.4)
        self.assertEqual(new_axis, old_axis)
        self.assertEqual(len(new), 4)
        np.testing.assert_allclose(new, old[1:], atol=1e-15, rtol=0)
        self.assertGreater(new[:, 0].min(), -2e-7)
        # The box-2 witness must be on the actual book, including its thin axis.
        witness = new[:, 1:4] + new[:, :1] * new[:, 4:7] * .5
        local = (witness - f['pos'][1]) @ np.asarray(f['mat'][1])
        self.assertTrue((np.abs(local) <= np.asarray(f['size'][1]) + 1e-12).all())

    def test_failure_pose_rotations_swap_and_scales(self):
        f = json.loads(FIXTURE.read_text())
        rng = np.random.RandomState(13)
        for scale in (1e-3, 1., 1e3):
            for swap in (False, True):
                for rotation in range(6):
                    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
                    matrix = np.empty(9); self.mujoco.mju_quat2Mat(matrix, quat)
                    matrix = matrix.reshape(3, 3)
                    size = np.asarray(f['size']) * scale
                    pos = np.asarray(f['pos']) @ matrix.T * scale
                    mat = matrix @ np.asarray(f['mat'])
                    if swap:
                        size, pos, mat = size[::-1], pos[::-1], mat[::-1]
                    with self.subTest(scale=scale, swap=swap, rotation=rotation):
                        contacts, _ = self.probe(size, pos, mat)
                        self.assertGreater(len(contacts), 0)
                        self.assertGreater(contacts[:, 0].min(), -2e-7 * scale)
                        self.assertLessEqual(contacts[:, 0].max(), 1e-12 * scale)

    def test_face_contacts_and_separation(self):
        size = np.ones((2, 3)); mat = np.tile(np.eye(3), (2, 1, 1))
        for z, margin in ((1.9, 0.), (2.01, .02), (2.1, .02)):
            with self.subTest(z=z, margin=margin):
                args = (size, [[0, 0, 0], [0, 0, z]], mat, margin)
                new, _ = self.probe(*args)
                old, _ = self.probe(*args, reference=True)
                if z == 2.1:
                    self.assertEqual(len(new), 0)
                else:
                    # Raw edge/face candidates contain duplicates; production
                    # exact-position dedup yields four corners for these boxes.
                    self.assertEqual(len(np.unique(new, axis=0)), 4)
                    np.testing.assert_allclose(new[:, 0], .01 if z > 2 else -.05,
                                               atol=1e-15, rtol=0)
                if z < 2:
                    np.testing.assert_array_equal(new, old)

    def test_projection_determinant_is_not_overwritten_by_first_corner(self):
        f = json.loads((FIXTURE.parent / 'boxbox_edge_projection.json').read_text())
        new, axis = self.probe(f['size'], f['pos'], f['mat'])
        old, _ = self.probe(f['size'], f['pos'], f['mat'], reference=True)
        self.assertGreaterEqual(axis, 12)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(new), 4)
        self.assert_box_witnesses(new, f['size'], f['pos'], f['mat'])

    def test_outside_candidate_has_positive_distance_and_fits_margin(self):
        f = json.loads(FIXTURE.read_text())
        for margin, count in ((.01, 4), (2., 5)):
            with self.subTest(margin=margin):
                new, axis = self.probe(f['size'], f['pos'], f['mat'], margin)
                self.assertEqual(axis, 18)
                self.assertEqual(len(new), count)
                self.assertGreater(new[:, 0].min(), -2e-7)
                self.assertLessEqual(new[:, 0].max(), margin)
                if margin == 2.:
                    self.assertAlmostEqual(new[0, 0], 1.4826042589494803)

    def assert_box_witnesses(self, contacts, size, pos, mat):
        size, pos, mat = np.asarray(size), np.asarray(pos), np.asarray(mat).reshape(2, 3, 3)
        for box in range(2):
            witness = contacts[:, 1:4] + (1 if box else -1) * contacts[:, :1] * contacts[:, 4:7] * .5
            local = (witness - pos[box]) @ mat[box]
            self.assertTrue((np.abs(local) <= size[box] + 1e-10).all())
            self.assertTrue((np.min(np.abs(np.abs(local) - size[box]), axis=1) < 1e-10).all())

    def test_random_nonparallel_contacts_have_valid_box_witnesses(self):
        rng = np.random.RandomState(41)
        colliding = edges = 0
        for i in range(1000):
            size = 10 ** rng.uniform(-2, 0, (2, 3))
            mat = np.empty((2, 9))
            for k in range(2):
                q = rng.normal(size=4); q /= np.linalg.norm(q)
                self.mujoco.mju_quat2Mat(mat[k], q)
            pos = np.zeros((2, 3))
            pos[1] = rng.uniform(-1, 1, 3) * np.sum(size, axis=0)
            new, axis = self.probe(size, pos, mat)
            old, _ = self.probe(size, pos, mat, reference=True)
            if axis < 12:
                np.testing.assert_array_equal(new, old, err_msg='random face case %d' % i)
            elif len(new):
                self.assert_box_witnesses(new, size, pos, mat)
                np.testing.assert_allclose(np.linalg.norm(new[:, 4:7], axis=1), 1., atol=1e-14)
            self.assertEqual(bool(len(new)), bool(len(old)), 'collision existence case %d' % i)
            colliding += len(new) > 0
            edges += len(new) > 0 and axis >= 12
        self.assertGreater(colliding, 300)
        self.assertGreater(edges, 100)


if __name__ == '__main__':
    unittest.main()
