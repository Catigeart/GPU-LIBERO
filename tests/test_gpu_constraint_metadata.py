"""Cold S qualification/descriptor contracts; no Warp import or CUDA work."""
import ast
import copy
from dataclasses import FrozenInstanceError
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
MODULE = 'libero.gpu.host.runtime.newton.head_metadata'
spec = importlib.util.spec_from_file_location(MODULE, GPU/'host/runtime/newton/head_metadata.py')
head_module = importlib.util.module_from_spec(spec)
sys.modules[MODULE] = head_module
spec.loader.exec_module(head_module)
HeadMetadata = head_module.HeadMetadata


def source_functions(relative, names, namespace):
    path = GPU/relative
    nodes = [copy.deepcopy(node) for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('missing source functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class Array:
    addresses = itertools.count(4096, 4096)

    def __init__(self, shape, dtype='i32', device='cuda:0'):
        self.shape = tuple(shape)
        self.size = math.prod(shape)
        self.dtype = dtype
        self.device = device
        stride = 8 if dtype == 'f64' else 4
        values = []
        for dimension in reversed(shape):
            values.append(stride)
            stride *= dimension
        self.strides = tuple(reversed(values))
        self.ptr = next(self.addresses)
        self.is_contiguous = True
        self.numeric_generation = 0

    def numpy(self):
        raise AssertionError('cold metadata must not read GPU values')


def arena(worlds=8, rows=12, contacts=7, nv=3):
    checks = []
    storage = {name: Array((worlds * rows,), 'f64' if name == 'head_mu' else 'i32')
               for name in ('compact_heads', 'head_type', 'head_id', 'head_dim', 'head_mu')}
    storage['head_friction'] = Array((worlds * 5 * rows,), 'f64')
    for name in ('compact_count', 'compact_valid', 'head_cone_valid'):
        storage[name] = Array((worlds,))
    plan = SimpleNamespace(validate_private_bindings=lambda: checks.append('private'))
    workspace = SimpleNamespace(nworld=worlds, rows=rows, contacts=contacts,
        counts=Array((worlds * 5,)), ncon=Array((worlds,)),
        _cuda_constraints_storage=storage, _cuda_constraint_plans=plan)
    forward = SimpleNamespace(s=SimpleNamespace(workspace=workspace,
        validate=lambda: checks.append('adapter')), runtime_shape=SimpleNamespace(nv=nv))
    ws = SimpleNamespace(nworld=worlds, rows=rows, contacts=contacts,
                         qacc=SimpleNamespace(shape=(worlds, nv)),
                         qM=SimpleNamespace(shape=(worlds, 6)))
    bundle = SimpleNamespace(runtime=SimpleNamespace(forward=forward),
        newton=SimpleNamespace(solver=SimpleNamespace(ws=ws)))
    return forward, bundle, checks


class ConstraintMetadataTests(unittest.TestCase):
    def setUp(self):
        self.env = {}
        self.namespace = {'os': SimpleNamespace(environ=self.env), 'wp': SimpleNamespace(
            array=Array, int32='i32', float64='f64')}
        source_functions('host/bindings/constraints.py',
            {'constraint_feature_selection', '_head_metadata_arrays', 'newton_head_metadata'}, self.namespace)
        source_functions('host/bindings/capacity.py', {'_resolve_head_metadata'}, self.namespace)
        self.select = self.namespace['constraint_feature_selection']
        self.describe = self.namespace['newton_head_metadata']
        self.resolve = self.namespace['_resolve_head_metadata']

    def test_default_n8_all_formal_shapes_preserves_disabled_features(self):
        for nv in (21, 22, 24, 27, 30, 33, 36, 37, 39, 43, 45, 51, 57):
            result = self.select('fast', nv, 8, 'single_cta')
            self.assertEqual(set(result['flags'].values()), {0})
            self.assertEqual(result['mode'], 'retained_defaults')
            self.assertFalse(result['candidate_pending_gpu'])
        for profile, nv, worlds in (('compatible', 43, 1), ('fast', 21, 1), ('fast', 43, 64)):
            self.assertEqual(set(self.select(profile, nv, worlds, 'single_cta')['flags'].values()), {0})

    def test_retained_small_nv43_defaults_and_common_zero_override(self):
        for worlds in (1, 2):
            flags = self.select('fast', 43, worlds, 'single_cta')['flags']
            self.assertEqual({k for k, value in flags.items() if value},
                {'BATCH_COMPACT', 'GROUP_PARAMETER_REUSE', 'UNIFIED_METADATA', 'INTERNAL_OWNER_RECEIPT'})
        result = self.select('fast', 43, 8, 'single_cta', environ={
            'LIBERO_CUDA_S_COMMON_ZERO': '1', 'LIBERO_CUDA_S_UNIFIED_METADATA': '1'})
        self.assertEqual(result['flags']['COMMON_ANCESTOR_ZERO'], 1)
        self.assertEqual(result['flags']['UNIFIED_METADATA'], 1)

    def test_isolated_modes_and_combination_do_not_enable_other_experiments(self):
        for mode, expected in (('baseline', (0, 0)), ('unified', (1, 0)),
                               ('group', (0, 1)), ('unified_group', (1, 1))):
            result = self.select('fast', 22, 8, 'single_cta', qualification_mode=mode)
            self.assertEqual((result['flags']['UNIFIED_METADATA'],
                              result['flags']['GROUP_PARAMETER_REUSE']), expected)
            self.assertTrue(result['candidate_pending_gpu'])
            self.assertEqual(result['flags']['BATCH_COMPACT'], 0)
            self.assertEqual(result['flags']['INTERNAL_OWNER_RECEIPT'], 0)
            self.assertEqual(result['flags']['STATIC_CANDIDATES'], 0)
            self.assertEqual(result['flags']['COMMON_ANCESTOR_ZERO'], 0)
        self.env['LIBERO_CUDA_S_QUALIFICATION_MODE'] = 'group'
        self.assertEqual(self.select('fast', 57, 8, 'single_cta')['mode'], 'group')

    def test_candidate_mode_rejects_domain_and_conflicting_flags(self):
        for profile, worlds, mapping in (('compatible', 8, 'single_cta'),
                                        ('fast', 1, 'single_cta'), ('fast', 8, 'contact_tiles')):
            with self.assertRaisesRegex(ValueError, 'fast N8 single_cta'):
                self.select(profile, 43, worlds, mapping, qualification_mode='unified')
        for name, value in (('UNIFIED_METADATA', '0'), ('GROUP_PARAMETER_REUSE', '1'),
                            ('BATCH_COMPACT', '1'), ('COMMON_ZERO', '1')):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                self.select('fast', 43, 8, 'single_cta', qualification_mode='unified',
                            environ={'LIBERO_CUDA_S_' + name: value})
        with self.assertRaisesRegex(ValueError, 'qualification mode'):
            self.select('fast', 43, 8, 'single_cta', qualification_mode='unknown')
        with self.assertRaisesRegex(ValueError, '0 or 1'):
            self.select('fast', 43, 8, 'single_cta', environ={'LIBERO_CUDA_S_UNIFIED_METADATA': '2'})

    def test_descriptor_retains_exact_owners_and_explicit_pitches(self):
        forward, _, checks = arena()
        metadata = self.describe(forward)
        self.assertIs(self.describe(forward), metadata)
        sw = forward.s.workspace
        self.assertIs(metadata['head_dim'], sw._cuda_constraints_storage['head_dim'])
        self.assertIs(metadata['source_counts'], sw.counts)
        self.assertIs(metadata['source_ncon'], sw.ncon)
        self.assertEqual(metadata.world_strides['head_dim'], 12)
        self.assertEqual(metadata.world_strides['head_friction'], 60)
        self.assertEqual(metadata.world_strides['source_counts'], 5)
        self.assertEqual(metadata.world_strides['valid'], 1)
        report = metadata.manifest()
        self.assertEqual(report['source_count_fields'], ['ne', 'nf', 'nlimit', 'nefc', 'nnzJ'])
        self.assertEqual(report['fields']['head_friction']['logical_shape'], [8, 5, 12])
        self.assertFalse(report['numeric_input_copy'])
        self.assertNotIn('pointer', str(report))
        self.assertIn('adapter', checks)
        with self.assertRaises(FrozenInstanceError):
            metadata.rows = 3
        with self.assertRaises(TypeError):
            metadata.world_strides['head_dim'] = 3

    def test_seal_does_not_authenticate_or_download_numeric_contents(self):
        forward, _, _ = arena()
        metadata = self.describe(forward)
        metadata['head_mu'].numeric_generation += 1
        metadata['source_counts'].numeric_generation += 1
        metadata.validate()  # Current-value authentication belongs to N's device gate.

    def test_owner_pointer_shape_stride_dtype_and_root_drift_rejected(self):
        changes = (
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['head_mu'], 'ptr', 999),
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['head_dim'], 'shape', (8, 12)),
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['head_mu'], 'strides', (16,)),
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['head_mu'], 'dtype', 'i32'),
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['valid'] if 'valid' in f.s.workspace._cuda_constraints_storage
                              else f.s.workspace._cuda_constraints_storage['head_cone_valid'], 'device', 'cuda:1'),
            lambda f: f.s.workspace._cuda_constraints_storage.update(head_mu=Array((96,), 'f64')),
            lambda f: setattr(f.s.workspace, 'rows', 13),
            lambda f: setattr(f, 's', SimpleNamespace(workspace=f.s.workspace, validate=lambda: None)),
        )
        for change in changes:
            forward, _, _ = arena()
            metadata = self.describe(forward)
            change(forward)
            with self.assertRaises((ValueError, TypeError)):
                metadata.validate()
        forward, _, _ = arena()
        metadata = self.describe(forward)
        forward._cuda_s_head_metadata = self.describe(arena()[0])
        with self.assertRaisesRegex(ValueError, 'descriptor owner'):
            self.describe(forward)

    def test_typed_layout_accepts_canonical_without_creating_views(self):
        forward, _, _ = arena()
        storage = forward.s.workspace._cuda_constraints_storage
        storage['head_friction'] = Array((8, 5, 12), 'f64')
        expected = storage['head_friction']
        metadata = self.describe(forward)
        self.assertIs(metadata['head_friction'], expected)
        self.assertEqual(metadata['head_friction'].strides, (480, 96, 8))

    def test_missing_noncontiguous_wrong_size_or_dtype_is_not_sealed(self):
        for change in (
            lambda f: f.s.workspace._cuda_constraints_storage.pop('head_id'),
            lambda f: setattr(f.s.workspace._cuda_constraints_storage['head_mu'], 'is_contiguous', False),
            lambda f: f.s.workspace._cuda_constraints_storage.update(head_mu=Array((95,), 'f64')),
            lambda f: f.s.workspace._cuda_constraints_storage.update(head_mu=Array((96,), 'i32')),
        ):
            forward, _, _ = arena()
            change(forward)
            with self.assertRaises((ValueError, KeyError)):
                self.describe(forward)

    def test_capacity_default_off_has_no_forward_or_gpu_dependency(self):
        self.assertIsNone(self.resolve(object(), 'compatible', False, object(), None))
        with self.assertRaisesRegex(ValueError, '0 or 1'):
            self.resolve(object(), 'fast', False, object(), None,
                         environ={'LIBERO_CUDA_S_ORDINAL_METADATA': '2'})

    def test_capacity_optin_is_independent_and_rejects_foreign_model_owner(self):
        forward, bundle, _ = arena()
        metadata = self.describe(forward)
        config = SimpleNamespace(strategy='single_cta')
        self.assertIs(self.resolve(bundle, 'fast', True, config, metadata), metadata)
        fake_module = SimpleNamespace(newton_head_metadata=self.describe)
        with patch.dict(sys.modules, {'libero.gpu.host.bindings.constraints': fake_module}):
            self.assertIs(self.resolve(bundle, 'fast', True, config, None,
                environ={'LIBERO_CUDA_S_ORDINAL_METADATA': '1'}), metadata)
        for profile, combine, strategy in (('compatible', True, 'single_cta'),
                                           ('fast', False, 'single_cta'), ('fast', True, 'multi_cta')):
            with self.assertRaisesRegex(ValueError, 'fast N8'):
                self.resolve(bundle, profile, combine, SimpleNamespace(strategy=strategy), metadata)
        foreign = self.describe(arena()[0])  # Same dimensions are not the same model/arena owner.
        with self.assertRaisesRegex(ValueError, 'another forward owner'):
            self.resolve(bundle, 'fast', True, config, foreign)
        with self.assertRaisesRegex(TypeError, 'sealed HeadMetadata'):
            self.resolve(bundle, 'fast', True, config, dict(metadata))
        bundle.newton.solver.ws.rows = 13
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.resolve(bundle, 'fast', True, config, metadata)

    def test_existing_core_reuse_distinguishes_ordinal_request(self):
        source_functions('host/bindings/capacity.py', {'install'}, self.namespace)
        self.namespace.update(Path=Path, qualify=lambda *args, **kwargs: {'admitted_by_formula': True},
                              inspect_resources=lambda core: {'passed': True})
        created = []
        class Core:
            def __init__(self, solver, root, **options):
                created.append(options)
                self.manifest = dict(ordinal_head_metadata={'requested': options['head_metadata'] is not None})
            def install(self):
                pass
        modules = {
            'libero.gpu.host.runtime.newton.handwritten_cuda': SimpleNamespace(HandwrittenCudaCore=Core),
            'libero.gpu.host.runtime.core.execution_config': SimpleNamespace(
                newton_execution_config=lambda **_: SimpleNamespace(block_threads=128, strategy='single_cta')),
            'libero.gpu.host.runtime.newton.algorithm_options': SimpleNamespace(
                normalize_newton_algorithm_options=lambda _: {'newton_islands': 'off'},
                effective_newton_algorithm_configuration=lambda *_: {}),
        }
        for previously_requested in (False, True):
            _, bundle, _ = arena()
            solver = bundle.newton.solver
            solver.launch_device = lambda: None
            old = SimpleNamespace(solver=solver, original=solver.launch_device, block_size=128,
                manifest=dict(cholesky_backend='cusolverdx', diagnostic_profile='fast',
                              hessian_algorithm={}, ordinal_head_metadata={'requested': previously_requested}))
            bundle.persistent_cuda_core = old
            before = len(created)
            with patch.dict(sys.modules, modules):
                result = self.namespace['install'](bundle, '/unused', dataflow_profile='fast')
            self.assertEqual(len(created) - before, int(previously_requested))
            self.assertEqual(result is old, not previously_requested)


if __name__ == '__main__':
    unittest.main()
