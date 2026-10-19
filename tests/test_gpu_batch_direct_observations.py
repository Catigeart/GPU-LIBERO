"""Cold batch sampler installation with injected products; no GPU imports."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1]/'libero/libero/gpu/host/bindings/observations.py'


class Array:
    next_ptr = 1

    def __init__(self, device):
        self.ptr = Array.next_ptr
        Array.next_ptr += 1
        self.shape, self.strides = (8,), (8,)
        self.dtype, self.device, self.ndim, self.is_contiguous = 'float64', device, 1, True


class MissingProduct(RuntimeError):
    pass


class BatchDirectObservationTests(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(arch=80)
        self.stream = SimpleNamespace(is_capturing=False)
        self.physics_plan = self.plan('physics')
        self.runtime = SimpleNamespace(observe=object())
        self.bundle = SimpleNamespace(stream=self.stream, runtime=self.runtime,
            dataflow_profile='fast', debug=False,
            observation_build_options=dict(build_root='/bank', mathdx_root='/actual-sdk',
                                           strict_build_receipts=False),
            build_report=dict(directory='/bank', events=[{'kind': 'physics'}]),
            cuda_native_plans=(self.physics_plan,), cuda_manifests=[self.physics_plan.manifest],
            cuda_selection_manifest={})
        self.fields = {name: Array(self.device) for name in
            ('qpos','qvel','body_xpos','body_xquat','body_xmat','geom_xpos',
             'geom_xmat','site_xpos','site_xmat','subtree_com')}
        self.project = dict(root=Path('/bank'), events=[{'kind': 'already-recorded'}])
        self.active, self.entries, self.installs = [], [], []
        self.observer = self.observer_for()
        self.binding = self.binding_for(self.observer)

        @contextmanager
        def scope(name, value=None):
            self.entries.append(name)
            self.active.append(name)
            try:
                yield value
            finally:
                self.assertEqual(self.active.pop(), name)

        def project(directory=None, *, lock_key=None):
            self.assertEqual(directory, '/bank')
            self.assertEqual(lock_key, 'exact-model')
            return scope('project', self.project)

        self.sdk_lookup = Mock(return_value=Path('/discovered-sdk'))
        self.paths = SimpleNamespace(build_project=project,
            product_build=lambda: scope('product'),
            toolchain_scope=lambda sdk: scope(('toolchain', str(sdk))),
            mathdx_root=self.sdk_lookup, cache_root=lambda: Path('/cache'))
        self.modules = {
            'libero.gpu.host.runtime.core.product_paths': self.paths,
            'libero.gpu.host.build.native': SimpleNamespace(
                strict_build_receipts_scope=lambda enabled: scope(('strict', enabled))),
            'libero.gpu.host.build.product_cache': SimpleNamespace(
                product_cache_scope=lambda root: scope(('cache', str(root)))),
        }

        def sampler(observer, fields, root, *, dataflow_profile):
            self.assertEqual(self.active[:2], ['project', 'product'])
            self.assertIn(('toolchain', '/actual-sdk'), self.active)
            self.assertIn(('stream', id(self.stream)), self.active)
            self.assertIs(fields, self.fields)
            self.installs.append((observer, fields, root, dataflow_profile))
            if getattr(self, 'missing_product', False):
                raise MissingProduct('native observation product absent')
            self.project['events'].append({'kind': 'direct-observations'})
            direct = self.plan('direct-' + ('images' if observer.plan.image_specs else 'no-camera'))
            if observer.scene_transforms is not None:
                observer.scene_transforms._cuda_scene_plan = self.plan('cached-scene')
            return SimpleNamespace(profile=dataflow_profile, plan=direct,
                baseline_sample=observer.sample,
                manifest=dict(direct.manifest, dataflow_profile=dataflow_profile,
                    scene_capture='schema-v2' if observer.depth else 'handwritten native scene'))

        names = ('_batch_array_token', '_batch_scene_arrays', '_batch_feature_key', 'install', 'install_batch')
        nodes = [n for n in ast.parse(SOURCE.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        self.namespace = dict(Path=Path, MethodType=MethodType, DirectSampler=sampler,
            wp=SimpleNamespace(ScopedStream=lambda stream: scope(('stream', id(stream)))))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.namespace)
        self.install_batch = self.namespace['install_batch']

    def plan(self, name):
        return SimpleNamespace(manifest=dict(name=name, sha256=name+'-sha',
            library='/bank/'+name+'.so'), launch=Mock(side_effect=AssertionError('must not launch')))

    def observer_for(self, *, images=(), depth=False):
        observer = SimpleNamespace(metadata={'model_sha256': 'exact-model'},
            lowdim=SimpleNamespace(observed=Array(self.device)),
            plan=SimpleNamespace(image_specs=images), depth=depth,
            visual_sources={'site_rgba': Array(self.device), 'geom_rgba': Array(self.device)},
            scene_transforms=None,
            sample=Mock(side_effect=AssertionError('must not sample')))
        observer.bind_graph_guard = lambda guard: setattr(observer, 'graph_guard', guard)
        if images:
            observer.scene_transforms = SimpleNamespace(**{
                name: Array(self.device) for name in
                ('cam_xpos','cam_xmat','light_xpos','light_xdir','all_worlds')})
            observer.scene_transforms.camera_static = [Array(self.device)]
            observer.scene_transforms.light_static = [Array(self.device)]
        return observer

    def binding_for(self, observer, *, installed=True):
        return SimpleNamespace(prepared=self.bundle, observer=observer,
            stream=self.stream, installed=installed, fields=Mock(side_effect=lambda: self.fields))

    def install(self, observer=None, binding=None, **kwargs):
        with patch.dict(sys.modules, self.modules):
            return self.install_batch(observer or self.observer, binding or self.binding,
                                      self.bundle, **kwargs)

    def test_formal_installs_retained_roots_in_parent_scopes_and_registers_products(self):
        report = self.bundle.build_report
        result = self.install()
        self.assertEqual(result['feature_key'], 'no-camera')
        self.assertEqual(result['dataflow_profile'], 'fast')
        self.assertEqual(result['build_directory'], '/bank/cuda/sm80/batch-observations/v1')
        self.assertEqual(self.installs[0][1], self.fields)
        self.assertIs(self.bundle.build_report, report)
        self.assertEqual(report['events'], [{'kind': 'physics'}, {'kind': 'direct-observations'}])
        self.assertIs(self.bundle.cuda_native_plans[0], self.physics_plan)
        self.assertEqual(len(self.bundle.cuda_native_plans), 2)
        self.assertEqual(len(self.bundle.cuda_manifests), 2)
        self.assertIs(self.bundle.cuda_selection_manifest['batch_observations']['no-camera/fast'], result)
        self.assertIs(report['observation_profiles']['no-camera/fast'], result)
        self.assertTrue(self.observer.graph_guard())
        self.sdk_lookup.assert_not_called()
        self.observer._cuda_direct_sampler.baseline_sample.assert_not_called()

    def test_debug_uses_compatible_without_changing_physics_profile(self):
        result = self.install(debug=True)
        self.assertEqual(result['dataflow_profile'], 'compatible')
        self.assertEqual(result['requested_dataflow_profile'], 'fast')
        self.assertEqual(self.bundle.dataflow_profile, 'fast')
        self.assertIn('no-camera/compatible', self.bundle.build_report['observation_profiles'])

    def test_rgb_and_depth_routes_register_optional_scene_plan(self):
        for depth in (False, True):
            images = [{'name': name+'_image', 'shape': (128,128,3)}
                      for name in ('agentview','robot0_eye_in_hand')]
            if depth:images.append({'name': 'agentview_depth', 'shape': (128,128,1)})
            observer = self.observer_for(images=images, depth=depth)
            binding = self.binding_for(observer, installed=False)
            with self.subTest(depth=depth):
                result = self.install(observer, binding)
                self.assertEqual(len(result['native_artifacts']), 2)
                self.assertFalse(result['runtime_hook_installed'])
                self.assertEqual(result['feature_key'], 'dual-rgb128' if not depth else
                                 self.namespace['_batch_feature_key'](observer))
                self.assertEqual(result['scene_capture'], 'schema-v2' if depth else 'handwritten native scene')
                self.assertTrue(observer.graph_guard())

    def test_alternate_observer_borrows_arena_without_replacing_live_hook(self):
        self.install()
        live = self.observer._cuda_direct_sampler
        hook = self.bundle.runtime.observe
        other = self.observer_for(images=[{'name':'agentview_image','shape':(128,128,3)}])
        binding = self.binding_for(other, installed=False)
        result = self.install(other, binding)
        self.assertIs(self.bundle.runtime.observe, hook)
        self.assertIs(self.observer._cuda_direct_sampler, live)
        self.assertIsNot(other._cuda_direct_sampler, live)
        self.assertIs(self.installs[-1][1], self.installs[0][1])
        self.assertFalse(result['runtime_hook_installed'])
        self.assertEqual(len(self.bundle.cuda_native_plans), 4)

    def test_same_feature_alternate_cannot_replace_live_route_manifest(self):
        live = self.install()
        other = self.observer_for()
        alternate = self.install(other, self.binding_for(other, installed=False))
        self.assertFalse(alternate['runtime_hook_installed'])
        self.assertIs(self.bundle.cuda_selection_manifest['observations'], live)
        self.assertIs(self.bundle.cuda_selection_manifest['batch_observations']['no-camera/fast'], live)
        self.assertIs(self.bundle.build_report['observation_profiles']['no-camera/fast'], live)
        self.assertEqual(len(self.bundle.cuda_native_plans), 3)
        self.assertEqual(len(self.bundle.cuda_manifests), 2)

    def test_idempotent_install_does_not_duplicate_plans_manifests_or_events(self):
        first = self.install()
        events = list(self.bundle.build_report['events'])
        self.assertIs(self.install(), first)
        self.assertEqual(len(self.installs), 1)
        self.assertEqual(len(self.bundle.cuda_native_plans), 2)
        self.assertEqual(len(self.bundle.cuda_manifests), 2)
        self.assertEqual(self.bundle.build_report['events'], events)
        with self.assertRaisesRegex(ValueError, 'owner/profile changed'):
            self.install(debug=True)

    def test_graph_guard_rejects_root_owner_and_pointer_replacement(self):
        self.install()
        old = self.fields['qpos']
        self.fields['qpos'] = Array(self.device)
        self.assertFalse(self.observer.graph_guard())
        self.fields['qpos'] = old
        self.assertTrue(self.observer.graph_guard())
        old.ptr += 100
        self.assertFalse(self.observer.graph_guard())

    def test_graph_guard_rejects_scene_sources_and_visual_owner_changes(self):
        observer = self.observer_for(images=[{'name':'agentview_image','shape':(128,128,3)}])
        self.install(observer, self.binding_for(observer))
        observer.scene_transforms.cam_xpos = Array(self.device)
        self.assertFalse(observer.graph_guard())

    def test_graph_guard_rejects_sampler_callback_and_scene_count_changes(self):
        observer = self.observer_for(images=[{'name':'agentview_image','shape':(128,128,3)}])
        self.install(observer, self.binding_for(observer))
        self.assertTrue(observer.graph_guard())
        callback = observer.sample
        observer.sample = lambda *args: None
        self.assertFalse(observer.graph_guard())
        observer.sample = callback
        self.assertTrue(observer.graph_guard())
        observer.scene_transforms.ncam = 2
        self.assertFalse(observer.graph_guard())

    def test_strict_product_mode_uses_parent_sdk_cache_and_preserves_miss_failure(self):
        self.bundle.observation_build_options['strict_build_receipts'] = True
        self.install(strict_build_receipts=True)
        self.assertIn(('strict', True), self.entries)
        self.assertIn(('cache', '/cache/products'), self.entries)
        other = self.observer_for()
        self.missing_product = True
        with self.assertRaisesRegex(MissingProduct, 'product absent'):
            self.install(other, self.binding_for(other), strict_build_receipts=True)
        self.assertFalse(hasattr(other, '_cuda_direct_sampler'))
        self.assertEqual(self.active, [])

    def test_rejects_capture_foreign_binding_and_receipt_mismatch_before_install(self):
        self.stream.is_capturing = True
        with self.assertRaisesRegex(ValueError, 'cold stream'):
            self.install()
        self.stream.is_capturing = False
        foreign = self.binding_for(self.observer)
        foreign.prepared = object()
        with self.assertRaisesRegex(ValueError, 'retained physics binding'):
            self.install(binding=foreign)
        with self.assertRaisesRegex(ValueError, 'receipt mode'):
            self.install(strict_build_receipts=True)
        self.assertFalse(self.installs)

    def test_direct_sampling_guard_precedes_any_sampler_validation_or_launch(self):
        cls = next(n for n in ast.parse(SOURCE.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'DirectSampler')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        observer = SimpleNamespace(assert_sampling_allowed=Mock(side_effect=ValueError('capture rejected')),
                                   _flat=Mock())
        with self.assertRaisesRegex(ValueError, 'capture rejected'):
            namespace['sample'](SimpleNamespace(observer=observer), {})
        observer._flat.assert_not_called()


if __name__ == '__main__':
    unittest.main()
