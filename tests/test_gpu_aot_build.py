"""Offline instance coverage and strict runtime cache-miss boundaries."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from libero.gpu.host.build.compilation import (compilation_scope, require_compilation,
    compilation_allowed, MissingCompiledProduct, install_warp_guard, record_event)

GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
source_cache = runpy.run_path(str(GPU/'host/runtime/core/specialization_cache.py'))


def observation_fixture(aot, root):
    """A complete temporary product bank with independently hashed files."""
    contract = aot.observation_product_contract()
    native, warp, profiles = {}, {}, {}
    for name, spec in contract['profiles'].items():
        manifests = []
        names = ['direct_observations'] + (['handwritten_scene'] if name == 'dual-rgb128' else [])
        for product in names:
            path = root/'cuda'/name/(product+'.so')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name+product).encode())
            digest = aot._sha(path)
            native[str(path.relative_to(root))] = digest
            manifests.append(dict(name=product, library=str(path), binary_sha256=digest))
        modules = {}
        for module in aot._observation_warp_modules(name):
            artifacts = {}
            for suffix in ('cubin', 'meta'):
                path = root/'warp'/module/('module.'+suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((module+suffix).encode())
                spelling, digest = str(path.relative_to(root)), aot._sha(path)
                artifacts[spelling] = warp[spelling] = digest
            modules[module] = dict(artifacts=artifacts)
        profiles[name] = dict(spec=spec, captured_features=aot._observation_features(name),
            capture='discarded-force-module-load', replayed=False, dataflow_profile='fast', debug=False,
            installation=dict(installed=True, feature_key=name, dataflow_profile='fast', debug=False,
                batch_observation_abi='direct-v1', graph_guard_registered=True, native_artifacts=manifests),
            native_artifacts={str(Path(m['library']).relative_to(root)): m['binary_sha256'] for m in manifests},
            warp_modules=modules)
    products = dict(contract=contract, physics_arenas=1, replayed=False,
        host_features=contract['host_features'], profiles=profiles)
    return contract, products, native, warp


class AotBuildTests(unittest.TestCase):
    def test_task_aliases_require_identical_model_bytes_not_just_dimensions(self):
        from libero.gpu.host.build import aot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'instances/libero_10__first/COMPILED.json'
            target.parent.mkdir(parents=True)
            source=dict(key='libero_10/first.bddl',model_sha256='complete-model-A',
                source_identity='source',num_envs=8,seed=0,native_compiler_invocations=4,
                warp_compilations=3,seconds=5,events=[],constraint_package='source-package')
            _, products, native, warp = observation_fixture(aot, root)
            source.update(observation_products=products, native_artifacts=native, warp_artifacts=warp)
            target.write_text(json.dumps(source))
            tasks=[dict(key='libero_10/first.bddl',suite='libero_10',model_sha256='complete-model-A'),
                   dict(key='libero_10/second.bddl',suite='libero_10',model_sha256='complete-model-A'),
                   dict(key='libero_10/third.bddl',suite='libero_10',model_sha256='complete-model-B')]
            rows=aot.propagate_scene_aliases(root,tasks,fingerprint='source',num_envs=8,seed=0)
            self.assertEqual([r['key'] for r in rows],['libero_10/second.bddl'])
            self.assertEqual(rows[0]['native_compiler_invocations'],0)
            self.assertFalse((root/'instances/libero_10__third/COMPILED.json').exists())

    def test_legacy_instance_rows_cannot_propagate_observation_readiness(self):
        from libero.gpu.host.build import aot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'instances/libero_10__first/COMPILED.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(key='libero_10/first.bddl',model_sha256='same-model',
                source_identity='source',num_envs=8,seed=0)))
            task=dict(key='libero_10/second.bddl',suite='libero_10',model_sha256='same-model')
            self.assertEqual(aot.propagate_scene_aliases(root,[task],fingerprint='source',num_envs=8,seed=0), [])

    def test_load_only_policy_covers_warp_worker_threads_and_restores_after_failure(self):
        with compilation_scope(False), ThreadPoolExecutor(1) as executor:
            future = executor.submit(require_compilation, 'GPU product', 'missing')
            with self.assertRaises(MissingCompiledProduct):
                future.result()
        self.assertTrue(compilation_allowed())

    def test_warp_cache_hits_load_and_misses_are_rejected_before_codegen(self):
        class Module:
            name = 'libero.fake'
            def __init__(self, cached):
                self.cached = cached
                self.generated = False
            def _compile(self, *args, **kwargs):
                if self.cached:
                    return False
                self._run_codegen()
                self.cached = True
                return True
            def _run_codegen(self):
                self.generated = True
        warp = ModuleType('warp')
        src = ModuleType('warp._src')
        context = ModuleType('warp._src.context')
        warp._src, src.context, context.Module = src, context, Module
        with patch.dict(sys.modules, {'warp': warp, 'warp._src': src, 'warp._src.context': context}):
            install_warp_guard()
            ready, missing = Module(True), Module(False)
            with compilation_scope(False):
                self.assertFalse(ready._compile())
                with self.assertRaises(MissingCompiledProduct):
                    missing._compile()
            self.assertFalse(missing.generated)
            with compilation_scope(True) as scope:
                self.assertTrue(missing._compile())
            self.assertTrue(missing.generated)
            self.assertEqual(len(scope['events']), 1)
            self.assertEqual(scope['events'][0]['kind'], 'warp')

    def test_preinstantiated_source_load_skips_generator_and_checks_bytes(self):
        manifest = {'nv': 21}
        def generator(model, static, root):
            root.mkdir()
            (root/'__init__.py').write_text('')
            (root/'MODEL_MANIFEST.json').write_text(json.dumps(model))
            raw = b'NV = 21\n'
            (root/'identity.py').write_bytes(raw)
            return dict(template_pins={}, generated_pins={'identity': {
                'path': str(root/'identity.py'), 'sha256': hashlib.sha256(raw).hexdigest()}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, _, _ = source_cache['generated_package'](manifest, {}, generator, root)
            source_cache['index_packages'](root, [package])
            def forbidden(*args):
                raise AssertionError('runtime regenerated a template')
            with compilation_scope(False):
                loaded, _, hit = source_cache['generated_package'](manifest, {}, forbidden, root)
                self.assertTrue(hit)
                self.assertEqual(loaded, package)
                with self.assertRaises(MissingCompiledProduct):
                    source_cache['generated_package']({'nv': 43}, {}, forbidden, root)
                (package/'identity.py').write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'prebuilt specialization source differs'):
                    source_cache['generated_package'](manifest, {}, forbidden, root)

    def test_parameter_table_covers_all_shipped_tasks_and_newton_dimensions(self):
        path = GPU/'data/libero_task_parameters.json'
        if not path.exists():
            self.skipTest('parameter census is still generating')
        data = json.loads(path.read_text())
        self.assertEqual(data['count'], 130)
        keys = {row['key'] for row in data['tasks']}
        bddl = GPU.parent/'bddl_files'
        expected = {str(p.relative_to(bddl)) for p in bddl.rglob('*.bddl')}
        self.assertEqual(keys, expected)
        dimensions = {(row['dimensions']['nv'], row['dimensions']['nM']) for row in data['tasks']}
        self.assertEqual(dimensions, {(r['nv'], r['nM']) for r in data['newton_instances']})


class ObservationAotReadinessTests(unittest.TestCase):
    def setUp(self):
        from libero.gpu.host.build import aot
        self.aot = aot
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract, products, native, warp = observation_fixture(aot, self.root)
        self.row = dict(key='suite/task.bddl',source_identity='source',num_envs=8,seed=0,
            model_sha256='model', native_artifacts=native, warp_required=sorted(warp),
            warp_artifacts=warp, observation_products=products)
        self.record = dict(status='complete',num_envs=8,seed=0,source_identity='source',
            observation_contract=self.contract, tasks={self.row['key']:self.row}, artifacts={**native, **warp})
        self.addCleanup(patch.stopall)
        patch.object(aot, 'source_identity', return_value='source').start()
        patch.object(aot, 'observation_product_contract', return_value=self.contract).start()

    def ready(self, **kwargs):
        (self.root/'AOT_BUILD.json').write_text(json.dumps(self.record))
        return self.aot.require_ready(self.root,'/bddl/suite/task.bddl',num_envs=8,seed=0,**kwargs)

    def test_default_profiles_accept_only_the_recorded_observer_abi(self):
        self.ready()
        self.ready(observation_options=dict(use_camera_obs=True))
        self.ready(observation_options=dict(use_camera_obs=True,
            camera_names=('agentview','robot0_eye_in_hand'),camera_heights=(128,128),camera_widths=128))
        invalid = [dict(use_object_obs=False), dict(use_camera_obs=True,use_object_obs=False),
            dict(use_camera_obs=True,camera_names=['agentview']),dict(use_camera_obs=True,camera_heights=64),
            dict(use_camera_obs=True,camera_depths=True),dict(use_camera_obs=True,camera_segmentations='instance')]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(MissingCompiledProduct):
                self.ready(observation_options=options)
        with self.assertRaises(MissingCompiledProduct):
            self.ready(debug=True)

    def test_legacy_complete_and_missing_cold_capture_evidence_are_not_ready(self):
        baseline = deepcopy(self.record)
        mutations = [lambda: self.record.pop('observation_contract'),
            lambda: self.row.pop('observation_products'),
            lambda: self.row['observation_products']['profiles']['dual-rgb128'].pop('installation'),
            lambda: self.row['observation_products']['profiles']['no-camera'].update(captured_features=[]),
            lambda: self.row['observation_products'].update(replayed=True),
            lambda: self.row['observation_products']['profiles']['dual-rgb128'].update(dataflow_profile='compatible')]
        for mutation in mutations:
            self.record = deepcopy(baseline)
            self.row = self.record['tasks']['suite/task.bddl']
            mutation()
            with self.subTest(mutation=mutations.index(mutation)), self.assertRaises(MissingCompiledProduct):
                self.ready()

    def test_each_profile_requires_its_native_and_loaded_warp_artifact_binding(self):
        baseline = deepcopy(self.record)
        mutations = [lambda: self.row['native_artifacts'].clear(),
            lambda: self.row['warp_artifacts'].clear(),
            lambda: self.row['observation_products']['profiles']['dual-rgb128']['warp_modules'].pop(
                'libero.gpu.host.adapters.observations.scene_kinematics'),
            lambda: self.row['observation_products']['profiles']['no-camera']['installation'].update(native_artifacts=[]),
            lambda: self.record['artifacts'].pop(next(iter(self.row['native_artifacts']))),
            lambda: self.record['artifacts'].pop(next(iter(self.row['warp_artifacts'])))]
        for mutation in mutations:
            self.record = deepcopy(baseline)
            self.row = self.record['tasks']['suite/task.bddl']
            mutation()
            with self.subTest(mutation=mutations.index(mutation)), self.assertRaises(MissingCompiledProduct):
                self.ready()

    def test_file_tampering_is_rejected_after_the_contract_check(self):
        path=self.root/next(iter(self.row['warp_artifacts']))
        path.write_bytes(b'changed Warp binary')
        with self.assertRaisesRegex(ValueError,'prebuilt artifact differs'):
            self.ready()

    def test_batch_constructor_source_changes_invalidate_the_explicit_contract(self):
        changed=deepcopy(self.contract)
        changed['source_files']['host/adapters/environment/batch_env.py']='changed-but-excluded-from-global-pin'
        with patch.object(self.aot,'observation_product_contract',return_value=changed):
            with self.assertRaises(MissingCompiledProduct):
                self.ready()

    def test_observer_preparation_cannot_be_entered_from_load_only_scope(self):
        with compilation_scope(False), patch.object(self.aot,'_prepare_observation_profiles') as prepare:
            with self.assertRaises(MissingCompiledProduct):
                self.aot.prepare_observation_products(object(),self.root)
        prepare.assert_not_called()

    def test_resume_checks_warp_sha_before_constructing_or_claiming_a_hit(self):
        path=self.root/'instances/suite__task/COMPILED.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.row))
        warp=ModuleType('warp')
        warp.init=lambda: None
        warp.set_mempool_release_threshold=lambda *args: None
        envs=ModuleType('libero.libero.envs')
        envs.GpuBatchEnv=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('rebuild-entered'))
        job=(dict(suite='suite',key='suite/task.bddl',directory=str(self.root)),str(self.root),8,0,'cuda:0','source',self.contract)
        with patch.dict(sys.modules,{'warp':warp,'libero.libero.envs':envs}), \
                patch.object(self.aot,'configure_warp_cache'), \
                patch('numpy.load',return_value=object()), \
                patch('libero.libero.get_libero_path',return_value=str(self.root)):
            self.assertTrue(self.aot._compile_instance_locked(job)['resumed'])
            (self.root/next(iter(self.row['warp_artifacts']))).write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'rebuild-entered'):
                self.aot._compile_instance_locked(job)

    def test_worker_prepares_observers_once_on_its_existing_physics_arena(self):
        products=self.row['observation_products']
        native=[m for row in products['profiles'].values() for m in row['installation']['native_artifacts']]
        bundle=SimpleNamespace(cuda_manifests=native)
        preload=Mock(side_effect=lambda: record_event(
            dict(kind='warp', commands_executed=[['mock-branch-warp']])))
        env=SimpleNamespace(_physics=SimpleNamespace(bundle=bundle, prepare_substep_modules=preload),build_report=dict(events=[]),
            built=SimpleNamespace(_gpu_native_intake_binding=SimpleNamespace(package=self.root/'package')),
            diagnostics=lambda:dict(actual_integration_count=[0]*8,first_error=[0]*8,model_sha256='model'))
        calls=[]
        @contextmanager
        def constructor(*args,**kwargs):
            calls.append((args,kwargs))
            self.assertFalse(kwargs['use_camera_obs'])
            self.assertFalse(kwargs['initialize'])
            yield env
        def prepare(owner,root,*,contract,reuse_only=False):
            preload.assert_called_once_with()
            self.assertIs(owner,env)
            self.assertIs(owner._physics.bundle,bundle)
            self.assertEqual(contract,self.contract)
            env.build_report['events'].append(dict(kind='warp',commands_executed=[['mock-warp']]))
            return products
        warp=SimpleNamespace(init=lambda:None,set_mempool_release_threshold=lambda *args:None,
            synchronize_device=lambda *args:None)
        envs=ModuleType('libero.libero.envs')
        envs.GpuBatchEnv=constructor
        job=(dict(suite='suite',key='suite/task.bddl',directory=str(self.root)),str(self.root),8,0,'cuda:0','source',self.contract)
        with patch.dict(sys.modules,{'warp':warp,'libero.libero.envs':envs}), \
                patch.object(self.aot,'configure_warp_cache'),patch('numpy.load',return_value=object()), \
                patch('libero.libero.get_libero_path',return_value=str(self.root)), \
                patch.object(self.aot,'prepare_observation_products',side_effect=prepare) as preparation, \
                patch.object(self.aot,'_loaded_warp_artifacts',return_value={'loaded':dict(artifacts=self.row['warp_artifacts'])}):
            row=self.aot._compile_instance_locked(job)
        self.assertEqual(len(calls),1)
        preparation.assert_called_once()
        self.assertEqual(row['warp_compilations'],2)
        self.assertIn(dict(kind='warp', commands_executed=[['mock-branch-warp']]), row['events'])
        self.assertEqual(row['native_artifacts'],self.row['native_artifacts'])
        self.assertEqual(row['warp_artifacts'],self.row['warp_artifacts'])
        self.assertEqual(set(row['observation_products']['profiles']),{'no-camera','dual-rgb128'})

    def test_catalog_does_not_mark_complete_if_final_artifacts_diverge_from_task_rows(self):
        row=deepcopy(self.row)
        row['constraint_package']=str(self.root/'package')
        task=dict(suite='suite',task_id=0,key='suite/task.bddl',model_sha256='model',dimensions=dict(nv=9,nM=45))
        class Pool:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def imap_unordered(self,function,jobs):return iter([row])
        calls=[]
        def configure(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                (self.root/next(iter(row['warp_artifacts']))).write_bytes(b'changed-after-worker')
        from libero.gpu.host.runtime.core import specialization_cache
        with patch.object(specialization_cache,'index_packages'), \
                patch.object(self.aot,'instantiate_newton'),patch.object(self.aot,'configure_warp_cache',side_effect=configure), \
                patch.object(self.aot,'get_context',return_value=SimpleNamespace(Pool=Pool)), \
                patch.object(self.aot,'propagate_scene_aliases',return_value=[]):
            with self.assertRaisesRegex(ValueError,'differ from final artifacts'):
                self.aot.compile_catalog(self.root,[task])
        self.assertEqual(json.loads((self.root/'AOT_BUILD.json').read_text())['status'],'building')

    @contextmanager
    def reuse_worker_harness(self, *, missing=None, reported_command=False, close_command=False):
        products=self.row['observation_products']
        native=[m for row in products['profiles'].values() for m in row['installation']['native_artifacts']]
        preload=Mock(side_effect=lambda: self.assertFalse(compilation_allowed()))
        env=SimpleNamespace(_physics=SimpleNamespace(bundle=SimpleNamespace(cuda_manifests=native),
                                                   prepare_substep_modules=preload),
            build_report=dict(events=[]),
            built=SimpleNamespace(_gpu_native_intake_binding=SimpleNamespace(package=self.root/'package')),
            diagnostics=lambda:dict(actual_integration_count=[0]*8,first_error=[0]*8,model_sha256='model'))
        calls=[]
        @contextmanager
        def constructor(*args,**kwargs):
            calls.append(kwargs)
            self.assertTrue(kwargs['allow_compile'])  # official builder skips global ready gate
            self.assertFalse(compilation_allowed())
            with compilation_scope(True),ThreadPoolExecutor(1) as executor:
                self.assertFalse(compilation_allowed())
                self.assertFalse(executor.submit(compilation_allowed).result())
                if missing:require_compilation(missing,'absent-cache-entry')
            env.build_report['events'].append(dict(kind='native-cuda',commands_executed=[['nvcc']] if reported_command else []))
            yield env
            if close_command:
                env.build_report['events'].append(dict(kind='warp',commands_executed=[['late-warp']]))
        def prepare(owner,root,*,contract,reuse_only):
            preload.assert_called_once_with()
            self.assertIs(owner,env)
            self.assertTrue(reuse_only)
            self.assertFalse(compilation_allowed())
            return products
        warp=SimpleNamespace(init=lambda:None,set_mempool_release_threshold=lambda *args:None,
            synchronize_device=lambda *args:None)
        envs=ModuleType('libero.libero.envs');envs.GpuBatchEnv=constructor
        task=dict(suite='suite',task_id=0,key='suite/task.bddl',model_sha256='model',
            dimensions=dict(nv=9,nM=45),directory=str(self.root))
        job=(task,str(self.root),8,0,'cuda:0','source',self.contract,True)
        with patch.dict(sys.modules,{'warp':warp,'libero.libero.envs':envs}), \
                patch.object(self.aot,'configure_warp_cache') as cache,patch('numpy.load',return_value=object()), \
                patch('libero.libero.get_libero_path',return_value=str(self.root)), \
                patch.object(self.aot,'prepare_observation_products',side_effect=prepare), \
                patch.object(self.aot,'_loaded_warp_artifacts',return_value={'loaded':dict(artifacts=self.row['warp_artifacts'])}):
            cache.side_effect=lambda *args,**kwargs:self.assertFalse(compilation_allowed())
            yield job,calls

    def old_completed_row(self):
        path=self.root/'instances/suite__task/COMPILED.json';path.parent.mkdir(parents=True)
        previous=dict(self.row,source_identity='old-host-fingerprint')
        path.write_text(json.dumps(previous))
        return path,path.read_bytes()

    def test_reuse_only_constructs_for_a_new_fingerprint_and_publishes_after_zero_commands(self):
        path,previous=self.old_completed_row()
        with self.reuse_worker_harness() as (job,calls):
            row=self.aot._compile_instance(job)
        self.assertEqual(len(calls),1)
        self.assertTrue(row['reuse_only'])
        self.assertEqual(row['refresh_kind'],'reuse_validation_refresh')
        self.assertEqual(row['source_identity'],'source')
        self.assertEqual(row['native_compiler_invocations'],0)
        self.assertEqual(row['warp_compilations'],0)
        self.assertNotEqual(path.read_bytes(),previous)
        self.assertEqual(json.loads(path.read_text()),row)
        self.assertTrue(compilation_allowed())

    def test_reuse_only_native_and_warp_misses_fail_without_replacing_old_qualification(self):
        path,previous=self.old_completed_row()
        for missing in ('native CUDA','Warp module'):
            with self.subTest(missing=missing),self.reuse_worker_harness(missing=missing) as (job,calls):
                with self.assertRaises(MissingCompiledProduct):self.aot._compile_instance(job)
                self.assertEqual(len(calls),1)
                self.assertEqual(path.read_bytes(),previous)
            self.assertTrue(compilation_allowed())

    def test_reuse_only_command_reports_including_close_events_prevent_compiled_publication(self):
        path,previous=self.old_completed_row()
        for reported,late in ((True,False),(False,True)):
            with self.subTest(reported=reported,late=late),self.reuse_worker_harness(
                    reported_command=reported,close_command=late) as (job,calls):
                with self.assertRaisesRegex(RuntimeError,'compiler commands'):
                    self.aot._compile_instance(job)
                self.assertEqual(path.read_bytes(),previous)

    def test_reuse_only_observer_preparation_requires_and_preserves_outer_freeze(self):
        env=SimpleNamespace(build_report=dict(events=[]))
        with self.assertRaisesRegex(ValueError,'requires a frozen'):
            self.aot.prepare_observation_products(env,self.root,reuse_only=True)
        def prepare(*args,**kwargs):
            self.assertFalse(compilation_allowed())
            return self.row['observation_products']
        with compilation_scope(False),patch.object(self.aot,'_prepare_observation_profiles',side_effect=prepare):
            self.aot.prepare_observation_products(env,self.root,reuse_only=True)
        from libero.gpu.host.build.compilation import record_event
        def bypassed_compiler(*args,**kwargs):
            record_event(dict(kind='warp',commands_executed=[['unexpected-compile']]))
            return self.row['observation_products']
        with compilation_scope(False),patch.object(self.aot,'_prepare_observation_profiles',side_effect=bypassed_compiler):
            with self.assertRaisesRegex(RuntimeError,'compiler commands'):
                self.aot.prepare_observation_products(env,self.root,reuse_only=True)

    def test_reuse_only_catalog_freezes_parent_and_delivers_explicit_worker_mode(self):
        from libero.gpu.host.runtime.core import specialization_cache
        owner=self
        class Pool:
            def __init__(self,*args,**kwargs):owner.assertFalse(compilation_allowed())
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def imap_unordered(self,function,jobs):
                owner.assertTrue(all(job[7] is True for job in jobs))
                return (function(job) for job in jobs)
        def frozen(*args,**kwargs):
            self.assertFalse(compilation_allowed())
            with compilation_scope(True):self.assertFalse(compilation_allowed())
        with self.reuse_worker_harness() as (job,calls), \
                patch.object(specialization_cache,'index_packages',side_effect=frozen), \
                patch.object(self.aot,'instantiate_newton',side_effect=frozen), \
                patch.object(self.aot,'get_context',return_value=SimpleNamespace(Pool=Pool)):
            result=self.aot.compile_catalog(self.root,[job[0]],reuse_only=True)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['reuse_only'])
        self.assertEqual(result['refresh_kind'],'reuse_validation_refresh')
        self.assertTrue(compilation_allowed())

    def test_reuse_only_catalog_refuses_parent_commands_or_nonzero_worker_counts(self):
        from libero.gpu.host.runtime.core import specialization_cache
        from libero.gpu.host.build.compilation import record_event
        task=dict(suite='suite',task_id=0,key='suite/task.bddl',model_sha256='model',dimensions=dict(nv=9,nM=45))
        row=dict(self.row,constraint_package=str(self.root/'package'),events=[],native_compiler_invocations=0,warp_compilations=0)
        class Pool:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def imap_unordered(self,function,jobs):return iter([row])
        for parent_commands in (True,False):
            def prepare(*args):
                if parent_commands:record_event(dict(kind='native-cuda',commands_executed=[['unexpected-parent-compile']]))
            row['native_compiler_invocations']=0 if parent_commands else 1
            with self.subTest(parent_commands=parent_commands), \
                    patch.object(specialization_cache,'index_packages'),patch.object(self.aot,'instantiate_newton',side_effect=prepare), \
                    patch.object(self.aot,'configure_warp_cache'),patch.object(self.aot,'propagate_scene_aliases',return_value=[]), \
                    patch.object(self.aot,'get_context',return_value=SimpleNamespace(Pool=Pool)):
                with self.assertRaisesRegex(RuntimeError,'compiler'):
                    self.aot.compile_catalog(self.root,[task],reuse_only=True)
                self.assertNotEqual(json.loads((self.root/'AOT_BUILD.json').read_text())['status'],'complete')

    def test_reuse_only_mode_requires_an_explicit_bool_before_any_preparation(self):
        with patch.object(self.aot,'instantiate_newton') as prepare:
            with self.assertRaisesRegex(ValueError,'reuse_only must be bool'):
                self.aot.compile_catalog(self.root,[],reuse_only=1)
            with self.assertRaisesRegex(ValueError,'reuse_only must be bool'):
                self.aot._compile_instance(({},str(self.root),8,0,'cuda:0','source',self.contract,1))
        prepare.assert_not_called()


class ObservationAotColdPreparationTests(unittest.TestCase):
    def test_loaded_warp_receipts_use_only_executed_module_contexts_and_hash_actual_files(self):
        from libero.gpu.host.build import aot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            device=SimpleNamespace(context=123)
            module=SimpleNamespace(name='fallback',execs={(123,64):object(),(999,128):object()},
                get_module_identifier=lambda block_dim:'fallback-'+str(block_dim),
                _get_compile_output_name=lambda device,block_dim:'module.cubin',
                _get_meta_name=lambda block_dim:'module.meta')
            cache=root/'warp'/'fallback-64'
            cache.mkdir(parents=True)
            for name in ('module.cubin','module.meta'):(cache/name).write_bytes(name.encode())
            wp=SimpleNamespace(get_device=lambda value:device,config=SimpleNamespace(kernel_cache_dir=str(root/'warp')))
            context=SimpleNamespace(user_modules={'fallback':module})
            with patch.dict(sys.modules,{'warp':wp,'warp._src.context':context}):
                loaded=aot._loaded_warp_artifacts(root,device,required_modules=['fallback'])
                expected={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in cache.iterdir()}
                self.assertEqual(loaded,{'fallback':dict(artifacts=expected)})
                with self.assertRaisesRegex(ValueError,'were not loaded'):
                    aot._loaded_warp_artifacts(root,device,required_modules=['fallback','absent'])

    def test_actual_installer_prepares_both_profiles_on_one_arena_without_replay(self):
        from libero.gpu.host.build import aot
        from libero.gpu.host.build.compilation import record_event
        # Reuse the real install_batch AST harness; only device/storage primitives
        # are injected. This catches project-root and parent-scope ABI mismatches.
        harness=runpy.run_path(str(Path(__file__).with_name('test_gpu_batch_direct_observations.py')))
        case=harness['BatchDirectObservationTests']()
        case.setUp()
        case.stream.device=case.device
        case.runtime.valid=object()
        original_hook=case.runtime.observe
        log=[]
        observers=[]
        digest='d'*64
        old_sampler=case.namespace['DirectSampler']
        def sampler(*args,**kwargs):
            result=old_sampler(*args,**kwargs)
            result.sample=lambda *args,**kwargs: log.append('direct')
            for plan, name in ((result.plan,'direct_observations'),
                    (getattr(args[0].scene_transforms,'_cuda_scene_plan',None),'handwritten_scene')):
                if plan is not None:
                    plan.manifest.update(name=name,binary_sha256=digest)
            result.manifest.update(result.plan.manifest)
            return result
        case.namespace['DirectSampler']=sampler
        def observer_factory(model,metadata,nworld,device):
            self.assertEqual(nworld,8)
            self.assertIs(device,case.device)
            images=[dict(name=name+'_image',shape=(128,128,3)) for name in metadata['camera_names']]
            observer=case.observer_for(images=images)
            observer.metadata=metadata
            observer.device=device
            def baseline(*args,**kwargs):
                self.assertTrue(case.stream.is_capturing)
                log.append('fallback')
                record_event(dict(kind='warp',commands_executed=[['mock-warp',len(log)]]))
            observer.sample=baseline
            observer.lowdim.reset=lambda **kwargs: log.append('lowdim-reset')
            observer.scene_cache=(SimpleNamespace(ncamera=2,elapsed=object(),captured=object(),scenes=object(),
                reset=lambda **kwargs: self.fail('host-reading scene.reset cannot be captured')) if images else None)
            observer.close=lambda: log.append('close')
            observers.append(observer)
            return observer
        @contextmanager
        def capture(*,stream,force_module_load):
            self.assertTrue(force_module_load)
            self.assertFalse(stream.is_capturing)
            log.append('capture')
            stream.is_capturing=True
            try:yield SimpleNamespace(graph=object())
            finally:stream.is_capturing=False
        warp=SimpleNamespace(ScopedCapture=capture,
            launch=lambda *args,**kwargs: log.append('scene-reset'))
        modules={**case.modules,'warp':warp,
            'libero.gpu.host.adapters.observations.adapter':SimpleNamespace(GpuObservationAdapter=observer_factory),
            'libero.gpu.host.adapters.observations.runtime_binding':SimpleNamespace(
                FixedRuntimeObservationBinding=lambda bundle,observer:case.binding_for(observer,installed=False)),
            'libero.gpu.host.adapters.observations.device':SimpleNamespace(reset_kernel=object()),
            'libero.gpu.host.adapters.observations.scene_kinematics':SimpleNamespace(
                DeviceSceneTransforms=SimpleNamespace(update=lambda *args:log.append('scene-fallback'))),
            'libero.gpu.host.bindings.observations':SimpleNamespace(install_batch=case.install_batch)}
        env=SimpleNamespace(_physics=SimpleNamespace(bundle=case.bundle),stream=case.stream,num_envs=8,
            built=SimpleNamespace(native_model=object(),metadata=dict(model_sha256='exact-model')),
            build_report=case.bundle.build_report)
        contract=aot.observation_product_contract()
        def loaded(root,device,*,required_modules):
            return {name:dict(artifacts={'warp/'+name:digest}) for name in required_modules}
        with patch.dict(sys.modules,modules),patch.object(aot,'_sha',return_value=digest), \
                patch.object(aot,'observation_product_contract',return_value=contract), \
                patch.object(aot,'_loaded_warp_artifacts',side_effect=loaded):
            products=aot.prepare_observation_products(env,'/bank',contract=contract)
        self.assertIs(case.runtime.observe,original_hook)
        self.assertIs(env._physics.bundle,case.bundle)
        self.assertEqual(len(case.installs),2)
        self.assertIs(case.installs[0][1],case.installs[1][1])
        self.assertEqual(log.count('capture'),2)
        self.assertEqual(log.count('direct'),2)
        self.assertEqual(log.count('fallback'),2)
        self.assertEqual(log.count('lowdim-reset'),2)
        self.assertEqual(log.count('scene-fallback'),1)
        self.assertEqual(log.count('scene-reset'),1)
        self.assertEqual(log.count('close'),2)
        self.assertEqual(sum(e['kind']=='warp' for e in env.build_report['events']),2)
        self.assertEqual(sum(e['kind']=='direct-observations' for e in env.build_report['events']),2)
        self.assertEqual(set(products['profiles']),{'no-camera','dual-rgb128'})
        self.assertTrue(all(not p['installation']['runtime_hook_installed'] for p in products['profiles'].values()))
        self.assertFalse(products['replayed'])


if __name__ == '__main__':
    unittest.main()
