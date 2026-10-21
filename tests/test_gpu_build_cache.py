"""Persistent compiler/source reuse checks with no CUDA initialization."""
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import inspect
import linecache
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import tempfile
import textwrap
import types
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
PATHS_NAME = 'libero.gpu.host.runtime.core.product_paths'
paths = ModuleType(PATHS_NAME)
paths.__dict__.update(runpy.run_path(str(GPU / 'host/runtime/core/product_paths.py')))
generated_package = runpy.run_path(
    str(GPU / 'host/runtime/core/specialization_cache.py'))['generated_package']


class NewtonPathAliasTests(unittest.TestCase):
    """Synthetic byte/receipt tests; no CUDA, environment, or compiler calls."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base/'current/Flash-LIBERO'
        self.old_repo = self.base/'previous/Flash-LIBERO'
        self.gpu = self.repo/'libero/libero/gpu'
        self.root = self.base/'project'
        self.directory = self.root/'cuda/sm80/capacity/nv21_nm86'
        self.directory.mkdir(parents=True)
        self.header = self.gpu/'cuda/newton/header.cuh'
        self.header.parent.mkdir(parents=True)
        self.header.write_bytes(b'project header\n')
        self.sdk_header = self.base/'sdk/include/cusolverdx.hpp'
        self.sdk_header.parent.mkdir(parents=True)
        self.sdk_header.write_bytes(b'SDK header A\n')
        self.archive = self.base/'sdk/lib/libcusolverdx.a'
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b'vendor archive\n')
        self.compiler = self.base/'nvcc'
        self.compiler.write_bytes(b'not an executable; pinned fixture only\n')
        fake_environment = ModuleType('libero.gpu.host.build.product_cache')
        fake_environment.compiler_environment_record = lambda: {'sha256': 'compiler-environment'}
        fake_compilation = ModuleType('libero.gpu.host.build.compilation')
        fake_compilation.MissingCompiledProduct = type('MissingCompiledProduct', (RuntimeError,), {})
        fake_compilation.require_compilation = lambda *args: None
        self.modules = {
            'libero.gpu.host.paths': SimpleNamespace(GPU_ROOT=self.gpu, REPO_ROOT=self.repo),
            PATHS_NAME: paths,
            'libero.gpu.host.build.product_cache': fake_environment,
            'libero.gpu.host.build.compilation': fake_compilation}
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        loaded = runpy.run_path(str(GPU/'host/build/aot.py'))
        self.aot = loaded['migrate_newton_path_alias'].__globals__
        self.aot['__package__'] = 'libero.gpu.host.build'
        self.aot['_NEWTON_LEGACY_SOURCE_ROOTS'] = (str(self.old_repo),)
        self.source = '#define NV 21\n#define NM 86\n// preserve CRLF\r\ncurrent kernel\n'
        self.bridge = '#define NV 21\n#include <cusolverdx.hpp>\n// preserve bridge CRLF\r\nbridge\n'
        self.closure = {str(self.header): self.sha(self.header.read_bytes())}
        old_gpu = self.old_repo/'libero/libero/gpu'
        old_closure = {str(old_gpu/self.header.relative_to(self.gpu)): self.closure[str(self.header)]}
        recipe = self.aot['_newton_recipe']
        archive_sha = self.sha(self.archive.read_bytes())
        self.old_key = recipe(self.source, self.bridge, old_closure, archive_sha)
        self.key = recipe(self.source, self.bridge, self.closure, archive_sha)
        self.previous = self.directory/(self.old_key+'.so')
        self.target = self.directory/(self.key+'.so')
        self.binary = b'complete compiled fixture bytes\n'
        self.previous.write_bytes(self.binary)
        for key in (self.old_key, self.key):
            (self.directory/(key+'.cu')).write_bytes(self.source.encode())
            (self.directory/(key+'_bridge.cu')).write_bytes(self.bridge.encode())
        self.inputs = self.dependency_inputs(self.key)
        old_inputs = self.dependency_inputs(self.old_key)
        self.old_dependencies = dict(schema='libero-newton-dependencies-v1', recipe=self.old_key,
            binary_sha256=self.sha(self.binary), inputs=old_inputs,
            inputs_sha256=self.sha(self.canonical(old_inputs)))
        self.dependency_path = self.previous.with_suffix('.dependencies.json')
        self.dependency_path.write_bytes(self.canonical(self.old_dependencies))
        self.metadata = dict(sha256=self.old_key, handwritten=True, nv=21, nM=86,
            source=str(self.previous.with_suffix('.cu')), device_bridge=str(self.directory/(self.old_key+'_bridge.cu')),
            source_sha256=self.sha(self.source.encode()), stage_sources=old_closure,
            library_file=str(self.previous), library_sha256=self.sha(self.binary),
            mathdx_library_sha256=archive_sha,
            floating_policy=dict(fma=True, fast_math=False, prec_div=True, prec_sqrt=True),
            command=self.launch_command(self.old_key), compile_commands=self.commands(self.old_key))
        self.previous.with_suffix('.json').write_bytes(self.canonical(self.metadata))
        spelling=str(self.previous.relative_to(self.root))
        self.record=dict(status='complete',source_identity='historical-global-identity',num_envs=8,seed=0,
            artifacts={spelling:self.sha(self.binary),
                str(self.dependency_path.relative_to(self.root)):self.sha(self.dependency_path.read_bytes())},
            tasks={key:dict(native_artifacts={spelling:self.sha(self.binary)}) for key in ('suite/a.bddl','suite/b.bddl')})
        self.save_record()

    @staticmethod
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def canonical(value):
        return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

    def file_record(self, path):
        raw=path.read_bytes()
        return dict(path=str(path),bytes=len(raw),sha256=self.sha(raw))

    def commands(self, key):
        base=str(self.directory/key)
        common=[str(self.compiler),'-arch=sm_80','--fmad=true']
        return [common+['--prec-div=true','--prec-sqrt=true','-dc',base+'.cu','-o',base+'_main.o'],
            common+['-dc','-dlto','-I'+str(self.sdk_header.parent),base+'_bridge.cu','-o',base+'_bridge.o'],
            common+['-shared','-dlto',base+'_main.o',base+'_bridge.o',str(self.archive),'-o',base+'.so']]

    def launch_command(self, key):
        return [str(self.compiler),'-shared','--fmad=true','--prec-div=true','--prec-sqrt=true','-arch=sm_80',
            str(self.directory/(key+'.cu')),'-o',str(self.directory/(key+'.so'))]

    def dependency_inputs(self, key):
        return dict(schema='libero-newton-compiler-inputs-v1',discovery='compiler-resolved-per-unit',commands=self.commands(key),
            units=[self.file_record(self.directory/(key+'.cu')),self.file_record(self.directory/(key+'_bridge.cu'))],
            dependencies=[self.file_record(self.sdk_header)],
            toolchain=dict(environment_sha256='compiler-environment',components=[self.file_record(self.compiler)]),
            archive=self.file_record(self.archive),architecture=80)

    def save_record(self):
        (self.root/'AOT_BUILD.json').write_bytes(self.canonical(self.record))

    def request(self, **changes):
        kwargs=dict(target=self.target,identity=self.key,source=self.source,bridge_source=self.bridge,
            stage_sources=self.closure,mathdx_archive=self.archive,architecture=80,
            commands=self.commands(self.key),launch_command=self.launch_command(self.key),
            nv=21,nM=86,dependency_inputs=self.inputs)
        kwargs.update(changes)
        return self.aot['migrate_newton_path_alias'](self.root,**kwargs)

    def test_exact_alias_is_idempotent_and_preserves_history_and_global_identity(self):
        old_aot=(self.root/'AOT_BUILD.json').read_bytes()
        old_meta=self.previous.with_suffix('.json').read_bytes()
        with patch('subprocess.run',side_effect=AssertionError('no compiler allowed')), \
                patch('subprocess.check_output',side_effect=AssertionError('no dependency scan allowed')):
            first=self.request()
            second=self.request()
        self.assertTrue(first['published'])
        self.assertFalse(second['published'])
        self.assertEqual(self.target.read_bytes(),self.binary)
        self.assertEqual(self.previous.read_bytes(),self.binary)
        self.assertEqual(self.previous.with_suffix('.json').read_bytes(),old_meta)
        self.assertEqual((self.root/'AOT_BUILD.json').read_bytes(),old_aot)
        self.assertEqual(self.target.with_suffix('.cu').read_bytes(),self.source.encode())
        receipt=json.loads(self.target.with_suffix('.path-alias.json').read_text())
        self.assertEqual(receipt['origin_source_identity'],'historical-global-identity')
        self.assertEqual(receipt['task_keys'],['suite/a.bddl','suite/b.bddl'])
        self.assertEqual(json.loads(self.target.with_suffix('.dependencies.json').read_text())['recipe'],self.key)

    def test_dry_run_writes_nothing(self):
        before={str(p.relative_to(self.base)):p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        self.assertFalse(self.request(dry_run=True)['published'])
        after={str(p.relative_to(self.base)):p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_crlf_source_and_bridge_are_recipe_bytes_not_normalized_text(self):
        receipt=self.request()
        self.assertEqual(self.target.with_suffix('.cu').read_bytes(),self.source.encode())
        self.assertEqual(self.target.with_name(self.key+'_bridge.cu').read_bytes(),self.bridge.encode())
        self.assertEqual(receipt['source_sha256'],self.sha(self.source.encode()))
        self.assertEqual(receipt['bridge_sha256'],self.sha(self.bridge.encode()))
        normalized=self.aot['_newton_recipe'](self.source.replace('\r\n','\n'),
            self.bridge.replace('\r\n','\n'),self.closure,self.sha(self.archive.read_bytes()))
        self.assertNotEqual(normalized,self.key)

    def test_missing_or_unanchored_historical_dependency_receipt_is_a_miss(self):
        self.assertIsNone(self.request(dependency_inputs=None))
        self.record['artifacts'].pop(str(self.dependency_path.relative_to(self.root)))
        self.save_record()
        self.assertIsNone(self.request())
        self.assertFalse(self.target.exists())

    def test_sdk_header_change_with_same_archive_and_paths_is_rejected(self):
        self.sdk_header.write_bytes(b'SDK header B\n')
        self.assertIsNone(self.request(dependency_inputs=self.dependency_inputs(self.key)))
        with self.assertRaisesRegex(ValueError,'input changed'):
            self.request()
        self.assertFalse(self.target.exists())

    def test_source_bridge_header_archive_and_flags_changes_never_alias(self):
        changed=self.source.replace('current kernel','different kernel')
        key=self.aot['_newton_recipe'](changed,self.bridge,self.closure,self.sha(self.archive.read_bytes()))
        self.assertIsNone(self.request(source=changed,identity=key,target=self.directory/(key+'.so')))
        changed_bridge=self.bridge+'different bridge\n'
        key=self.aot['_newton_recipe'](self.source,changed_bridge,self.closure,self.sha(self.archive.read_bytes()))
        self.assertIsNone(self.request(bridge_source=changed_bridge,identity=key,target=self.directory/(key+'.so')))
        flags=self.commands(self.key);flags[0].append('--ftz=true')
        self.assertIsNone(self.request(commands=flags))
        self.archive.write_bytes(b'different archive\n')
        key=self.aot['_newton_recipe'](self.source,self.bridge,self.closure,self.sha(self.archive.read_bytes()))
        self.assertIsNone(self.request(identity=key,target=self.directory/(key+'.so')))
        self.assertFalse(self.target.exists())

    def test_current_header_digest_and_recipe_mismatch_raise(self):
        self.header.write_bytes(b'changed project header\n')
        with self.assertRaisesRegex(ValueError,'current header differs'):
            self.request()
        self.header.write_bytes(b'project header\n')
        wrong='f'*64
        with self.assertRaisesRegex(ValueError,'current complete recipe differs'):
            self.request(identity=wrong,target=self.directory/(wrong+'.so'))

    def test_binary_tamper_and_existing_product_conflict_raise_without_overwrite(self):
        self.previous.write_bytes(b'tampered binary')
        with self.assertRaisesRegex(ValueError,'original binary differs'):
            self.request()
        self.assertFalse(self.target.exists())
        self.previous.write_bytes(self.binary)
        self.target.write_bytes(b'foreign existing product')
        with self.assertRaisesRegex(ValueError,'current binary differs'):
            self.request()
        self.assertEqual(self.target.read_bytes(),b'foreign existing product')

    def test_family_architecture_launch_argv_and_whitelist_are_explicit(self):
        with self.assertRaisesRegex(ValueError,'exact current CUDA product path'):
            self.request(nv=22)
        with self.assertRaisesRegex(ValueError,'exact current CUDA product path'):
            self.request(architecture=90)
        launch=self.launch_command(self.key)+['--ftz=true']
        self.assertIsNone(self.request(launch_command=launch))
        self.aot['_NEWTON_LEGACY_SOURCE_ROOTS']=()
        self.assertIsNone(self.request())

    def test_changed_global_source_identity_is_not_certified_by_migration(self):
        self.request()
        self.aot['source_identity']=lambda:'different-current-source'
        error=self.modules['libero.gpu.host.build.compilation'].MissingCompiledProduct
        with self.assertRaisesRegex(error,'incomplete or stale'):
            self.aot['require_ready'](self.root,'suite/a.bddl',num_envs=8,seed=0)
        self.assertEqual(json.loads((self.root/'AOT_BUILD.json').read_text())['source_identity'],
            'historical-global-identity')

    def test_dependency_collector_records_actual_units_headers_and_toolchain(self):
        build=ModuleType('libero.gpu.host.build.build_receipts')
        calls=[]
        build.file_record=lambda p:self.file_record(Path(p))
        build.compiler_dependencies=lambda compiler,flags,units:(
            calls.append((compiler,flags,units)) or [self.file_record(self.sdk_header)])
        build.toolchain_record=lambda compiler,flags:dict(components=[self.file_record(self.compiler)])
        with patch.dict(sys.modules,{'libero.gpu.host.build.build_receipts':build}), \
                patch('subprocess.run',side_effect=AssertionError('no real compiler')), \
                patch('subprocess.check_output',side_effect=AssertionError('no real dependency scan')):
            inputs=self.aot['collect_newton_dependency_inputs'](self.commands(self.key),
                architecture=80,mathdx_archive=self.archive)
        self.assertEqual(len(calls),2)
        self.assertEqual(inputs['dependencies'],[self.file_record(self.sdk_header)])
        self.assertEqual(inputs['toolchain']['environment_sha256'],'compiler-environment')
        self.assertEqual(len(inputs['units']),2)
        self.assertTrue(all('-o' not in flags and '-dc' not in flags and '-dlto' not in flags
            for _,flags,_ in calls))
        self.target.write_bytes(self.binary)
        receipt=self.aot['publish_newton_dependency_receipt'](self.target,identity=self.key,inputs=inputs)
        result=json.loads(receipt.read_text())
        self.assertEqual(result['binary_sha256'],self.sha(self.binary))
        self.assertEqual(result['inputs'],inputs)

    def test_collect_publish_migrate_roundtrip_uses_collector_schema(self):
        build=ModuleType('libero.gpu.host.build.build_receipts')
        build.file_record=lambda p:self.file_record(Path(p))
        build.compiler_dependencies=lambda compiler,flags,units:[self.file_record(self.sdk_header)]
        build.toolchain_record=lambda compiler,flags:dict(components=[self.file_record(self.compiler)])
        self.dependency_path.unlink()
        with patch.dict(sys.modules,{'libero.gpu.host.build.build_receipts':build}):
            old_inputs=self.aot['collect_newton_dependency_inputs'](self.commands(self.old_key),
                architecture=80,mathdx_archive=self.archive)
            self.aot['publish_newton_dependency_receipt'](self.previous,identity=self.old_key,inputs=old_inputs)
            current_inputs=self.aot['collect_newton_dependency_inputs'](self.commands(self.key),
                architecture=80,mathdx_archive=self.archive)
        self.record['artifacts'][str(self.dependency_path.relative_to(self.root))]=self.sha(self.dependency_path.read_bytes())
        self.save_record()
        self.assertEqual(old_inputs['discovery'],'compiler-resolved-per-unit')
        self.assertTrue(self.request(dependency_inputs=current_inputs)['published'])
        result=json.loads(self.target.with_suffix('.dependencies.json').read_text())
        self.assertEqual(result['inputs'],current_inputs)

    def test_building_manifest_uses_sha_anchored_previous_complete_record(self):
        original=(self.root/'AOT_BUILD.json').read_bytes()
        anchor=self.aot['preserve_complete_build'](self.root)
        self.assertEqual((self.root/anchor['path']).read_bytes(),original)
        self.record=dict(status='building',tasks={},previous_complete_build=anchor,
            source_identity='new-build-is-not-complete')
        self.save_record()
        self.assertEqual(self.aot['preserve_complete_build'](self.root),anchor)
        result=self.request()
        self.assertTrue(result['published'])
        self.assertEqual(result['origin_source_identity'],'historical-global-identity')
        self.assertEqual(json.loads((self.root/'AOT_BUILD.json').read_text())['status'],'building')

    def test_previous_manifest_anchor_tamper_refuses_migration(self):
        anchor=self.aot['preserve_complete_build'](self.root)
        self.record=dict(status='building',tasks={},previous_complete_build=anchor)
        self.save_record()
        (self.root/anchor['path']).write_bytes(b'{"status":"complete","tasks":{}}')
        with self.assertRaisesRegex(ValueError,'manifest SHA differs'):
            self.request()
        self.assertFalse(self.target.exists())

    def test_dependency_collection_obeys_frozen_scope_and_changed_inputs_refuse_receipt(self):
        compilation=self.modules['libero.gpu.host.build.compilation']
        def frozen(*args):
            raise compilation.MissingCompiledProduct('runtime compilation is disabled')
        compilation.require_compilation=frozen
        with self.assertRaisesRegex(compilation.MissingCompiledProduct,'runtime compilation'):
            self.aot['collect_newton_dependency_inputs'](self.commands(self.key),
                architecture=80,mathdx_archive=self.archive)
        self.target.write_bytes(self.binary)
        self.sdk_header.write_bytes(b'changed during compilation\n')
        with self.assertRaisesRegex(ValueError,'input changed'):
            self.aot['publish_newton_dependency_receipt'](self.target,identity=self.key,inputs=self.inputs)
        self.assertFalse(self.target.with_suffix('.dependencies.json').exists())


class NewtonCompletionAdmissionTests(unittest.TestCase):
    """Completion/historical receipt checks with temporary inert binaries."""

    def setUp(self):
        from libero.gpu.host.build import aot
        self.aot=aot
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.target=self.root/'cuda'/'recipe.so'
        self.target.parent.mkdir()
        self.target.write_bytes(b'completed binary')
        self.metadata=self.target.with_suffix('.json')
        self.receipt_path=self.target.with_suffix('.dependencies.json')
        self.previous=dict(sha256='recipe',source_sha256='source',library_sha256=aot._sha(self.target),
            command=['nvcc','-arch=sm_80'],compile_commands=[['nvcc','-arch=sm_80','source.cu']])
        self.inputs=dict(schema='libero-newton-compiler-inputs-v1',discovery='compiler-resolved-per-unit',
            architecture=80,commands=self.previous['compile_commands'],dependencies=[],units=[],
            toolchain=dict(components=[]),archive={})

    def admit(self):
        return self.aot.admit_newton_cached_product(self.target,identity='recipe',source_sha256='source',
            architecture=80,preserve_root=self.root)

    def publish_metadata(self,*,receipt=False):
        if receipt:
            self.previous['dependency_receipt_file']=str(self.receipt_path)
            encoded=json.dumps(self.inputs,sort_keys=True,separators=(',',':')).encode()
            self.receipt=dict(schema='libero-newton-dependencies-v1',recipe='recipe',
                binary_sha256=self.previous['library_sha256'],inputs=self.inputs,
                inputs_sha256=hashlib.sha256(encoded).hexdigest())
            self.receipt_path.write_text(json.dumps(self.receipt))
        self.metadata.write_text(json.dumps(self.previous))

    def test_load_only_rejects_orphan_without_writing_or_discarding_it(self):
        from libero.gpu.host.build.compilation import compilation_scope,MissingCompiledProduct
        with compilation_scope(False),self.assertRaisesRegex(MissingCompiledProduct,'completion metadata'):
            self.admit()
        self.assertEqual(self.target.read_bytes(),b'completed binary')
        self.assertFalse((self.root/'incomplete-newton-products').exists())

    def test_offline_preserves_orphan_binary_and_receipt_then_allows_only_fresh_completion(self):
        self.receipt_path.write_bytes(b'not completion evidence without metadata')
        self.assertIsNone(self.admit())
        self.assertFalse(self.target.exists())
        self.assertFalse(self.receipt_path.exists())
        isolated=list((self.root/'incomplete-newton-products').iterdir())
        self.assertEqual(len(isolated),1)
        self.assertEqual((isolated[0]/self.target.name).read_bytes(),b'completed binary')
        record=json.loads((isolated[0]/'INCOMPLETE.json').read_text())
        self.assertFalse(record['completed'])
        self.assertEqual(len(record['files']),2)
        self.target.write_bytes(b'completed binary')
        self.publish_metadata()
        self.assertEqual(self.admit(),self.previous)
        self.assertEqual(len(list((self.root/'incomplete-newton-products').iterdir())),1)

    def test_matching_legacy_metadata_is_accepted_but_mismatches_are_rejected(self):
        self.publish_metadata()
        self.assertEqual(self.admit(),self.previous)
        self.previous['source_sha256']='other source'
        self.publish_metadata()
        with self.assertRaisesRegex(ValueError,'source/binary/architecture differs'):
            self.admit()
        self.assertEqual(self.target.read_bytes(),b'completed binary')

    def test_receipt_must_bind_schema_recipe_binary_and_complete_inputs(self):
        self.publish_metadata(receipt=True)
        original=copy.deepcopy(self.receipt)
        changes=[dict(schema='wrong'),dict(recipe='wrong'),dict(binary_sha256='wrong'),
            dict(inputs_sha256='wrong'),dict(inputs={})]
        for changed in changes:
            row=dict(original,**changed)
            self.receipt_path.write_text(json.dumps(row))
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'does not bind'):
                self.admit()

    def test_historical_receipt_is_not_reissued_or_rejected_after_a_system_upgrade(self):
        library=self.root/'compiler-loader.so'
        library.write_bytes(b'old library')
        self.inputs['toolchain']['components']=[dict(path=str(library),bytes=11,
            sha256=hashlib.sha256(library.read_bytes()).hexdigest())]
        self.publish_metadata(receipt=True)
        original=self.receipt_path.read_bytes()
        library.write_bytes(b'new library')
        with patch.object(self.aot,'_check_newton_dependency_inputs',side_effect=AssertionError('historical inputs are not current pins')):
            self.assertEqual(self.admit(),self.previous)
        self.assertEqual(self.receipt_path.read_bytes(),original)

    def test_declared_missing_dependency_receipt_is_rejected(self):
        self.publish_metadata(receipt=True)
        self.receipt_path.unlink()
        with self.assertRaisesRegex(ValueError,'receipt is absent'):
            self.admit()

    def test_actual_fresh_receipt_failure_removes_only_the_new_binary(self):
        source=GPU/'host/runtime/newton/handwritten_cuda.py'
        tree=ast.parse(source.read_text())
        node=next(node for node in ast.walk(tree) if isinstance(node,ast.Try) and
            any(isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and
                child.func.id=='publish_newton_dependency_receipt' for child in ast.walk(node)))
        code=compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec')
        context=dict(target=self.target,reused_product=False,dependency_inputs={},digest='recipe',
            verify_completed_product=lambda:None)
        with patch.object(self.aot,'publish_newton_dependency_receipt',side_effect=ValueError('changed input')):
            with self.assertRaisesRegex(ValueError,'changed input'):
                exec(code,context)
        self.assertFalse(self.target.exists())
        self.target.write_bytes(b'completed binary')
        context.update(reused_product=True,verify_completed_product=lambda:(_ for _ in ()).throw(ValueError('bad ABI')))
        with self.assertRaisesRegex(ValueError,'bad ABI'):
            exec(code,context)
        self.assertEqual(self.target.read_bytes(),b'completed binary')


class GpuBuildCacheTests(unittest.TestCase):
    def test_formal_rollout_runs_once_and_excludes_progress_io_from_timing(self):
        main = runpy.run_path(str(GPU.parents[2]/'benchmark_scripts/benchmark_gpu_batch.py'))['main']
        clock, owners = [0.0], []
        class Environment:
            action_dim = 7
            substeps = 25
            def __init__(self, *args, **kwargs):
                self.build_report = dict(directory='prebuilt', events=[])
                self.integrations = self.steps = self.resets = 0
                self.options = kwargs
                self.graph_preparations = 0
                def prepare_graph():
                    self.graph_preparations += 1
                    clock[0] += 7
                self._physics = SimpleNamespace(prepare_control_graph=prepare_graph)
                clock[0] += 4
                owners.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def reset(self, **kwargs):
                self.resets += 1
                self.integrations = 0
                clock[0] += 2
            def step(self, action):
                self.steps += 1
                self.integrations += 25
                clock[0] += .5 if self.steps % 2 else 1.5
            def diagnostics(self):
                return dict(actual_integration_count=[self.integrations]*8, first_error=[0]*8)
        suite = SimpleNamespace(n_tasks=1, get_task=lambda i: SimpleNamespace(name='task'),
            get_task_bddl_file_path=lambda i: 'task.bddl', get_task_init_states=lambda i: [])
        package = ModuleType('libero.libero')
        package.benchmark = SimpleNamespace(get_benchmark=lambda name: lambda: suite)
        envs = ModuleType('libero.libero.envs')
        envs.GpuBatchEnv = Environment
        write = Path.write_text
        def slow_write(path, *args, **kwargs):
            clock[0] += 3  # Deliberately much slower than a simulated control step.
            return write(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output, progress = Path(directory)/'result.json', Path(directory)/'progress.json'
            argv = ['benchmark', '--steps', '600', '--warmup', '20', '--repeats', '1',
                    '--output', str(output), '--progress-output', str(progress), '--progress-every', '1']
            with patch.dict(sys.modules, {'libero.libero': package, 'libero.libero.envs': envs}), \
                    patch.object(sys, 'argv', argv), patch.object(Path, 'write_text', slow_write), \
                    patch('time.perf_counter', side_effect=lambda: clock[0]), redirect_stdout(io.StringIO()):
                main()
            row = json.loads(output.read_text())['tasks'][0]
            self.assertEqual(json.loads(progress.read_text())['completed_steps'], 600)
        self.assertEqual((owners[0].steps, owners[0].resets), (620, 1))
        self.assertFalse(owners[0].options['initialize'])
        self.assertFalse(owners[0].options['allow_compile'])
        self.assertEqual(owners[0].graph_preparations, 1)
        self.assertEqual(row['constructor_seconds'], 4)
        self.assertEqual(row['graph_capture_seconds'], 7)
        self.assertEqual(row['setup_seconds'], 11)
        self.assertEqual(len(row['samples']), 1)
        self.assertEqual(row['samples'][0]['seconds'], 600)
        self.assertEqual(row['samples'][0]['env_steps_per_second'], 8)
        self.assertEqual(row['samples'][0]['reset_seconds'], 2)
        self.assertEqual(row['samples'][0]['integration_delta'], [15000]*8)
        self.assertEqual(len(row['samples'][0]['step_seconds']), 600)
        self.assertEqual(row['samples'][0]['step_ms_min'], 500)
        self.assertEqual(row['samples'][0]['step_ms_max'], 1500)
        self.assertEqual(row['samples'][0]['step_ms_std'], 500)
        self.assertEqual(row['samples'][0]['step_time_cv'], .5)

    def test_newton_reuse_scans_native_products_and_verifies_binary(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': directory}):
            root = Path(directory)/'project'
            candidate = root/'cuda/old/product.so'
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b'verified native binary')
            metadata = dict(sha256='identity', source_sha256='source',
                library_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                command=['nvcc', '-arch=sm_80'])
            candidate.with_suffix('.json').write_text(json.dumps(metadata))
            target = root/'cuda/new/product.so'
            target.parent.mkdir(parents=True)
            original_walk = os.walk
            with paths.build_project(root), patch.object(os, 'walk', wraps=original_walk) as walk:
                self.assertEqual(paths.reuse_newton_binary(target, 'identity', 'source', 80), str(candidate))
                self.assertEqual(target.read_bytes(), candidate.read_bytes())
                self.assertEqual({Path(call.args[0]) for call in walk.call_args_list},
                                 {Path(directory)/'builds', root/'cuda'})
                target.unlink()
                candidate.write_bytes(b'tampered')
                self.assertIsNone(paths.reuse_newton_binary(target, 'identity', 'source', 80))
                self.assertFalse(target.exists())

    def test_stable_warp_kernels_keep_distinct_launcher_owner_bindings(self):
        source = GPU / 'host/assembly/pipeline/newton.py'
        tree = ast.parse(source.read_text())
        compile_schedule = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == '_compile_schedule')
        loader_source = GPU / 'host/assembly/compose/compose.py'
        load = next(n for n in ast.parse(loader_source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_load')
        launches, kernels, registered = [], {}, []
        wp = SimpleNamespace(set_module_options=lambda *a, **kw: registered.append(kw),
            launch=lambda kernel, nworld, inputs, **kw: launches.append(inputs))
        loader = dict(hashlib=hashlib, sys=sys, GPU_ROOT=GPU, linecache=linecache,
                      types=types, wp=wp, __name__='_cache_owner_test')
        exec(compile(ast.Module(body=[load], type_ignores=[]), str(loader_source), 'exec'), loader)
        def stage(value: int):
            pass
        source_kernel = SimpleNamespace(func=stage, key='stage')
        def compose(name, stages):
            if name not in kernels:
                kernels[name] = SimpleNamespace(fusion_argument_names=tuple(stages[0][1].values()))
            return kernels[name]
        namespace = dict(ast=ast, copy=copy, hashlib=hashlib, inspect=inspect,
                         textwrap=textwrap, _INLINE=set(), compose=compose, _load=loader['_load'])
        exec(compile(ast.Module(body=[compile_schedule], type_ignores=[]), str(source), 'exec'), namespace)
        class Solver:
            def launch_device(self):
                ws = self.ws
                wp.launch(source_kernel, ws.nworld, inputs=[owner.value], device=ws.device, stream=ws.stream)
        owners = [SimpleNamespace(value=11), SimpleNamespace(value=22)]
        compiled = [namespace['_compile_schedule'](Solver, 'launch_device',
            extra_namespace=dict(owner=owner, wp=wp, source_kernel=source_kernel),
            identity_suffix=':event-owner:' + str(id(owner)),
            kernel_identity_suffix=':event-publication') for owner in owners]
        self.assertIsNot(compiled[0][0], compiled[1][0])
        self.assertIs(compiled[0][2][0], compiled[1][2][0])
        caller = SimpleNamespace(ws=SimpleNamespace(nworld=1, device='cuda:0', stream=object()))
        for item in compiled:
            item[0](caller)
        self.assertEqual(launches, [[11], [22]])
        self.assertEqual(registered, [])

    def test_build_only_cli_never_resets_or_steps_and_closes_owner(self):
        script = GPU.parents[2] / 'benchmark_scripts/benchmark_gpu_batch.py'
        main = runpy.run_path(str(script))['main']
        constructed, closed = [], []
        class Environment:
            def __init__(self, bddl, num_envs, **kwargs):
                constructed.append(kwargs)
                self.build_report = dict(directory='persistent/project', events=[
                    dict(kind='native-cuda', commands_executed=[], cache_hit=True)])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                closed.append(True)
            def diagnostics(self):
                return dict(actual_integration_count=[0] * 8)
            def reset(self, *args, **kwargs):
                raise AssertionError('build-only must not reset')
            def step(self, *args, **kwargs):
                raise AssertionError('build-only must not step')
        suite = SimpleNamespace(n_tasks=1,
            get_task=lambda i: SimpleNamespace(name='task'),
            get_task_bddl_file_path=lambda i: 'task.bddl',
            get_task_init_states=lambda i: [])
        package = ModuleType('libero.libero')
        package.benchmark = SimpleNamespace(get_benchmark=lambda name: lambda: suite)
        envs = ModuleType('libero.libero.envs')
        envs.GpuBatchEnv = Environment
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'build.json'
            with patch.dict(sys.modules, {'libero.libero': package, 'libero.libero.envs': envs}), \
                    patch.object(sys, 'argv', ['benchmark', '--build-only', '--output', str(output)]):
                main()
            row = json.loads(output.read_text())['tasks'][0]
        self.assertFalse(constructed[0]['initialize'])
        self.assertEqual(closed, [True])
        self.assertEqual(row['status'], 'built')
        self.assertEqual(row['compiler_invocations'], 0)
        self.assertNotIn('samples', row)

    def test_project_scope_is_persistent_and_records_only_current_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with paths.build_project(root) as first:
                paths.record_build('native-cuda', 'key', (('nvcc', 'source.cu'),))
                with paths.build_project(root / 'unused') as nested:
                    self.assertIs(first, nested)
            self.assertIsNone(paths.current_build_project())
            with paths.build_project(root) as second:
                paths.record_build('native-cuda', 'key')
            self.assertEqual(first['root'], second['root'])
            self.assertFalse(first['events'][0]['cache_hit'])
            self.assertTrue(second['events'][0]['cache_hit'])
            self.assertEqual(len(first['events']), 1)
            self.assertFalse((root / 'unused').exists())

    def test_module_addresses_get_fresh_records_with_shared_compiled_library(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': directory}):
            target = Path(directory) / 'project/collision'
            target.mkdir(parents=True)
            (target / 'libcollision.so').write_bytes(b'shared compiled image')
            files = []
            lifetimes = []
            for address in (101, 202):
                with paths.build_project(Path(directory) / 'project'), paths.product_build():
                    lifetimes.append(paths.current_build_lifetime().retain())
                    files.append(Path(paths.record_module_binding(target, {'cuda_function': address})))
            self.assertNotEqual(files[0], files[1])
            self.assertEqual([json.loads(p.read_text())['cuda_function'] for p in files], [101, 202])
            self.assertTrue(all(p.is_relative_to(Path(directory) / 'storage-v1/runs') for p in files))
            self.assertEqual((target / 'libcollision.so').read_bytes(), b'shared compiled image')
            for lifetime in lifetimes:
                lifetime.close()
            self.assertTrue(all(not p.exists() for p in files))

    def test_distinct_generated_source_paths_do_not_collide_in_build_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project/collision'
            gpu = Path(directory) / 'gpu'
            with paths.build_project(root.parent):
                fixed = paths.collision_build_root(root, gpu / 'cuda/collision/native', gpu)
                first = paths.collision_build_root(root, root / 'source-one', gpu)
                second = paths.collision_build_root(root, root / 'source-two', gpu)
                again = paths.collision_build_root(root, root / 'source-one', gpu)
            self.assertEqual(fixed, root)
            self.assertNotEqual(first, second)
            self.assertEqual(first, again)

    def test_loaded_collision_library_cache_is_specific_to_cuda_device(self):
        source = GPU / 'host/runtime/collision/native.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_prepare_native')
        class Device:
            is_cuda, arch = True, 80
            def __init__(self, alias):
                self.alias = alias
            def __str__(self):
                return self.alias
        class ExistingProducts(dict):
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                return self.setdefault(key, object())
        namespace = dict(wp=SimpleNamespace(get_device=Device), Path=Path, os=os,
                         GPU_ROOT=GPU, MODULE=GPU / 'cuda/collision/native',
                         _native_source_bindings=lambda path: {}, hashlib=hashlib,
                         json=json, _CACHE=ExistingProducts(), current_product_authority=lambda: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {PATHS_NAME: paths}), \
                patch.dict(os.environ, {'P06C_NATIVE_BUILD_ROOT': directory}):
            first = namespace['_prepare_native']('cuda:0')
            again = namespace['_prepare_native']('cuda:0')
            other = namespace['_prepare_native']('cuda:1')
        self.assertIs(first, again)
        self.assertIsNot(first, other)

    def test_source_namespace_reuse_change_and_tampering(self):
        def generator(manifest, static, root):
            root.mkdir()
            (root / '__init__.py').write_text('')
            source = 'NV = ' + str(manifest['nv']) + '\n'
            (root / 'identity.py').write_text(source)
            (root / 'MODEL_MANIFEST.json').write_text(json.dumps(manifest))
            return dict(template_pins={'identity': {'sha256': 'template'}},
                        generated_pins={'identity': {
                            'path': str(root / 'identity.py'),
                            'sha256': hashlib.sha256(source.encode()).hexdigest()}})
        with tempfile.TemporaryDirectory() as directory:
            first, report, hit = generated_package({'nv': 21}, {}, generator, directory)
            self.assertFalse(hit)
            self.assertEqual(report['generated_pins']['identity']['path'], str(first / 'identity.py'))
            # Python imports may add bytecode; it must not alter the source key.
            (first / '__pycache__').mkdir()
            second, _, hit = generated_package({'nv': 21}, {}, generator, directory)
            self.assertTrue(hit)
            self.assertEqual(first, second)
            third, _, hit = generated_package({'nv': 43}, {}, generator, directory)
            self.assertFalse(hit)
            self.assertNotEqual(first, third)
            (first / 'identity.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'cached specialization source differs'):
                generated_package({'nv': 21}, {}, generator, directory)

    def test_native_products_compile_once_and_invalidate_changed_inputs(self):
        source_path = GPU / 'host/build/native.py'
        tree = ast.parse(source_path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_compile_library_legacy')
        calls = []
        def compiler(command, **kwargs):
            calls.append(command)
            Path(command[command.index('-o') + 1]).write_bytes(b'compiled test image')
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        namespace = dict(Path=Path, wp=SimpleNamespace(get_device=lambda device:
                         SimpleNamespace(is_cuda=True, arch=80)),
                         hashlib=hashlib, json=json, re=re, os=os, shutil=shutil,
                         subprocess=SimpleNamespace(run=compiler),
                         nvcc_path=lambda: '/toolkit/bin/nvcc',
                         load_cdll=lambda path: str(path),
                         ct=SimpleNamespace(CDLL=lambda path: path))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
        compile_ = namespace['_compile_library_legacy']
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {PATHS_NAME: paths}), \
                patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': directory,
                                       'LIBERO_CUDA_FUSION_CACHE': ''}):
            project_root = Path(directory) / 'project'
            build = project_root / 'cuda/sm80'
            header = Path(directory) / 'input.cuh'
            header.write_text('const int version = 1;\n')
            source = '#include "' + str(header) + '"\n'
            with paths.build_project(project_root):
                _, first = compile_('first', source, build)
            self.assertEqual(len(calls), 1)
            with paths.build_project(project_root):
                _, second = compile_('first', source, build)
                _, alias = compile_('other_plan', source, build)
            self.assertEqual(len(calls), 1)
            self.assertEqual(second['compile_commands_executed'], [])
            self.assertEqual(first['sha256'], alias['sha256'])
            self.assertEqual(alias['compile_commands_executed'], [])
            with paths.build_project(project_root):
                _, changed_flags = compile_('first', source, build, defines={'NV': 43})
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(first['sha256'], changed_flags['sha256'])
            header.write_text('const int version = 2;\n')
            with paths.build_project(project_root):
                _, changed_header = compile_('first', source, build)
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(first['sha256'], changed_header['sha256'])
            Path(changed_header['library']).write_bytes(b'tampered')
            with paths.build_project(project_root), \
                    self.assertRaisesRegex(ValueError, 'cached native binary differs'):
                compile_('first', source, build)
            self.assertEqual(len(calls), 3)

    def test_verified_old_newton_product_can_migrate_without_compilation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'LIBERO_GPU_CACHE_DIR': directory}):
            root = Path(directory)
            old = root / 'builds/environment-old/capacity'
            old.mkdir(parents=True)
            candidate = old / 'digest.so'
            candidate.write_bytes(b'completed Newton image')
            manifest = dict(sha256='digest', source_sha256='source', command=['nvcc', '-arch=sm_80'],
                            library_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
            (old / 'MANIFEST.json').write_text(json.dumps(manifest))
            target = root / 'project/cuda/sm80/digest.so'
            target.parent.mkdir(parents=True)
            with paths.build_project(root / 'project'):
                self.assertIsNone(paths.reuse_newton_binary(target, 'digest', 'source', 90))
                self.assertIsNone(paths.reuse_newton_binary(target, 'digest', 'changed', 80))
                candidate.write_bytes(b'corrupted')
                self.assertIsNone(paths.reuse_newton_binary(target, 'digest', 'source', 80))
                candidate.write_bytes(b'completed Newton image')
                self.assertEqual(paths.reuse_newton_binary(target, 'digest', 'source', 80), str(candidate))
            self.assertEqual(candidate.read_bytes(), target.read_bytes())
            self.assertEqual(json.loads(target.with_suffix('.json').read_text()), manifest)


if __name__ == '__main__':
    unittest.main()
