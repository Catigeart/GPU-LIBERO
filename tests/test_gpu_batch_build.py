"""Declared CUDA model coverage and ordinary CMake-only build orchestration."""
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from libero.gpu.host.build import batch
from libero.gpu.host.build.compilation import (
    compilation_allowed, compilation_scope, environment_scope, MissingCompiledProduct,
    require_compilation,
)

GPU = Path(__file__).resolve().parents[1]/'libero/libero/gpu'


class BatchBuildTests(unittest.TestCase):
    def test_cpp_instances_cover_every_task_and_no_world_count_is_instantiated(self):
        rows=json.loads((GPU/'data/libero_task_parameters.json').read_text())['tasks']
        expected={tuple(t['dimensions'][k] for k in ('nv','nq','nM','njnt','nu')) for t in rows}
        declared=[tuple(map(int, values)) for values in re.findall(
            r'^LIBERO_MODEL\((\d+), *(\d+), *(\d+), *(\d+), *(\d+)\)$',
            (GPU/'cuda/step/instantiations/libero_models.def').read_text(), re.M)]
        self.assertEqual(len(rows),130)
        self.assertEqual(len(declared),len(set(declared)))
        self.assertEqual(set(declared),expected)

    def test_build_does_not_import_physics_or_construct_environments(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'warp':None,'mujoco':None,'robosuite':None}), \
                patch.object(batch,'nvcc_path',return_value='/offline/nvcc'), \
                patch.object(batch.subprocess,'run') as run:
            result=batch.compile_project(directory,precision='fp64',modes=('dev','release'),
                                         mathdx_root=Path(directory)/'sdk',jobs=3)
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands),4)
        for command in commands[1::2]:
            self.assertEqual(command[command.index('--target')+1],'flash_native_batch_all')
            self.assertEqual(command[command.index('--parallel')+1],'3')
        self.assertEqual(result['environment_constructions'],0)
        self.assertEqual(result['physics_integrations'],0)

    def test_fp32_and_ordered_select_their_complete_cmake_targets(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(batch,'nvcc_path',return_value='/offline/nvcc'), \
                patch.object(batch.subprocess,'run') as run:
            for precision,ordered,target in (('fp32',False,'flash_cuda_fp32_all'),
                    ('fp64',True,'flash_native_batch_ordered_all')):
                result=batch.compile_project(directory,precision=precision,
                    mathdx_root=directory,ordered_reductions=ordered)
                self.assertEqual(result['builds'][0]['target'],target)
            self.assertEqual(run.call_count,4)
            with self.assertRaises(ValueError):
                batch.compile_project(directory,precision='fp32',ordered_reductions=True)

    def test_offline_failure_is_not_reported_as_a_completed_build(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(batch,'nvcc_path',return_value='/offline/nvcc'), \
                patch.object(batch.subprocess,'run',side_effect=RuntimeError('compiler failed')):
            with self.assertRaisesRegex(RuntimeError,'compiler failed'):
                batch.compile_project(directory,mathdx_root=directory)


class RuntimePermissionTests(unittest.TestCase):
    def test_offline_constructor_permission_does_not_leak_into_public_calls(self):
        class Env:
            allow_compile = True
            build_report = dict(events=[])
            @environment_scope
            def step(self):
                require_compilation('unexpected runtime compiler','missing')
        env=Env()
        with self.assertRaises(MissingCompiledProduct): env.step()
        with compilation_scope(True): env.step()
        with compilation_scope(False),self.assertRaises(MissingCompiledProduct): env.step()
        self.assertTrue(compilation_allowed())


if __name__=='__main__':unittest.main()
