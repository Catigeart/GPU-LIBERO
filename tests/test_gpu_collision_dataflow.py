"""Cold collision stage selection, without loading a GPU runtime."""
import ast
from collections import UserDict
from pathlib import Path
from types import MethodType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/bindings/collision.py'


class CollisionDataflowTests(unittest.TestCase):
    def stages(self, *, profile='fast', diagnostics=False, invalid_restore=False):
        array = lambda ptr: SimpleNamespace(ptr=ptr)
        active, shadow, source, destination = [array(i) for i in range(4)]
        tree = SimpleNamespace(stage=array(4), stage_count=array(5), active_in=array(6))
        ws = SimpleNamespace(_cuda_collision_dataflow_profile=profile,
            _cuda_collision_diagnostics=diagnostics, before_outputs={'output': shadow},
            tree=tree, active=active, fe_code=array(7), contact_capacity=10,
            stage=tree.stage, stage_count=tree.stage_count, effective_active=active,
            finalize=SimpleNamespace(ncon=array(8), contact_f=array(9), contact_i=array(10)))
        dedup = [array(i+20) for i in range(13)]
        dedup[7], dedup[12] = 2, 3
        params = [array(i+40) for i in range(29)]
        params[0], params[1], params[3] = dedup[5], dedup[8], dedup[3]
        params[26], params[27] = 150, 2
        stages = [
            ('collision_cuda::cf_copy_f64', [source, shadow, 10], 'block'),
            ('collision_cuda::cf_copy_i32', [source, destination, 10], 'block'),
            ('collision_cuda::cf_tree', [tree.stage, tree.stage_count, tree.active_in], 'block'),
            ('collision_cuda::cf_calls_finalize', [], 'block'),
            ('collision_cuda::cf_dedup', dedup, 'block'),
            ('collision_cuda::cf_params', params, 'block'),
            ('collision_cuda::cf_restore_f64',
                [source if invalid_restore else active, shadow, destination, 10], 'block'),
            None,
        ]
        storage = {k: array(i+100) for i, k in enumerate(
            ('offsets', 'frame_error', 'contact_f', 'contact_i'))}
        tree_ast = ast.parse(SOURCE.read_text())
        function = next(n for n in tree_ast.body
                        if isinstance(n, ast.FunctionDef) and n.name == '_dataflow_stages')
        namespace = {'_workspace_stages': lambda ws: stages,
                     '_dataflow_storage': lambda ws: storage,
                     '_pair_parameter_cache': lambda ws, params, storage: []}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        metadata = SimpleNamespace(prepare_collision_metadata=lambda ws: {})
        with patch.dict(sys.modules, {'libero.gpu.host.build.collision_metadata': metadata}):
            return namespace['_dataflow_stages'](ws)

    def test_formal_fast_keeps_physics_copy_and_uses_live_contact_rows(self):
        names = [stage[0] for stage in self.stages() if stage is not None]
        self.assertNotIn('collision_cuda::cf_copy_f64', names)
        self.assertNotIn('collision_cuda::cf_restore_f64', names)
        self.assertIn('collision_cuda::cf_copy_i32', names)
        self.assertIn('collision_cuda::cf_dedup_params_live', names)
        self.assertIn('collision_cuda::cf_calls_finalize_live', names)
        self.assertIn('collision_cuda::cf_commit_contacts_live', names)
        self.assertIn('collision_cuda::cf_tree_live', names)
        self.assertNotIn('collision_cuda::cf_tree', names)

    def test_debug_and_compatible_keep_capacity_diagnostics(self):
        for options in ({'diagnostics': True}, {'profile': 'compatible'}):
            with self.subTest(options=options):
                names = [stage[0] for stage in self.stages(**options) if stage is not None]
                self.assertIn('collision_cuda::cf_copy_f64', names)
                self.assertIn('collision_cuda::cf_restore_f64', names)
                self.assertIn('collision_cuda::cf_dedup_params', names)
                self.assertIn('collision_cuda::cf_calls_finalize', names)
                self.assertIn('collision_cuda::cf_commit_contacts', names)
                self.assertIn('collision_cuda::cf_tree', names)
                self.assertNotIn('collision_cuda::cf_tree_live', names)
                self.assertNotIn('collision_cuda::cf_dedup_params_live', names)

    def test_unexpected_restore_owner_cannot_be_silently_omitted(self):
        with self.assertRaisesRegex(ValueError, 'restore owner'):
            self.stages(invalid_restore=True)


class PairParameterCacheTests(unittest.TestCase):
    def setUp(self):
        node = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_pair_parameter_cache')
        self.plan = Mock()
        self.plan.return_value = SimpleNamespace(launch=Mock())
        self.warp = SimpleNamespace(empty=Mock(side_effect=lambda *a, **k: object()),
            float64='f64', int32='i32', get_stream=Mock(return_value='stream'),
            synchronize_stream=Mock())
        namespace = {'wp': self.warp, 'NativePlan': self.plan,
                     'GPU_ROOT': Path('/gpu'), 'UserDict': UserDict}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.cache = namespace['_pair_parameter_cache']
        self.params = [object() for _ in range(29)]
        self.params[27:29] = [2, 0]
        self.storage = {'contact_f': object(), 'contact_i': object()}

    def workspace(self, *, immutable=True):
        return SimpleNamespace(_cuda_collision_pair_parameter_root=Path('/build'),
            _collision_cold_admission=object() if immutable else None, device='cuda:0')

    def test_mutable_workspace_uses_direct_mixing(self):
        result = self.cache(self.workspace(immutable=False), self.params, self.storage)
        self.assertEqual(result, [self.storage['contact_f'], self.storage['contact_i'], 0])
        self.plan.assert_not_called()
        self.warp.empty.assert_not_called()

    def test_native_batch_static_parameter_contract_enables_shared_table(self):
        ws = self.workspace(immutable=False)
        ws._cuda_collision_static_parameters = True
        self.assertEqual(self.cache(ws, self.params, self.storage)[2], 2)
        self.plan.assert_called_once()

    def test_static_table_is_initialized_once_and_new_model_gets_new_table(self):
        ws = self.workspace()
        first = self.cache(ws, self.params, self.storage)
        self.assertEqual(first, self.cache(ws, self.params, self.storage))
        self.assertEqual(self.plan.call_count, 1)
        self.plan.return_value.launch.assert_called_once_with('stream')
        self.warp.synchronize_stream.assert_called_once_with('stream')
        second = self.cache(self.workspace(), self.params, self.storage)
        self.assertIsNot(first[0], second[0])
        self.assertEqual(self.plan.call_count, 2)

    def test_large_model_keeps_direct_path_without_allocating_dense_table(self):
        self.params[27] = 10000
        self.assertEqual(self.cache(self.workspace(), self.params, self.storage)[2], 0)
        self.plan.assert_not_called()
        self.warp.empty.assert_not_called()


class CompleteCollisionInstallationTests(unittest.TestCase):
    def test_combined_installation_skips_partial_products_for_both_storage_backends(self):
        node = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'install_workspace')
        for backend in ('legacy', 'handwritten_only'):
            with self.subTest(backend=backend):
                plan = Mock(side_effect=lambda name, *a, **k: SimpleNamespace(
                    manifest={'name': name}, launch=Mock()))
                stages = [('before', [], 'block'), None, ('after', [], 'block')]
                namespace = dict(Path=Path, MethodType=MethodType, NativePlan=plan,
                    GPU_ROOT=Path('/gpu'), _source_files=lambda ws: [],
                    _dataflow_stages=lambda ws: stages,
                    _guard_inactive=lambda ws, stages: (stages, ''),
                    temporal_front_manifest=lambda ws: {}, _validate_temporal_position_owner=Mock())
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
                ws = SimpleNamespace(_detector_backend=backend, device='cuda:0',
                    launch=Mock(), reset_errors=Mock(), reset_mask=object(),
                    fe_stage=object(), fe_code=object(), fe_item=object(),
                    borrow_detector_raw=False, _device_input_binding=None,
                    validate_device_bindings=Mock(), **{k:2 for k in
                        ('nworld','nbody','ngeom','call_capacity','pair_capacity',
                         'stage_slots','req_slots','contact_capacity')})
                namespace['install_workspace'](ws, Path('/cache'),
                    dataflow_profile='fast', complete_only=True, diagnostics=False)
                self.assertEqual([call.args[0] for call in plan.call_args_list],
                                 ['collision_reset_errors'])
                self.assertEqual(ws._cuda_collision_plans, [])
                with self.assertRaisesRegex(RuntimeError, 'requires finalized complete plan'):
                    ws.launch(stream='stream')

    def test_pending_complete_backend_skips_replaced_transport(self):
        node = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'install')
        workspace = SimpleNamespace()
        bundle = SimpleNamespace(dynamics=object(), runtime=SimpleNamespace(forward=SimpleNamespace(
            c=SimpleNamespace(workspace=workspace), transport=object())))
        install_workspace = Mock(return_value={})
        install_transport = Mock(return_value='transport')
        namespace = dict(Path=Path, configure_temporal_front=Mock(),
                         install_workspace=install_workspace, install_transport=install_transport)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        for complete_only in (True, False):
            with self.subTest(complete_only=complete_only):
                install_workspace.reset_mock(); install_transport.reset_mock()
                namespace['install'](bundle, Path('/cache'), dataflow_profile='fast',
                                     diagnostics=False, complete_only=complete_only)
                self.assertEqual(install_workspace.call_args.kwargs['complete_only'], complete_only)
                self.assertEqual(install_transport.call_count, 0 if complete_only else 1)


if __name__ == '__main__':
    unittest.main()
