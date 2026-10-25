"""Explicit FP64 construction and transactional replacement, without CUDA."""
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np


class ConstructorTests(unittest.TestCase):
    def test_dynamics_and_post_constructors_are_forwarded_without_source_introspection(self):
        from libero.gpu.host.assembly.compose import factory as first
        from libero.gpu.host.assembly.pipeline import factory as third
        from libero.gpu.host.runtime.dynamics import wide_factory, wide_runtime
        with patch('inspect.getsource', side_effect=AssertionError('constructor source read')):
            for entry in (first, third):
                with patch.object(wide_factory, 'prepare_from_built', return_value='owner') as prepare:
                    self.assertEqual(entry.prepare_from_built('built', 2), 'owner')
                    self.assertIs(prepare.call_args.kwargs['adapter_type'], entry.FusedWideDynamicsAdapter)
            with patch.object(wide_factory, 'prepare_post', return_value='post') as prepare, \
                 patch.object(third.dynamics, 'prepare_post') as install:
                self.assertEqual(third.prepare_post('intake', 2), 'post')
                self.assertIs(prepare.call_args.kwargs['adapter_type'], third.FusedRuntimePostAdapter)
                install.assert_called_once_with('post')
            with patch.object(wide_runtime.GenericWideRuntimeDynamicsAdapter, '__init__', return_value=None) as base:
                first.FusedWideDynamicsAdapter('arrays')
                self.assertIs(base.call_args.kwargs['smooth_type'], first.FusedSmoothWorkspace)
                with patch.object(third.dynamics, 'prepare_dynamics'):
                    third.FusedWideDynamicsAdapter('arrays')
                self.assertIs(base.call_args.kwargs['hold_type'], third.FusedNoForwardHold)

    def test_newton_variants_keep_private_loader_and_adapter_selection(self):
        from libero.gpu.host.runtime.newton import live_copy_api
        from libero.gpu.host.runtime.step import newton_final_event, newton_jar_candidate_r01 as jar
        from libero.gpu.host.assembly.compose import factory as first, factory_r02 as second
        from libero.gpu.host.assembly.pipeline import factory as third
        from libero.gpu.host.assembly.compose.newton import FusedFinalEventAdapter
        variants = ((newton_final_event.prepare_final_event_runtime, newton_final_event.FinalEventAdapter, None),
                    (jar.prepare_jar_final_event_runtime, jar.FinalEventAdapter, jar.load_candidate),
                    (first._newton_factory, FusedFinalEventAdapter, jar.load_candidate),
                    (second._newton_factory, second.FusedFinalEventAdapter, second._load_candidate),
                    (third._newton_factory, third.newton.FusedFinalEventAdapter, third._load_candidate))
        with patch('inspect.getsource', side_effect=AssertionError('constructor source read')):
            for factory, adapter_type, loader in variants:
                owner = NS(solver=object(), source_manifest={})
                with patch.object(live_copy_api, 'prepare_live_copy_runtime', return_value=owner) as prepare:
                    result = factory(rows=16)
                self.assertIs(result.final_event, owner.solver)
                self.assertIs(prepare.call_args.kwargs['adapter_type'], adapter_type)
                self.assertIs(prepare.call_args.kwargs.get('candidate_loader'), loader)
                self.assertEqual(prepare.call_args.kwargs['rows'], 16)

    def test_collision_uses_explicit_workspace(self):
        from libero.gpu.host.assembly.pipeline import collision
        with patch.object(collision, '_original_adapter', return_value='owner') as prepare, \
             patch('inspect.getsource', side_effect=AssertionError('constructor source read')):
            self.assertEqual(collision.make_fused_adapter('static', 'plan', stream='stream'), 'owner')
        self.assertIs(prepare.call_args.kwargs['workspace_type'], collision.FusedCollisionChainWorkspace)


class ReplacementTests(unittest.TestCase):
    def exercise(self, failure=False):
        from libero.gpu.host.runtime.step import recovery_cleanup_environment_r02 as wrapper
        from libero.gpu.host.runtime.step import recovery_cleanup_factory_r02 as factory
        from libero.gpu.host.runtime.step import recovery_final_event_environment as source
        from libero.gpu.host.runtime.step import cold_model, continuation
        from libero.gpu.host.runtime.core import build_lifetime
        events = []
        class Array:
            def __init__(self, value): self.value = np.array([value])
            def assign(self, value): self.value = value.copy()
            def numpy(self): return self.value
        stream = NS(device='device')
        fresh = NS(runtime=NS(forward=NS(validate=lambda: events.append('validate'), valid=object()), valid=object()),
                   expected_model_epoch=Array(0), device_model_epoch=Array(0), model_ready=Array(0))
        failure_error = RuntimeError('restore failure')
        def close():
            events.append('close fresh')
            if failure: raise RuntimeError('release failure')
        fresh.close = close
        def prepare(*args, **kwargs):
            events.append('prepare')
            self.assertEqual(events[:3], ['admit', 'release old', 'prepare'])
            return fresh
        admission = NS(admission_domain_for_built=lambda *a, **kw: events.append('admit') or 'lease')
        bundle = NS(runtime=NS(), stream=stream, reset_ctrl=object(), initial_state={'qpos': np.zeros(1)},
                    domain_admission=admission, cold_recovery_prepare=prepare, domain_factory_entry='chosen-entry')
        built = object()
        spec = NS(model_sha256='new-model')
        preparation = NS(built=object(), spec=spec, constraint_package=object(), descriptor_plan=object())
        old = NS(release=lambda: events.append('release old'), observations=NS(snapshot_continuation=lambda **k: 'observations'))
        def launch(*args, inputs, **kwargs):
            expected, actual, ready, *_ = inputs
            actual.assign(expected.numpy()); ready.assign(np.ones(1, dtype=np.int32))
            events.append('commit')
        def restore(*args):
            events.append('restore')
            if failure: raise failure_error
            return 'restored'
        def assemble(*args, **kwargs):
            events.append('assemble')
            self.assertEqual(kwargs['observation_continuation'], 'observations')
            self.assertEqual(fresh.domain_lease, 'lease')
            self.assertIs(fresh.cold_recovery_prepare, prepare)
            return 'components'
        with ExitStack() as stack:
            for target, attr, replacement in (
                (source, 'RecoveryBoundaryStatus', lambda bundle: object()),
                (source.wp, 'ScopedStream', lambda stream: nullcontext()),
                (source.wp, 'launch', launch), (source.wp, 'synchronize_stream', lambda s: None),
                (cold_model, 'prepare_parameter_model', lambda *a: preparation),
                (continuation, 'continuation_snapshot', lambda b: dict(data={}, headers={}, source_model_sha256='old')),
                (continuation, 'restore_continuation', restore),
                (build_lifetime, 'cold_build_scope', lambda **kw: lambda f: f),
                (factory, '_publication', lambda *a: events.append('publish')),
                (factory, 'assemble_prepared', assemble),
            ):
                stack.enter_context(patch.object(target, attr, replacement))
            callbacks = wrapper.make_boundary_callbacks(bundle, built)
            callbacks['bind_parameter_replacement'](lambda build, *a: build(old))
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'restore failure') as caught:
                    callbacks['update_parameters_transaction']({}, 0, stream, NS(next_generation=2), object())
                self.assertIs(caught.exception, failure_error)
                self.assertEqual(len(caught.exception.pending_gpu_cleanup), 1)
                self.assertIs(caught.exception.pending_gpu_cleanup[0].owner, fresh)
                self.assertNotIn('publish', events)
                self.assertNotIn('assemble', events)
            else:
                result = callbacks['update_parameters_transaction']({}, 0, stream, NS(next_generation=2), object())
                self.assertEqual(result, ('components', preparation.descriptor_plan))
                self.assertEqual(events, ['admit', 'release old', 'prepare', 'restore', 'validate', 'commit', 'publish', 'assemble'])

    def test_replacement_retains_admission_publication_and_assembly_order(self):
        self.exercise()

    def test_failed_restore_retains_cleanup_owner_and_original_error(self):
        self.exercise(failure=True)


if __name__ == '__main__':
    unittest.main()
