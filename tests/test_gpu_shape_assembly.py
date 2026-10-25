"""FP64 constructor selection and failure cleanup without a CUDA device."""
from dataclasses import FrozenInstanceError
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from libero.gpu.host.runtime.step import recovery_event_factory as event
from libero.gpu.host.runtime.step import shape_prepare
from libero.gpu.host.runtime.step.shape_assembly import ShapeAssembly


class ShapeAssemblyTests(unittest.TestCase):
    def collect(self, factory):
        collected = []
        def prepare(*args, _assembly, **kwargs):
            collected.append(_assembly)
            return SimpleNamespace(dynamics=object(), post=object(), close=lambda: None)
        with patch.object(shape_prepare, 'prepare_shape_physics', side_effect=prepare):
            bundle = factory()
        return collected.pop(), bundle

    def test_event_defaults_and_partial_override_are_private(self):
        from libero.gpu.host.runtime.dynamics import wide_factory
        from libero.gpu.reference.integration.euler_integrator.wide_angle import WideAngleEulerWorkspace
        first, _ = self.collect(event.prepare_recovery_event_physics)
        chosen = lambda: ('collision', 'newton', 'cn')
        second, _ = self.collect(lambda: event.prepare_recovery_event_physics(
            _assembly=ShapeAssembly(cold_factories=chosen)))
        third, _ = self.collect(event.prepare_recovery_event_physics)
        self.assertIs(first.dynamics_factory, wide_factory.prepare_from_built)
        self.assertIs(first.euler_type, WideAngleEulerWorkspace)
        self.assertIs(first.runtime_factory, event._recovery_loop)
        self.assertIs(second.cold_factories, chosen)
        self.assertIs(second.dynamics_factory, first.dynamics_factory)
        self.assertIs(first.cold_factories, third.cold_factories)
        with self.assertRaises(FrozenInstanceError):
            first.cold_factories = chosen

    def test_all_fusion_entries_keep_their_constructor_choices(self):
        from libero.gpu.host.assembly.compose import factory as first, factory_r02 as second
        from libero.gpu.host.assembly.compose import backup, s_rows
        from libero.gpu.host.assembly.pipeline import factory as third
        one, _ = self.collect(first.prepare_fused_physics)
        two, _ = self.collect(second.prepare_fused_physics)
        with patch.object(third, '_validate_selection', return_value=True), \
             patch.object(third.dynamics, 'plan_catalog', return_value={}):
            three, bundle = self.collect(third.prepare_fused_physics)
        self.assertIs(one.dynamics_factory, first.prepare_from_built)
        self.assertIs(one.cold_factories, first._cold_factories)
        self.assertIs(one.runtime_factory, event._recovery_loop)
        self.assertIsNone(one.forward_type)
        self.assertIs(two.dynamics_factory, first.prepare_from_built)
        self.assertIs(two.forward_type, backup.FusedShapePhysicsForward)
        self.assertIs(two.constraint_workspace_type, s_rows.fused_workspace_type)
        self.assertIs(two.runtime_factory.func, event._recovery_loop)
        self.assertIs(two.runtime_factory.keywords['runtime_type'], backup.FusedEventRecoveringRuntime)
        self.assertIs(three.dynamics_factory, third.prepare_from_built)
        self.assertIs(three.post_factory, third.prepare_post)
        self.assertIs(three.forward_type, third.FusedShapePhysicsForward)
        self.assertIs(three.foundation_type, third.FusedFoundationForwardAdapter)
        self.assertIs(three.controller_type, third.FusedRuntimeControllerAdapter)
        self.assertIs(three.euler_type, third.euler.FusedEulerWorkspace)
        self.assertIs(three.runtime_factory.keywords['runtime_type'], third.FusedRuntime)
        self.assertIs(three.runtime_factory.keywords['recovery_workspace_type'],
                      third.recovery.FusedDataRecoveryWorkspace)
        self.assertEqual(bundle.fusion_batch, 3)

    def test_fusion_failure_closes_the_completed_bundle(self):
        from libero.gpu.host.assembly.pipeline import factory
        closed = []
        bundle = SimpleNamespace(dynamics=object(), post=object(), close=lambda: closed.append(True))
        with patch.object(shape_prepare, 'prepare_shape_physics', return_value=bundle), \
             patch.object(factory.dynamics, 'plan_catalog', side_effect=RuntimeError('plan failure')):
            with self.assertRaisesRegex(RuntimeError, 'plan failure'):
                factory.prepare_fused_physics()
        self.assertEqual(closed, [True])

    def test_failed_shape_allocation_synchronizes_and_preserves_error(self):
        stream = object()
        error = RuntimeError('allocation failure')
        assembly = ShapeAssembly()
        with patch.object(shape_prepare, '_prepare_shape_physics', side_effect=error) as prepare, \
             patch.object(shape_prepare.wp, 'synchronize_stream') as synchronize, \
             patch.object(shape_prepare.gc, 'collect'):
            with self.assertRaisesRegex(RuntimeError, 'allocation failure') as caught:
                shape_prepare.prepare_shape_physics.__wrapped__(object(), state={}, stream=stream,
                                                                _assembly=assembly)
        self.assertIs(caught.exception, error)
        self.assertIs(prepare.call_args.kwargs['_assembly'], assembly)
        synchronize.assert_called_once_with(stream)

    def test_failed_sync_is_attached_to_original_error(self):
        error = RuntimeError('allocation failure')
        with patch.object(shape_prepare, '_prepare_shape_physics', side_effect=error), \
             patch.object(shape_prepare.wp, 'synchronize_stream', side_effect=RuntimeError('sync failure')):
            with self.assertRaisesRegex(RuntimeError, 'allocation failure') as caught:
                shape_prepare.prepare_shape_physics.__wrapped__(object(), state={}, stream=object())
        self.assertIs(caught.exception, error)
        self.assertEqual(error.partial_cleanup_error, 'sync failure')

    def test_explicit_historical_recovery_entries_still_reach_intake(self):
        from libero.gpu.host.runtime.step.shape_schema import RuntimeShape
        # These explicit variants still derive their constructor from the base
        # source. Exercise compilation plus the ordinary pre-allocation failure.
        entries = (
            ('recovery_factory', 'prepare_recovery_physics'),
            ('recovery_wide_factory', 'prepare_wide_recovery_physics'),
            ('recovery_final_event_factory', 'prepare_final_event_recovery_physics'),
            ('recovery_full_angle_factory', 'prepare_full_angle_recovery_physics'),
            ('recovery_inventory_factory_r01', 'prepare_inventory_recovery_physics'),
            ('quality_factory_r01', 'prepare_quality_recovery_physics'),
        )
        with patch.object(RuntimeShape, 'from_native_metadata', return_value=object()), \
             patch.object(shape_prepare.wp, 'synchronize_stream'):
            for module, name in entries:
                with self.subTest(entry=name):
                    prepare = getattr(importlib.import_module('libero.gpu.host.runtime.step.' + module), name)
                    with self.assertRaisesRegex(ValueError, 'exact native ModelSpec'):
                        prepare(SimpleNamespace(native_model=object(), metadata={}),
                                state={}, stream=object())


if __name__ == '__main__':
    unittest.main()
