"""Resident collision checks without filesystem certificates or GPU execution."""
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libero.gpu.host.runtime.collision.cold_geometry import CollisionColdAdmission
from libero.gpu.host.runtime.collision import shared
from libero.gpu.host.runtime.core.model_bytes import actual_decoder_record


class ModelBindingTests(unittest.TestCase):
    def prepare(self):
        context = dict(model_identity='model', spec_identity='spec', generation=1,
                       field_permissions={'mass': 'static'})
        source = np.arange(6, dtype=np.float64).reshape(2, 3)
        admission = CollisionColdAdmission({'body_pos': source, 'count': 2}, context=context)
        workspace = NS(nworld=2, ngeom=2, nbody=2, slots=16, device='cuda:0')
        admission.attach(workspace)
        plans = (NS(manifest={'diagnostic': 'old'}),)
        admission.seal(plans)
        return admission, workspace, source

    def test_construction_and_validation_do_not_read_source_or_products(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('source/binary read')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('source read')), \
                patch.object(Path, 'rglob', side_effect=AssertionError('source tree scan')):
            admission, _, source = self.prepare()
            source[:] = 100
            self.assertTrue(admission.validate())
            admission.plans[0].manifest = {'diagnostic': 'revised'}
            self.assertTrue(admission.validate())
            self.assertTrue(admission.manifest()['host_static_reuse'])
        self.assertFalse(admission.static['body_pos'].flags.writeable)

    def test_model_context_static_values_capacity_and_options_still_checked(self):
        for change in (
                lambda a, w: a.context_source.update(generation=2),
                lambda a, w: a.static.update(count=3),
                lambda a, w: a.static.update(body_pos=np.ones((2, 3))),
                lambda a, w: setattr(w, 'slots', 32),
                lambda a, w: setattr(w, 'device', 'cuda:1'),
                lambda a, w: a.options.update(multiccd=True),
                lambda a, w: setattr(a, 'plans', (NS(),))):
            with self.subTest(change=change):
                admission, workspace, _ = self.prepare()
                change(admission, workspace)
                with self.assertRaises(ValueError):
                    admission.validate()

    def test_installation_and_close_contracts(self):
        admission = CollisionColdAdmission({})
        with self.assertRaisesRegex(ValueError, 'not sealed'):
            admission.validate()
        with self.assertRaisesRegex(ValueError, 'plans required'):
            admission.seal(())
        admission, workspace, _ = self.prepare()
        with self.assertRaisesRegex(ValueError, 'one-shot'):
            admission.attach(workspace)
        with self.assertRaisesRegex(ValueError, 'one-shot'):
            admission.seal(admission.plans)
        admission.revoke()
        self.assertEqual(admission.plans, ())
        with self.assertRaisesRegex(ValueError, 'revoked'):
            admission.validate()

    def installed_workspace(self):
        class Params(ct.Structure):
            _fields_ = [('pointer', ct.c_void_p)]
        function = ct.CFUNCTYPE(ct.c_int)(lambda: 0)
        library = NS(_handle=17, launch=function, launch_collision_detect_tiled=function,
                     launch_collision_detect_compact_producer=function,
                     launch_collision_detect_parallel=function)
        owner = NS(ptr=1000, shape=(2, 3), strides=(24, 8), dtype=np.float64, device='cuda:0')
        plans = [NS(library=library, params=Params(1000), device_params=NS(ptr=2000),
                    owners=[owner], device='cuda:0', manifest={'build': 'old'},
                    validate=Mock(), validate_bindings=Mock()) for _ in range(2)]
        ws = NS(_detector_backend='handwritten_only', _cuda_collision_complete=plans[1],
                detect=NS(native=None, _cuda_detector_plan=plans[0]),
                tree=NS(), dispatch=NS(), finalize=NS(params=NS()), device='cuda:0')
        for name in ('nworld', 'ngeom', 'nbody', 'npair_static', 'nexclude', 'slots',
                     'call_capacity', 'pair_capacity', 'stage_slots', 'req_slots', 'contact_capacity'):
            setattr(ws, name, 2)
        admission = CollisionColdAdmission({})
        admission.attach(ws)
        admission.seal(plans)
        return ws, plans, admission

    def test_loaded_exports_buffers_and_parameters_remain_bound(self):
        ws, plans, _ = self.installed_workspace()
        baseline = shared._native_token(ws)
        for plan in plans:
            plan.manifest = {'build': 'revised'}
            plan.validate = lambda: None
            plan.validate_bindings = lambda: None
        self.assertEqual(shared._native_token(ws), baseline)
        plans[0].params.pointer = 3000
        self.assertNotEqual(shared._native_token(ws), baseline)
        plans[0].params.pointer = 1000
        plans[0].owners[0].ptr = 4000
        self.assertNotEqual(shared._native_token(ws), baseline)
        plans[0].owners[0].ptr = 1000
        plans[0].library.launch_collision_detect_tiled = ct.CFUNCTYPE(ct.c_int)(lambda: 1)
        self.assertNotEqual(shared._native_token(ws), baseline)

    def test_binding_validates_model_once_and_rejects_owner_and_stream_changes(self):
        ws, _, admission = self.installed_workspace()
        binding = object.__new__(shared.DeviceInputBinding)
        stream = NS(device='cuda:0', cuda_stream=100)
        binding._native = shared._native_token(ws)
        binding._admission = admission
        binding._stream_token = (id(stream), str(stream.device), stream.cuda_stream)
        binding._scalars = (2,) * 11
        binding._modules = shared._consumer_modules(ws)
        binding._module_scalars = binding._scalar_state(ws)
        binding._argument_tokens = {}
        binding.NAMES = ()  # No source edges in this host-only fixture.
        with patch.object(admission, 'validate', wraps=admission.validate) as validate:
            self.assertTrue(binding.validate(ws, stream))
            validate.assert_called_once_with(require_installed=True)
        stream.cuda_stream += 1
        with self.assertRaisesRegex(shared.DeviceBindingError, 'stream replaced'):
            binding.validate(ws, stream)
        stream.cuda_stream -= 1
        ws._collision_cold_admission = NS()
        with self.assertRaisesRegex(shared.DeviceBindingError, 'retained model binding'):
            binding.validate(ws, stream)

    def test_decoder_description_does_not_reauthenticate_source(self):
        class Built:
            metadata = {'decoder_pin': {'source_sha256': '1' * 64, 'module': 'old.name'}}
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('decoder source read')):
            record = actual_decoder_record(Built())
        self.assertEqual(record['sha256'], '1' * 64)
        self.assertEqual(record['path'], str(Path(__file__).resolve()))


if __name__ == '__main__':
    unittest.main()
