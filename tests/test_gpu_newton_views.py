"""The prepared Newton ABI must agree across CUDA/C++ and ctypes."""
import ctypes as ct
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

GPU = Path(__file__).resolve().parents[1] / "libero/libero/gpu"
spec = importlib.util.spec_from_file_location("newton_views", GPU / "host/runtime/step/newton_binding.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


class NewtonViewTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "host C++ compiler required")
    def test_bounded_cache_pitch_across_catalogue_sizes(self):
        program = '#include <iostream>\n#include "' + str(GPU / "cuda/step/newton_jacobian_layout.cuh") + '"\n'
        program += 'using namespace physics_newton;\nint main() {\n'
        expected = []
        for nv in (21,22,24,27,30,33,36,37,39,43,45,51,57,65):
            for rows in (0,1,17,30053):
                row_bytes = (rows * 8 + 127) // 128 * 128
                prefix = (128 + 3 * row_bytes + nv * (nv+1) // 2 * 8 + 127) // 128 * 128
                program += f'std::cout << {prefix} + NewtonJacobianLayout::bytes({rows}, {nv}) << "\\n";\n'
                expected.append(views.device_workspace_bytes(nv, rows))
        program += '}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pitch.cpp").write_text(program)
            subprocess.run(["g++", "-std=c++17", str(root / "pitch.cpp"), "-o", str(root / "pitch")],
                           check=True, capture_output=True)
            actual = subprocess.check_output([str(root / "pitch")], text=True)
        self.assertEqual([int(value) for value in actual.splitlines()], expected)

    @unittest.skipUnless(shutil.which("g++"), "host C++ compiler required")
    def test_all_descriptor_offsets_match_the_compiled_header(self):
        program = '#include <iostream>\n#include "' + str(GPU / "cuda/step/newton_types.cuh") + '"\n'
        program += '#include "' + str(GPU / "cuda/step/newton_coefficient_layout.cuh") + '"\n'
        program += 'using namespace physics_newton;\nint main() {\n'
        expected = []
        for kind in (views.NewtonModelData, views.NewtonWorldData,
                     views.NewtonPostData, views.NewtonArguments):
            program += f'std::cout << sizeof({kind.__name__}) << "\\n";\n'
            expected.append(ct.sizeof(kind))
            for name, _ in kind._fields_:
                program += f'std::cout << offsetof({kind.__name__}, {name}) << "\\n";\n'
                expected.append(getattr(kind, name).offset)
        for field,value in (("row_bytes",views.COEFFICIENT_ROW_BYTES),
                            ("sqrt_d",views.COEFFICIENT_SQRT_D),
                            ("friction_limit",views.COEFFICIENT_FRICTION_LIMIT),
                            ("cone_dm",views.COEFFICIENT_CONE_DM),
                            ("bytes",views.COEFFICIENT_BYTES)):
            program += f'std::cout << NewtonCoefficientLayout::{field} << "\\n";\n'
            expected.append(value)
        program += 'std::cout << NewtonDeviceWorkspaceLayout::atomic << "\\n";\n'
        program += 'std::cout << NewtonDeviceWorkspaceLayout::bytes << "\\n";\n'
        expected += [views.COEFFICIENT_BYTES, views.device_workspace_bytes(39,views.COEFFICIENT_ROWS)]
        program += '}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abi.cpp").write_text(program)
            subprocess.run(["g++", "-std=c++17", str(root / "abi.cpp"), "-o", str(root / "abi")],
                           check=True, capture_output=True)
            actual = subprocess.check_output([str(root / "abi")], text=True)
        self.assertEqual([int(value) for value in actual.splitlines()], expected)

    def test_old_library_is_rejected_before_a_kernel_can_be_launched(self):
        from libero.gpu.host.runtime.step import fused_substep
        fusion = fused_substep.FusedSubstep.__new__(fused_substep.FusedSubstep)
        fusion.manifest = {"library": "old-substep.so"}
        with patch.object(fused_substep.substep_native, "finish", return_value=fusion.manifest), \
                patch.object(fused_substep.ct, "CDLL", return_value=types.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "offline rebuild"):
                fusion.load(allow_build=False)

    def test_incompatible_inner_layout_is_rejected(self):
        from libero.gpu.host.runtime.step import fused_substep
        fusion = fused_substep.FusedSubstep.__new__(fused_substep.FusedSubstep)
        fusion.manifest = {"library": "incompatible-substep.so"}
        library = types.SimpleNamespace(flash_libero_newton_arguments_size=lambda: 1)
        with patch.object(fused_substep.substep_native, "finish", return_value=fusion.manifest), \
                patch.object(fused_substep.ct, "CDLL", return_value=library):
            with self.assertRaisesRegex(RuntimeError, "layout differs: arguments"):
                fusion.load(allow_build=False)

    def test_old_world_stride_is_rejected_even_when_root_size_matches(self):
        from libero.gpu.host.runtime.step import fused_substep
        fusion = fused_substep.FusedSubstep.__new__(fused_substep.FusedSubstep)
        fusion.manifest = {"library": "old-world-stride.so"}
        library = types.SimpleNamespace(
            flash_libero_newton_arguments_size=lambda: ct.sizeof(views.NewtonArguments),
            flash_libero_newton_world_size=lambda: 544)
        with patch.object(fused_substep.substep_native, "finish", return_value=fusion.manifest), \
                patch.object(fused_substep.ct, "CDLL", return_value=library):
            with self.assertRaisesRegex(RuntimeError, "layout differs: world"):
                fusion.load(allow_build=False)

    def _reduction_fixture(self, mode, workspace_bytes):
        from libero.gpu.host.runtime.step import fused_substep
        from libero.gpu.host.runtime.step.substep_types import SubstepArguments
        from libero.gpu.host.build.substep_catalog import select_variant
        fusion=fused_substep.FusedSubstep.__new__(fused_substep.FusedSubstep)
        fusion.variant=v=select_variant(39)
        fusion.arguments=SubstepArguments()
        fusion.manifest={'library':'fixture.so','config':{'atomic_reductions':True}}
        dimensions=(v.nv,v.nq,v.nM,v.njnt,v.nu,v.rows,v.contacts)
        library=types.SimpleNamespace(
            flash_libero_substep_arguments_size=lambda:ct.sizeof(fusion.arguments),
            flash_libero_substep_dimension=lambda i:dimensions[i],
            flash_libero_newton_atomic_reductions=lambda:mode,
            flash_libero_newton_device_workspace_bytes=lambda:workspace_bytes)
        for name,kind in [('arguments',views.NewtonArguments),('world',views.NewtonWorldData),
                          ('model',views.NewtonModelData),('post',views.NewtonPostData)]:
            setattr(library,'flash_libero_newton_'+name+'_size',lambda kind=kind:ct.sizeof(kind))
        return fused_substep,fusion,library

    def test_wrong_reduction_binary_is_rejected_before_launch(self):
        module,fusion,library=self._reduction_fixture(0,views.device_workspace_bytes(39,30053))
        with patch.object(module.substep_native,'finish',return_value=fusion.manifest), \
                patch.object(module.ct,'CDLL',return_value=library):
            with self.assertRaisesRegex(RuntimeError,'reduction mode differs'):
                fusion.load(allow_build=False)

    def test_old_coefficient_only_workspace_is_rejected(self):
        module,fusion,library=self._reduction_fixture(1,views.COEFFICIENT_BYTES)
        with patch.object(module.substep_native,'finish',return_value=fusion.manifest), \
                patch.object(module.ct,'CDLL',return_value=library):
            with self.assertRaisesRegex(RuntimeError,'workspace pitch differs'):
                fusion.load(allow_build=False)


if __name__ == "__main__":
    unittest.main()
