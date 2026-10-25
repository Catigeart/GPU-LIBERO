"""Offline MJWarp product lookup and ABI rejection without importing MJWarp."""
import ast
import ctypes as ct
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.build import project_paths
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.runtime.dynamics import mjwarp_motion


# Import only the real locator function; module-level MJWarp imports and Warp
# decorators are unrelated to product discovery and need no GPU in these tests.
def product_locator(module, function):
    path = GPU_ROOT / 'host/backends/mjwarp' / (module + '.py')
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == function)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[function]


PRODUCTS = (
    ('publication', 'publication_product', ('publication',)),
    ('euler', 'euler_product', ('euler',)),
    ('transmission', 'transmission_product', ('transmission',)),
    ('com_pos', 'com_pos_product', ('com_pos',)),
    ('acceleration', 'mass_product', ('mass_solve', 'smooth_force')),
    ('collision', 'broadphase_product', ('broadphase', 'plane', 'box', 'convex')),
    ('camlight', 'camlight_product', ('camlight',)),
    ('kinematics', 'kinematics_product', ('kinematics',)),
    ('velocity', 'velocity_product', ('velocity', 'bias', 'postconstraint', 'passive')),
    ('newton', 'hessian_product', ('hessian', 'direction', 'constraint', 'linesearch', 'solver_state')),
    ('queries', 'queries_product', ('queries',)),
    ('constraint_build', 'constraint_build_product', ('constraint_build',)),
    ('mass_matrix', 'matrix_product', ('mass_matrix',)),
    ('actuation', 'actuation_product', ('actuation',)),
    ('mass_factor', 'factor_product', ('mass_factor',)),
    ('external_wrench', 'external_wrench_product', ('external_wrench',)),
    ('fluid', 'fluid_product', ('fluid',)),
)


class ProductLookupTests(unittest.TestCase):
    def test_all_products_load_without_ninja_or_a_build_tree(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
                'subprocess.run', side_effect=AssertionError('runtime build-tool invocation')):
            root = Path(temporary)
            for module, function, components in PRODUCTS:
                locator = product_locator(module, function)
                for fast_compile in (False, True):
                    for component in components:
                        with self.subTest(module=module, component=component, fast_compile=fast_compile):
                            options = dict(fast_compile=fast_compile)
                            if len(components) > 1:
                                options['component'] = component
                            folder = root / 'cmake-control' / ('sm80-dev' if fast_compile else 'sm80-release')
                            path = folder / ('mjwarp_' + component + '_fp32.so')
                            with self.assertRaisesRegex(MissingCompiledProduct, 'flash_mjwarp_' + component + '_fp32'):
                                locator(root, **options)
                            self.assertFalse(path.exists())
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b'offline library fixture')
                            self.assertEqual(locator(root, **options), path)
                            if component == components[0]:
                                self.assertEqual(locator(root, fast_compile=fast_compile), path)
                            path.unlink()
                            path.mkdir()
                            with self.assertRaises(MissingCompiledProduct):
                                locator(root, **options)
                            path.rmdir()

    def test_existing_component_selection_rejects_unknown_variants(self):
        for module, function in (('acceleration', 'mass_product'), ('velocity', 'velocity_product'),
                                 ('newton', 'hessian_product')):
            with self.subTest(module=module), self.assertRaises(ValueError):
                product_locator(module, function)('/no-build-tree', component='unknown')

    def test_precision_namespace_resolves_shared_products_and_rejects_wrong_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = root / 'fp32'
            namespace.mkdir()
            (namespace / 'PROJECT_NAMESPACE.json').write_text(json.dumps(dict(
                schema='libero-project-namespace-v1', real_bits=32, base_relative='..')))
            product = root / 'cmake-control/sm80-release/mjwarp_euler_fp32.so'
            product.parent.mkdir(parents=True)
            product.write_bytes(b'offline library fixture')
            locate = product_locator('euler', 'euler_product')
            with patch.object(project_paths.scalar, 'bits', 32):
                self.assertEqual(locate(namespace), product)
            with patch.object(project_paths.scalar, 'bits', 64), self.assertRaisesRegex(ValueError, 'FP32 namespace'):
                locate(namespace)


class LeafAdmissionTests(unittest.TestCase):
    def test_binding_checks_abi_architecture_and_entries_without_source_files(self):
        class Arguments(ct.Structure):
            _fields_ = [('worlds', ct.c_int), ('values', ct.c_void_p)]

        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / 'mjwarp_queries_fp32.so'
            product.write_bytes(b'offline library fixture')
            stream = SimpleNamespace(device=SimpleNamespace(is_cuda=True, arch=80))
            library = SimpleNamespace(
                flash_mjwarp_queries_source_id=Mock(return_value=b'current-source'),
                flash_mjwarp_queries_arguments_bytes=Mock(return_value=ct.sizeof(Arguments)),
                flash_mjwarp_queries_arch=Mock(return_value=80))
            with patch.object(mjwarp_motion.ct, 'CDLL', return_value=library), patch.object(
                    mjwarp_motion, 'source_id', side_effect=AssertionError('runtime source read')):
                self.assertEqual(mjwarp_motion.load_leaf('queries', Arguments, product, stream), (product, library))
                library.flash_mjwarp_queries_arguments_bytes.return_value += 8
                with self.assertRaisesRegex(ValueError, 'ABI differs'):
                    mjwarp_motion.load_leaf('queries', Arguments, product, stream)
                library.flash_mjwarp_queries_arguments_bytes.return_value = ct.sizeof(Arguments)
                library.flash_mjwarp_queries_source_id.assert_not_called()
                library.flash_mjwarp_queries_arch.return_value = 90
                with self.assertRaisesRegex(MissingCompiledProduct, 'architecture differs'):
                    mjwarp_motion.load_leaf('queries', Arguments, product, stream)
                library.flash_mjwarp_queries_arch.return_value = 80
                with self.assertRaisesRegex(MissingCompiledProduct, 'lacks extra entry'):
                    mjwarp_motion.load_leaf('queries', Arguments, product, stream, entry='extra')


if __name__ == '__main__':
    unittest.main()
