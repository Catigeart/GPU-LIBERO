"""Public install contract checks; usable against source, editable, and wheels.

Run with ``python -m unittest discover -s tests -p test_installation_layout.py``.
No GPU module is executed and no CUDA context or compiler is created.
"""
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import unittest


class InstallationLayoutTests(unittest.TestCase):
    def test_original_public_packages_and_cli_modules_exist(self):
        for name in ("libero.libero", "libero.lifelong", "libero.configs",
                     "scripts.config_copy", "scripts.create_template", "templates"):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.util.find_spec(name))

    def test_gpu_is_a_real_package_with_runtime_compiler_sources(self):
        warp_before = "warp" in sys.modules
        import libero
        spec = importlib.machinery.PathFinder.find_spec("libero.gpu", libero.__path__)
        self.assertIsNotNone(spec)
        gpu = Path(spec.origin).parent
        for relative in ("cuda/newton/hessian.cuh", "host/factory.py",
                         "cuda/collision/native/native_kernel.cu",
                         "CMakeLists.txt", "cmake/Substep.cmake",
                         "host/runtime/collision/native_originals/engine_collision_box.c.txt"):
            with self.subTest(source=relative):
                self.assertTrue((gpu / relative).is_file())
        self.assertFalse(any(name == "libero.libero.gpu" or
                            name.startswith("libero.libero.gpu.")
                            for name in sys.modules))
        self.assertEqual("warp" in sys.modules, warp_before)
        self.assertTrue(list((gpu / "cuda/observations/qualified_r02/assets").glob("*.cu")))

    def test_runtime_catalog_and_templates_are_installed(self):
        spec = importlib.util.find_spec("libero.libero")
        library = Path(spec.origin).parent
        self.assertTrue((library / "upstream_resources.json").is_file())
        templates = importlib.util.find_spec("templates")
        self.assertTrue(any((Path(base) / "scene_template.xml").is_file()
                            for base in templates.submodule_search_locations))


if __name__ == "__main__":
    unittest.main()
