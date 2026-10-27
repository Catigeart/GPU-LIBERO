"""Cold compiler/config path selection checks without importing GPU modules."""
import ast
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def gpu_source():
    import libero
    import importlib.machinery
    spec = importlib.machinery.PathFinder.find_spec("libero.gpu", libero.__path__)
    return Path(spec.origin).parent


class ColdToolPathTests(unittest.TestCase):
    def test_explicit_compiler_and_path_discovery(self):
        path = gpu_source() / "host/runtime/core/product_paths.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "nvcc_path")
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / "toolkit with spaces/bin/nvcc"
            compiler.parent.mkdir(parents=True)
            compiler.write_text("never executed\n")
            for environment, located in (({"LIBERO_NVCC": str(compiler)}, None), ({}, str(compiler))):
                globals_ = {"os": types.SimpleNamespace(environ=environment),
                            "shutil": types.SimpleNamespace(which=lambda name: located), "Path": Path}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), globals_)
                self.assertEqual(globals_["nvcc_path"](), str(compiler.resolve()))
            globals_["os"].environ["LIBERO_NVCC"] = str(Path(directory) / "absent")
            with self.assertRaises(FileNotFoundError):
                globals_["nvcc_path"]()

    def test_collision_override_has_priority_and_newton_uses_selected_tools(self):
        native = gpu_source() / "host/runtime/collision/native.py"
        tree = ast.parse(native.read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "nvcc" for t in n.targets))
        calls = []
        def selected():
            calls.append(True)
            return "/chosen/toolkit/bin/nvcc"
        for environment, expected in (({"P06C_NVCC": "/private/compiler"}, "/private/compiler"),
                                      ({}, "/chosen/toolkit/bin/nvcc")):
            namespace = {"Path": Path, "os": types.SimpleNamespace(environ=environment), "nvcc_path": selected, "offline": True}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(native), "exec"), namespace)
            self.assertEqual(str(namespace["nvcc"]), expected)
        self.assertEqual(len(calls), 1)
        for name in ("host/build/newton.py", "host/runtime/newton/persistent_cuda.py"):
            source = (gpu_source() / name).read_text()
            self.assertNotIn("'/usr/local/cuda/bin/nvcc'", source)
            self.assertIn("nvcc_path()", source)
        source = (gpu_source() / "host/runtime/newton/handwritten_cuda.py").read_text()
        self.assertNotIn('nvcc_path()', source)
        self.assertNotIn('cuobjdump', source)

    def test_compiler_child_config_default_and_explicit(self):
        warp_before = "warp" in sys.modules
        # Load this standalone stdlib-only helper without GPU package execution.
        helper = runpy.run_path(str(gpu_source() / "host/build/child_environment.py"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "default"
            explicit = root / "explicit"
            for config in (default, explicit):
                config.mkdir()
                (config / "config.yaml").write_text("assets: /resource-cache\n")
            stub = types.ModuleType("libero.libero")
            stub.libero_config_path = str(default)
            with patch.dict(sys.modules, {"libero.libero": stub}):
                for selected, expected in ((None, default), (str(explicit), explicit)):
                    environment = {"PATH": os.environ.get("PATH", "")}
                    if selected is not None:
                        environment["LIBERO_CONFIG_PATH"] = selected
                    with patch.dict(os.environ, environment, clear=True):
                        before = dict(os.environ)
                        child = helper["child_env"](root / ("default-child" if selected is None else "explicit-child"))
                        self.assertEqual(dict(os.environ), before)
                    copy = Path(child["LIBERO_CONFIG_PATH"]) / "config.yaml"
                    self.assertEqual(copy.read_bytes(), (expected / "config.yaml").read_bytes())
        self.assertEqual("warp" in sys.modules, warp_before)


if __name__ == "__main__":
    unittest.main()
