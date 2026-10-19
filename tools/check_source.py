"""Portable source and CPU-maintenance checks; no CUDA/GL/physics execution."""
from pathlib import Path
import argparse
import ast
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "libero/libero/gpu"
SKIP = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
CPU_TESTS = ("test_installation_layout.py", "test_resource_paths.py", "test_product_tool_paths.py",
             "test_native_capsule_sources.py", "test_gpu_memory_budget.py", "test_gpu_batch_env.py",
             "test_deferred_native_build.py", "test_release_inputs.py",
             "test_validation_tools.py", "test_validation_runner.py")


def source_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.endswith(".egg-info")
                   and not (Path(base) == root and d in {"build", "dist", ".venv"})]
        for name in files:
            yield Path(base) / name


def check():
    python_count = 0
    missing = set()
    for p in source_files(ROOT):
        if p.suffix == ".py":
            tree = ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
            python_count += 1
            if p.is_relative_to(ROOT / "libero") or p.is_relative_to(ROOT / "scripts"):
                for node in ast.walk(tree):
                    modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                               [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    for module in modules:
                        if p.is_relative_to(GPU) and (module == "scripts" or module.startswith("scripts.")):
                            raise RuntimeError("GPU product code imports development scripts: " +
                                               str(p.relative_to(ROOT)) + " -> " + module)
                        if module.startswith("libero.gpu."):
                            target = GPU.joinpath(*module.split(".")[2:])
                        elif module.startswith("scripts."):
                            target = ROOT.joinpath(*module.split("."))
                        else:
                            continue
                        if not (target.with_suffix(".py").is_file() or target.is_dir()):
                            missing.add((str(p.relative_to(ROOT)), module))
        elif p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8-sig"))
        elif p.suffix == ".xml":
            ET.parse(p)
    if missing:
        raise RuntimeError("Missing local import dependencies: " + repr(sorted(missing)))
    required = ("licenses/LIBERO-MIT.txt", "licenses/Apache-2.0.txt", "licenses/libccd-BSD-3-Clause.txt",
                "LICENSE", "libero/libero/upstream_resources.json", "templates/scene_template.xml",
                "CMakeLists.txt", "libero/libero/gpu/CMakeLists.txt",
                "libero/libero/gpu/cmake/Substep.cmake",
                "libero/libero/gpu/host/build/native.py", "benchmark_scripts/benchmark_gpu_batch.py")
    for relative in required:
        if not (ROOT / relative).is_file():
            raise RuntimeError("Required source/resource missing: " + relative)
    import libero
    if Path(libero.__file__).resolve() != ROOT / "libero/__init__.py":
        raise RuntimeError("Python imports another checkout: " + str(libero.__file__))
    spec = importlib.machinery.PathFinder.find_spec("libero.gpu", libero.__path__)
    if spec is None or Path(spec.origin).resolve() != GPU / "__init__.py":
        raise RuntimeError("The canonical GPU package does not resolve to this checkout")
    print(json.dumps({"source_python_files": python_count, "local_imports": "complete", "CUDA_executed": False}))


def cpu_tests():
    with tempfile.TemporaryDirectory(prefix="flash-libero-cpu-checks-") as work:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LIBERO_TEST_SOURCE=str(ROOT),
                   LIBERO_TEST_TMP=str(Path(work) / "resources"), LIBERO_CONFIG_PATH=str(Path(work) / "config"),
                   LIBERO_AUTO_DOWNLOAD="1")
        for filename in CPU_TESTS:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", filename],
                           cwd=ROOT, env=env, check=True, timeout=90)
        subprocess.run([sys.executable, "tests/test_gpu_vector_protocol.py"], cwd=ROOT, env=env,
                       check=True, timeout=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-tests", action="store_true")
    args = parser.parse_args()
    check()
    if args.cpu_tests:
        cpu_tests()
