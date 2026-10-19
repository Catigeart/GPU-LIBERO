"""Actual capsule source closure tests without loading GPU or CUDA tools."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest


def generator_namespace():
    import libero
    import importlib.machinery
    spec = importlib.machinery.PathFinder.find_spec("libero.gpu", libero.__path__)
    gpu = Path(spec.origin).parent
    script = gpu / "host/runtime/collision/capsule_extension.py"
    tree = ast.parse(script.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace = {"GPU_ROOT": gpu, "REPO_ROOT": script.parents[2], "Path": Path,
                 "hashlib": hashlib, "json": json, "re": re, "ROUTES": ((3, 5), (3, 6), (3, 7)),
                 "__file__": str(script)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(script), "exec"), namespace)
    return namespace


class NativeCapsuleSourceTests(unittest.TestCase):
    def fixture(self, destination, namespace):
        gpu = destination / "source-gpu"
        relatives = ["cuda/collision/native/native_source.cuh", "cuda/collision/native/native_kernel.cu",
                     "cuda/collision/native/native_support_reuse.cuh",
                     "host/runtime/collision/native_originals/engine_collision_box.c.txt",
                     "host/runtime/collision/native.py"]
        for name in relatives:
            target = gpu / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(namespace["GPU_ROOT"] / name, target)
        namespace["GPU_ROOT"] = gpu
        return gpu

    def test_actual_sources_copy_header_exactly_and_bind_every_byte(self):
        namespace = generator_namespace()
        with tempfile.TemporaryDirectory() as directory:
            target, identity = namespace["generate_sources"](directory)
            header = namespace["GPU_ROOT"] / "cuda/collision/native/native_support_reuse.cuh"
            self.assertEqual((target / header.name).read_bytes(), header.read_bytes())
            self.assertIn(str(header.resolve()), identity["accepted_inputs"])
            self.assertEqual(identity["generated_files"][header.name], hashlib.sha256(header.read_bytes()).hexdigest())
            self.assertEqual(set(identity["generated_files"]), {"native_source.cuh", "native_kernel.cu", header.name})
            again, repeated = namespace["generate_sources"](directory)
            self.assertEqual(again, target)
            self.assertEqual(repeated, identity)
            (target / header.name).write_bytes(b"corrupt copied header")
            with self.assertRaisesRegex(ValueError, "changed content"):
                namespace["generate_sources"](directory)

    def test_header_drift_builds_new_directory_without_overwriting_old(self):
        namespace = generator_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = self.fixture(root, namespace)
            header = gpu / "cuda/collision/native/native_support_reuse.cuh"
            old_raw = header.read_bytes()
            old, old_identity = namespace["generate_sources"](root / "generated")
            header.write_bytes(old_raw + b"\n// source drift fixture\n")
            new, new_identity = namespace["generate_sources"](root / "generated")
            self.assertNotEqual(old, new)
            self.assertNotEqual(old_identity["candidate_source_sha256"], new_identity["candidate_source_sha256"])
            self.assertEqual((old / header.name).read_bytes(), old_raw)
            self.assertEqual((new / header.name).read_bytes(), header.read_bytes())

    def test_nested_and_cyclic_headers_preserve_exact_bytes(self):
        namespace = generator_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = self.fixture(root, namespace)
            native = gpu / "cuda/collision/native"
            header = native / "native_support_reuse.cuh"
            header.write_bytes(header.read_bytes() + b'\n#include "nested/a.cuh"\n')
            (native / "nested").mkdir()
            (native / "nested/a.cuh").write_bytes(b'#include "b.cuh"\r\n// A\r\n')
            (native / "nested/b.cuh").write_bytes(b'#include "a.cuh"\n// B\n')
            target, identity = namespace["generate_sources"](root / "generated")
            for name in ("nested/a.cuh", "nested/b.cuh"):
                self.assertEqual((target / name).read_bytes(), (native / name).read_bytes())
                self.assertIn(name, identity["generated_files"])
                self.assertIn(str((native / name).resolve()), identity["accepted_inputs"])

    def test_missing_and_external_relative_headers_are_explicit_errors(self):
        for include, error in (("missing.cuh", FileNotFoundError), ("../external.cuh", ValueError)):
            namespace = generator_namespace()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                gpu = self.fixture(root, namespace)
                native = gpu / "cuda/collision/native"
                header = native / "native_support_reuse.cuh"
                header.write_bytes(header.read_bytes() + ('\n#include "' + include + '"\n').encode())
                if include.startswith(".."):
                    (native.parent / "external.cuh").write_bytes(b"// external source\n")
                with self.assertRaises(error) as context:
                    namespace["generate_sources"](root / "generated")
                self.assertIn(include, str(context.exception))

    def test_candidate_and_metadata_symlinks_cannot_escape_destination(self):
        namespace = generator_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, identity = namespace["generate_sources"](root / "normal")
            outside = root / "outside"
            outside.mkdir()
            generated = root / "symlink-candidate"
            generated.mkdir()
            (generated / original.name).symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "root escapes destination"):
                namespace["generate_sources"](generated)
            self.assertEqual(list(outside.iterdir()), [])
            manifest_root = root / "symlink-manifest"
            manifest_root.mkdir()
            candidate = manifest_root / original.name
            candidate.mkdir()
            protected = outside / "unchanged.json"
            protected.write_bytes(b"protected")
            (candidate / "SOURCE_CANDIDATE.json").symlink_to(protected)
            with self.assertRaisesRegex(ValueError, "manifest escapes candidate"):
                namespace["generate_sources"](manifest_root)
            self.assertEqual(protected.read_bytes(), b"protected")

    def test_absolute_external_gpu_headers_remain_at_the_actual_source_location(self):
        namespace = generator_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = self.fixture(root, namespace)
            external = gpu / "cuda/common/external.cuh"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"// external source\n")
            native = gpu / "cuda/collision/native"
            header = native / "native_support_reuse.cuh"
            header.write_bytes(header.read_bytes() + ('\n#include "' + str(external) + '"\n').encode())
            target, identity = namespace["generate_sources"](root / "generated")
            self.assertIn(str(external.resolve()), identity["accepted_inputs"])
            self.assertNotIn("external.cuh", identity["generated_files"])
            self.assertEqual((target / header.name).read_bytes(), header.read_bytes())


if __name__ == "__main__":
    unittest.main()
