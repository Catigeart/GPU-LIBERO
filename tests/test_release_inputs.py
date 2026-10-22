"""Release staging must retain native build inputs, without running CMake."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ReleaseInputTests(unittest.TestCase):
    def load_builder(self):
        spec = importlib.util.spec_from_file_location("release_builder", ROOT / "tools/release/build.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        return builder

    def test_all_maintained_cmake_inputs_are_staged(self):
        builder = self.load_builder()
        selected = set(builder.selected_files())
        self.assertIn(ROOT / "CMakeLists.txt", selected)
        self.assertIn(ROOT / "libero/libero/gpu/CMakeLists.txt", selected)
        for path in (ROOT / "libero/libero/gpu/cmake").rglob("*"):
            if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
                self.assertIn(path, selected)

    def test_incomplete_allowlist_is_rejected_before_packaging(self):
        builder = self.load_builder()
        policy = dict(builder.POLICY, source_extensions=[
            suffix for suffix in builder.POLICY["source_extensions"] if suffix != ".cmake"])
        with patch.object(builder, "POLICY", policy):
            with self.assertRaisesRegex(RuntimeError, "omits CMake build inputs"):
                builder.selected_files()


if __name__ == "__main__":
    unittest.main()
