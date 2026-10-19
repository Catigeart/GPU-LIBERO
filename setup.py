"""Package the original LIBERO API and canonical GPU implementation."""
from pathlib import Path
from setuptools import setup
from setuptools.command.editable_wheel import editable_wheel

ROOT = Path(__file__).resolve().parent


def packages_below(directory, prefix, excluded=()):
    """Discover Python namespace packages without treating assets as packages."""
    base = ROOT / directory
    names = {prefix}
    for source in base.rglob("*.py"):
        relative = source.relative_to(base)
        if any(part in {"__pycache__", ".git"} for part in relative.parts):
            continue
        if any(relative.parts[:len(parts)] == parts for parts in excluded):
            continue
        parent = relative.parent
        while parent != Path("."):
            names.add(prefix + "." + ".".join(parent.parts))
            parent = parent.parent
    return names


packages = {"libero"}
packages |= packages_below("libero/libero", "libero.libero", (("gpu",),))
packages |= packages_below("libero/lifelong", "libero.lifelong")
packages |= packages_below("libero/configs", "libero.configs")
packages |= packages_below("libero/libero/gpu", "libero.gpu")
packages |= packages_below("scripts", "scripts")
packages |= packages_below("benchmark_scripts", "benchmark_scripts")
packages |= packages_below("templates", "templates")


def source_data(directory, extensions):
    base = ROOT / directory
    return sorted(
        str(file.relative_to(base)).replace("\\", "/")
        for file in base.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
        and "__pycache__" not in file.parts
    )


class SourceEditableWheel(editable_wheel):
    """Use an ordinary .pth path for the existing source tree, never a finder."""
    def initialize_options(self):
        super().initialize_options()
        self.mode = "compat"


setup(
    name="Flash-LIBERO",
    version="0.2.0",
    description="Flash-LIBERO: GPU simulation with the original LIBERO public API",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Flash-LIBERO contributors and LIBERO authors",
    license="MIT (project code); Apache-2.0 and BSD-3-Clause (third-party components)",
    license_files=["LICENSE", "licenses/LIBERO-MIT.txt", "licenses/Apache-2.0.txt",
                   "licenses/libccd-BSD-3-Clause.txt"],
    python_requires=">=3.10",
    packages=sorted(packages),
    package_dir={"": ".", "libero.gpu": "libero/libero/gpu"},
    include_package_data=False,
    package_data={
        "libero.libero": source_data("libero/libero", {".xml", ".bddl"}) + ["upstream_resources.json"],
        "libero.configs": source_data("libero/configs", {".yaml", ".yml", ".json"}),
        "libero.gpu": source_data("libero/libero/gpu", {".cu", ".cuh", ".h", ".hpp", ".c", ".cpp", ".cmake", ".json", ".txt", ".md", ".ps1"}),
        "scripts": source_data("scripts", {".cu", ".cuh", ".h", ".hpp", ".c", ".cpp", ".xml", ".yaml", ".yml", ".json"}),
        "templates": ["scene_template.xml"],
    },
    install_requires=[
        "numpy>=1.24,<2", "scipy", "PyYAML>=6", "warp-lang>=1.17,<2",
        "mujoco==2.3.7", "robosuite==1.4.0", "bddl==1.0.1",
        "easydict", "h5py", "opencv-python", "gym==0.25.2", "cloudpickle",
        "torch", "hydra-core>=1.3,<2", "wandb", "transformers", "robomimic==0.2.0",
        "einops", "thop", "future", "matplotlib",
    ],
    cmdclass={"editable_wheel": SourceEditableWheel},
    entry_points={"console_scripts": [
        "lifelong.main=libero.lifelong.main:main",
        "lifelong.eval=libero.lifelong.evaluate:main",
        "libero.config_copy=scripts.config_copy:main",
        "libero.create_template=scripts.create_template:main",
    ]},
)
