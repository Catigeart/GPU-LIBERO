"""Build and check one wheel and one sdist from the current source tree."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((Path(__file__).with_name("policy.json")).read_text())
LICENSES = ("LICENSE", "licenses/LIBERO-MIT.txt", "licenses/Apache-2.0.txt", "licenses/libccd-BSD-3-Clause.txt")


def selected_files():
    selected = set()
    for name in POLICY["root_files"]:
        p = ROOT / name
        if not p.is_file():
            raise RuntimeError("Missing root source file: " + name)
        selected.add(p)
    for name in POLICY["roots"]:
        for p in (ROOT / name).rglob("*"):
            rel = p.relative_to(ROOT)
            if (p.is_file() and not any(part in POLICY["excluded_directories"] or part.endswith(".egg-info") for part in rel.parts)
                    and p.name not in POLICY["excluded_filenames"] and rel.as_posix() not in POLICY["excluded_paths"]
                    and p.suffix.lower() in POLICY["source_extensions"]):
                selected.add(p)
    # MANIFEST.in cannot recover files absent from this staging tree. Check
    # the build entry points independently of the publication allowlist.
    build_inputs = {ROOT / "CMakeLists.txt", ROOT / "libero/libero/gpu/CMakeLists.txt"}
    build_inputs.update((ROOT / "libero/libero/gpu/cmake").rglob("*.cmake"))
    build_inputs.update((ROOT / "libero/libero/gpu/cmake").rglob("CMakeLists.txt"))
    missing = sorted(str(p.relative_to(ROOT)) for p in build_inputs - selected)
    if missing:
        raise RuntimeError("Release policy omits CMake build inputs: " + repr(missing))
    return sorted(selected)


def version():
    tree = ast.parse((ROOT / "setup.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setup")
    return ast.literal_eval(next(k.value for k in call.keywords if k.arg == "version"))


def verify(wheel, sdist, source):
    with zipfile.ZipFile(wheel) as archive:
        payload = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    with tarfile.open(sdist) as archive:
        source_payload = {str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0])): archive.extractfile(m).read()
                          for m in archive.getmembers() if m.isfile()}
    for p in source:
        relative = p.relative_to(ROOT).as_posix()
        if relative not in (".gitignore", ".gitattributes", ".github/workflows/ci.yml", ".github/workflows/release.yml"):
            if source_payload.get(relative) != p.read_bytes():
                raise RuntimeError("Source archive omitted or changed: " + relative)
        if relative.startswith(("libero/", "scripts/", "benchmark_scripts/", "templates/")):
            installed = ("libero/gpu/" + relative[len("libero/libero/gpu/"):] if relative.startswith("libero/libero/gpu/") else relative)
            if payload.get(installed) != p.read_bytes():
                raise RuntimeError("Wheel omitted or changed: " + installed)
    for relative in LICENSES:
        matches = [n for n in payload if n.endswith("/" + relative)]
        if not matches or not any(payload[n] == (ROOT / relative).read_bytes() for n in matches):
            raise RuntimeError("Wheel omitted license: " + relative)
    for names in (payload, source_payload):
        for name in names:
            parts = PurePosixPath(name).parts
            if (any(p in POLICY["excluded_directories"] for p in parts)
                    or PurePosixPath(name).name in POLICY["excluded_filenames"]
                    or PurePosixPath(name).suffix.lower() in {".pyc", ".pyo", ".so", ".dll", ".a", ".o", ".cubin", ".ptx", ".npz", ".npy"}):
                raise RuntimeError("Non-source publication content: " + name)
    return len(payload), len(source_payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", help="Optional version tag; must equal v<setup.py version>")
    args = parser.parse_args()
    current_version = version()
    if args.tag is not None and args.tag != "v" + current_version:
        parser.error("The tag must match setup.py version: v" + current_version)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")) or list(output.glob("*.tar.gz")):
        parser.error("Use an empty artifact output directory; existing releases are not overwritten")
    source = selected_files()
    with tempfile.TemporaryDirectory(prefix="flash-libero-packaging-") as temporary:
        staging = Path(temporary) / "source"
        artifacts = Path(temporary) / "artifacts"
        artifacts.mkdir()
        for p in source:
            target = staging / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        # PEP 517 backend hooks use only the installed setuptools implementation.
        worker = ("import setuptools.build_meta as backend; "
                  "backend.build_sdist(" + repr(str(artifacts)) + "); "
                  "backend.build_wheel(" + repr(str(artifacts)) + ")")
        log = output / "backend-build.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-c", worker], cwd=staging, check=True, stdout=stream, stderr=subprocess.STDOUT)
        wheels, sdists = list(artifacts.glob("*.whl")), list(artifacts.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError("Expected exactly one wheel and one sdist")
        counts = verify(wheels[0], sdists[0], source)
        copied = []
        for artifact in (wheels[0], sdists[0]):
            target = output / artifact.name
            shutil.copyfile(artifact, target)
            copied.append({"name": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size})
        report = {"version": current_version, "source_files": len(source), "wheel_entries": counts[0], "sdist_entries": counts[1],
                  "artifacts": copied, "CUDA_compiled": False, "repeat_generation": False}
        (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
