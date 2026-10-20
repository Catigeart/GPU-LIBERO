"""Unselected diagnostic jar-only S04 candidate, with retained generic math."""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import types

ORIGINAL_S04_SHA = "fb67805419a08cd96f6acaafb883dbdc0b75fb56b82b4509947322270287354f"
FMA_POLICY_PROVENANCE = {'previous_no_fma_source_sha256': '433b9e216bb19e6511765bfbb4be67a0a924849ed2e48e50bed14bf89b9844fe', 'change': 'module fuse_fp False -> True; arithmetic AST unchanged'}
JAR_CAP = scalar.jar_cap


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def s04_candidate_source():
    path = (GPU_ROOT / 'reference/constraints/constraint_update/kernel.py')
    raw = path.read_bytes()
    if _sha(raw) != ORIGINAL_S04_SHA:
        raise ValueError("original S04 source identity changed")
    text = raw.decode("utf-8")
    before = "cap=scalar.wp_real(10000.0);d_cap=scalar.wp_real(2.0e16)"
    after = before + ";jar_cap=scalar.wp_real(scalar.jar_cap);jar_lower=_flip_sign(jar_cap)"
    predicate = "jar[rb+r]>cap or jar[rb+r]<lower"
    replacement = "jar[rb+r]>jar_cap or jar[rb+r]<jar_lower"
    if text.count(before) != 1 or text.count(predicate) != 1:
        raise ValueError("exact original jar guard required")
    generated = text.replace(before, after).replace(predicate, replacement)
    # Undo the declared two guard-only differences to prove every other AST.
    restored = generated.replace(after, before).replace(replacement, predicate)
    if ast.dump(ast.parse(restored)) != ast.dump(ast.parse(text)):
        raise ValueError("S04 math or another guard changed")
    return generated.encode(), dict(source_path=str(path), source_sha256=_sha(raw), fma_policy_provenance=FMA_POLICY_PROVENANCE,
        generated_sha256=_sha(generated.encode()), jar_cap_binary64_hex=JAR_CAP.hex(),
        other_guards_unchanged=True, acceptance="unselected-diagnostic-source-candidate")


def candidate_sources(nv, nm):
    from libero.gpu.host.runtime.newton.specialization import candidate_sources as original
    outputs, base = original(nv, nm)
    outputs = dict(outputs)
    solver = outputs["solver"].decode()
    before = "from libero.gpu.reference.constraints.constraint_update import kernel as s04"
    after = "from . import s04"
    if solver.count(before) != 1:
        raise ValueError("original generated solver S04 import required")
    outputs["solver"] = solver.replace(before, after).encode()
    outputs["s04"], guard = s04_candidate_source()
    if ast.dump(ast.parse(outputs["solver"].decode().replace(after, before))) != ast.dump(ast.parse(solver)):
        raise ValueError("generated solver math changed")
    files = dict(base["files"])
    files["solver"] = dict(files["solver"], generated_sha256=_sha(outputs["solver"]))
    files["s04_jar_only"] = guard
    identity = _sha(json.dumps(dict(nv=nv, nM=nm, files=files), sort_keys=True).encode())
    manifest = dict(base, candidate_id=identity, files=files,
        acceptance="unselected-diagnostic-jar-only-candidate",
        source_generator_sha256=_sha(Path(__file__).read_bytes()),
        original_candidate_id=base["candidate_id"], dimensions_are_actual_native=True)
    return outputs, manifest


def load_candidate(nv, nm, build_root):
    from libero.gpu.host.runtime.core.product_paths import current_build_project
    from libero.gpu.host.build.aot import offline_newton_source
    project = current_build_project()
    prepared = offline_newton_source(project['root'], nv, nm) if project is not None else None
    if prepared is None:
        sources, manifest = candidate_sources(nv, nm)
    else:
        destination, manifest = prepared
        sources = {}
    digest = manifest["candidate_id"]
    if prepared is None:
        destination = Path(build_root).resolve() / ("jar_nv%d_nm%d_%s" % (nv, nm, digest[:16]))
    destination.mkdir(parents=True, exist_ok=True)
    products = dict(sources)
    if prepared is None:
        products["candidate"] = (json.dumps(manifest, indent=2, sort_keys=True) + chr(10)).encode()
    for name, raw in products.items():
        path = destination / (name + (".json" if name == "candidate" else ".py"))
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(("candidate source cache drift", str(path)))
        else:
            with path.open("xb") as file:
                file.write(raw)
    package_name = "_libero_newton_jar_candidate_" + digest
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(destination)]
        sys.modules[package_name] = package
        for name in ("s04", "kernels", "s05", "s06", "workspace", "solver"):
            fullname = package_name + "." + name
            spec = importlib.util.spec_from_file_location(fullname, destination / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            setattr(package, name, module)
            spec.loader.exec_module(module)
    return sys.modules[package_name], manifest
