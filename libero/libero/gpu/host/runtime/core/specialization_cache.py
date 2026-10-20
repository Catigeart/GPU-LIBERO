"""Persist identical generated source, while every intake keeps a fresh owner."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def _manifest_key(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def index_packages(root, packages=None):
    """Publish the complete explicit instance table at the offline boundary."""
    root = Path(root)
    entries = {}
    selected = root.glob('shape_s_*') if packages is None else map(Path, packages)
    for package in sorted(set(selected)):
        model = package / 'MODEL_MANIFEST.json'
        generated = package / 'SOURCE_GENERATION.json'
        if not model.is_file() or not generated.is_file():
            continue
        report = json.loads(generated.read_text())
        # A historical package can share the same model manifest while its
        # template version is obsolete. Never let scan order replace a current
        # entry with that package. Runtime intake still validates every pin.
        pins = [pin for group in ('template_pins', 'generated_pins')
                for pin in report.get(group, {}).values()]
        current = bool(pins) and all(Path(pin['path']).is_file() and
            hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256'] for pin in pins)
        if not current:
            if packages is not None:
                raise ValueError('explicit specialization has stale source pins: ' + str(package))
            continue
        manifest = json.loads(model.read_text())
        key = _manifest_key(manifest)
        entry = dict(package=package.name,
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in package.iterdir() if p.is_file()})
        if key in entries and entries[key]['package'] != entry['package']:
            raise ValueError('multiple current specializations for one manifest: ' + key)
        entries[key] = entry
    (root / 'INSTANCES.json').write_text(json.dumps(entries, indent=2, sort_keys=True)+'\n')
    return entries


def generated_package(manifest, static, generator, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    from libero.gpu.host.build.compilation import compilation_allowed, MissingCompiledProduct
    if not compilation_allowed():
        index = root / 'INSTANCES.json'
        entries = json.loads(index.read_text()) if index.exists() else {}
        entry = entries.get(_manifest_key(manifest))
        if entry is None:
            raise MissingCompiledProduct('Constraint instance is absent from the offline build catalog')
        package = root / entry['package']
        for name, digest in entry['files'].items():
            if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
                raise ValueError('prebuilt specialization source differs: ' + str(package / name))
        return package, json.loads((package / 'SOURCE_GENERATION.json').read_text()), True
    with tempfile.TemporaryDirectory(prefix='.generate-', dir=root) as staging:
        candidate = Path(staging) / 'package'
        report = generator(manifest, static, candidate)
        identity = dict(manifest=manifest,
                        templates={k: v['sha256'] for k, v in report['template_pins'].items()},
                        generated={k: v['sha256'] for k, v in report['generated_pins'].items()})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        package = root / ('shape_s_' + key)
        for name, entry in report['generated_pins'].items():
            entry['path'] = str(package / (name + '.py'))
        expected_report = json.dumps(report, indent=2, sort_keys=True) + '\n'
        (candidate / 'SOURCE_GENERATION.json').write_text(expected_report)
        if package.exists():
            expected = {p.name: p.read_bytes() for p in candidate.iterdir() if p.is_file()}
            actual = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}
            if actual != expected:
                raise ValueError('cached specialization source differs: ' + str(package))
            hit = True
        else:
            shutil.move(str(candidate), str(package))
            hit = False
        return package, report, hit
