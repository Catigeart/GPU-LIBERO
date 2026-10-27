"""Precision namespaces and shared CMake products; resolution never writes files."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

import libero.gpu.precision as scalar
from libero.gpu.host.runtime.core.product_paths import cache_root

_MARKER = 'PROJECT_NAMESPACE.json'


@dataclass(frozen=True)
class ProjectPaths:
    base: Path
    precision: Path
    cmake: Path


def _namespace_bits(root):
    marker = root / _MARKER
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record != dict(schema='libero-project-namespace-v1', real_bits=32, base_relative='..'):
            raise ValueError('invalid project precision namespace: ' + str(marker))
        return 32
    # Existing caches predate the explicit namespace marker. Their build
    # receipt identifies the namespace; directory spelling is not evidence.
    legacy = root / 'AOT_BUILD.json'
    if legacy.is_file():
        return json.loads(legacy.read_text()).get('real_bits', 64)
    return None


def project_paths(root=None, *, precision=None):
    bits = scalar.bits if precision is None else scalar._normalize(precision)
    root = Path(root).expanduser().resolve() if root is not None else cache_root(create=False) / 'project'
    namespace = _namespace_bits(root)
    if namespace == 32:
        if bits != 32:
            raise ValueError('FP32 namespace cannot be opened by an FP64 process')
        base = root.parent
    else:
        base = root
    selected = base / 'fp32' if bits == 32 else base
    return ProjectPaths(base, selected, base / 'cmake-control')


def prepare_precision_project(root=None):
    """Prepare the precision namespace and its view of shared scene data."""
    paths = project_paths(root)
    paths.precision.mkdir(parents=True, exist_ok=True)
    if scalar.bits == 32:
        marker = paths.precision / _MARKER
        if not marker.exists():
            marker.write_text(json.dumps(dict(schema='libero-project-namespace-v1',
                real_bits=32, base_relative='..'), indent=2) + '\n')
        source, target = paths.base / 'TASK_INSTANCES.json', paths.precision / 'TASK_INSTANCES.json'
        if source.is_file():
            catalog = json.loads(source.read_text())
            for row in catalog['tasks']:
                for name in ('directory', 'constraint_package'):
                    if name in row and not Path(row[name]).is_absolute():
                        row[name] = os.path.relpath(paths.base / row[name], paths.precision)
            data = json.dumps(catalog, indent=2, sort_keys=True) + '\n'
            if not target.is_file() or target.read_text() != data:
                target.write_text(data)
    return paths


def cmake_root(root):
    """Accept a CMake directory, including one below a recorded namespace."""
    root = Path(root).expanduser().resolve()
    if root.name == 'cmake-control':
        return project_paths(root.parent).cmake
    return root


def cmake_directory(root, arch, *, fast_compile=False):
    return project_paths(root,precision='fp32').cmake / f"sm{arch}-{'dev' if fast_compile else 'release'}"


def relative_record(value, root):
    """Store preparation-internal paths relative to their movable project root."""
    if isinstance(value, dict):
        return {key:relative_record(item, root) for key,item in value.items()}
    if isinstance(value, (tuple, list)):
        return [relative_record(item, root) for item in value]
    if isinstance(value, str) and (value == str(root) or value.startswith(str(root) + '/')):
        return str(Path(value).relative_to(root))
    return value
