"""Precision namespaces and shared CMake products; resolution never writes files."""
from dataclasses import dataclass
import json
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


def project_paths(root=None):
    root = Path(root).expanduser().resolve() if root is not None else cache_root(create=False) / 'project'
    namespace = _namespace_bits(root)
    if namespace == 32:
        if scalar.bits != 32:
            raise ValueError('FP32 namespace cannot be opened by an FP64 process')
        base = root.parent
    else:
        base = root
    selected = base / 'fp32' if scalar.bits == 32 else base
    return ProjectPaths(base, selected, base / 'cmake-control')


def prepare_precision_project(root=None):
    """Explicit offline preparation; refuse a stale copied task catalogue."""
    paths = project_paths(root)
    paths.precision.mkdir(parents=True, exist_ok=True)
    if scalar.bits == 32:
        marker = paths.precision / _MARKER
        if not marker.exists():
            marker.write_text(json.dumps(dict(schema='libero-project-namespace-v1',
                real_bits=32, base_relative='..'), indent=2) + '\n')
        source, target = paths.base / 'TASK_INSTANCES.json', paths.precision / 'TASK_INSTANCES.json'
        if source.is_file():
            data = source.read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError('precision task catalogue differs from the base project; prepare the intended catalogue explicitly')
            if not target.exists():
                target.write_bytes(data)
    return paths


def cmake_root(root):
    """Accept a CMake directory, including one below a recorded namespace."""
    root = Path(root).expanduser().resolve()
    if root.name == 'cmake-control':
        return project_paths(root.parent).cmake
    return root
