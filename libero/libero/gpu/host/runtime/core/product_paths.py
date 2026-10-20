"""User-owned cold build locations and the tested external MathDx SDK."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import uuid

_CURRENT = ContextVar('libero_gpu_product_build', default=None)
_SDK = ContextVar('libero_gpu_product_sdk', default=None)
_PROJECT = ContextVar('libero_gpu_build_project', default=None)
_MATHDX_URL = ('https://developer.nvidia.com/downloads/compute/cublasdx/redist/'
               'cublasdx/cuda12/nvidia-mathdx-25.12.1-cuda12.tar.gz')
_MATHDX_SHA = 'd905fb32e4baafe7f6cd3ad94ac194d86dcf463481f37221da972b2784fa73ac'


def cache_root(*, create=True):
    explicit = os.environ.get('LIBERO_GPU_CACHE_DIR')
    root = Path(explicit).expanduser() if explicit else Path(
        os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'libero-gpu'
    root = root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def current_build_project():
    return _PROJECT.get()


def record_build(kind, identity, commands=()):
    event = dict(kind=kind, identity=str(identity), commands_executed=[list(c) for c in commands],
                 cache_hit=not bool(commands))
    from libero.gpu.host.build.compilation import record_event
    record_event(event)
    project = _PROJECT.get()
    if project is not None:
        project['events'].append(event)


def record_module_binding(target, execution):
    """Keep process-local CUDA addresses outside the shared compiler products."""
    import json
    directory = build_directory('module-binding') if _PROJECT.get() is not None else Path(target)
    observation = directory / 'MODULE_IDENTITY_OBSERVATION.json'
    with observation.open('x') as stream:
        json.dump(execution, stream, sort_keys=True, indent=2)
    return str(observation)


def collision_build_root(root, module, gpu_root):
    """Different generated source paths retain distinct exact compiler argv."""
    root, module, gpu_root = Path(root), Path(module).resolve(), Path(gpu_root).resolve()
    if _PROJECT.get() is not None and not module.is_relative_to(gpu_root):
        root = root / ('generated-' + hashlib.sha256(str(module).encode()).hexdigest()[:24])
    return root


def reuse_newton_binary(target, identity, source_sha256, architecture):
    """Reuse only a completed, byte-checked matching diagnostic Newton product."""
    if _PROJECT.get() is None or target.exists():
        return None
    import json
    candidates = set()
    # Warp publishes and removes temporary directories concurrently. Only scan
    # native products, and tolerate directories disappearing during traversal.
    for root in (cache_root() / 'builds', _PROJECT.get()['root'] / 'cuda'):
        for directory, _, files in os.walk(root):
            if target.name in files:
                candidates.add(Path(directory) / target.name)
    for candidate in sorted(candidates):
        if candidate == target:
            continue
        try:
            metadata = candidate.with_suffix('.json')
            manifest = json.loads((metadata if metadata.exists() else candidate.parent / 'MANIFEST.json').read_text())
            flags = manifest['command']
            raw = candidate.read_bytes()
            matches = (manifest['sha256'] == identity and
                       manifest['source_sha256'] == source_sha256 and
                       manifest['library_sha256'] == hashlib.sha256(raw).hexdigest() and
                       '-arch=sm_' + str(architecture) in flags)
        except (OSError, ValueError, KeyError):
            continue
        if matches:
            shutil.copyfile(candidate, target)
            target.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
            return str(candidate)
    return None


@contextmanager
def build_project(directory=None, *, lock_key=None):
    """One persistent compilation bank, with fresh runtime owners on every use."""
    previous = _PROJECT.get()
    if previous is not None:
        yield previous
        return
    import fcntl
    root = Path(directory).expanduser().resolve() if directory is not None else cache_root() / 'project'
    root.mkdir(parents=True, exist_ok=True)
    lock_name = '.build.lock' if lock_key is None else '.build-' + hashlib.sha256(str(lock_key).encode()).hexdigest() + '.lock'
    with (root / lock_name).open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        project = dict(root=root, events=[])
        token = _PROJECT.set(project)
        try:
            yield project
        finally:
            _PROJECT.reset(token)


@contextmanager
def build_unit_lock(directory):
    import fcntl
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.compile.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def serialized_build(function):
    """Serialize one compiler output directory across offline build workers."""
    from functools import wraps
    import inspect
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        with build_unit_lock(values['build_root']):
            return function(*args, **kwargs)
    return wrapped


def build_directory(label):
    """A unique mutable owner directory; no source/module authority is inherited."""
    current = _CURRENT.get()
    parent = current if current is not None else cache_root() / 'builds'
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=label + '-', dir=parent))


def current_build_root():
    current = _CURRENT.get()
    if current is not None:
        return current
    return build_directory('assembly')


def _sdk_valid(root):
    return all((root / relative).is_file() for relative in (
        'include/cublasdx.hpp', 'include/cusolverdx.hpp', 'lib/libcusolverdx.a'))


def mathdx_root(*, download=True):
    explicit = os.environ.get('LIBERO_MATHDX_ROOT')
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _sdk_valid(root):
            raise FileNotFoundError('Incomplete LIBERO_MATHDX_ROOT: ' + str(root))
        return root
    roots = []
    for entry in sys.path:
        roots.extend(Path(entry or '.').glob('nvidia/mathdx/25.12'))
    sdk = cache_root() / 'toolchains' / 'mathdx-25.12.1-cuda12'
    roots.extend(sdk.rglob('nvidia/mathdx/25.12') if sdk.exists() else ())
    for root in roots:
        if _sdk_valid(root):
            return root.resolve()
    if not download or os.environ.get('LIBERO_AUTO_DOWNLOAD', '1') == '0':
        raise FileNotFoundError('MathDx 25.12.1 CUDA12 is absent; set LIBERO_MATHDX_ROOT.')
    parent = sdk.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Cross-process serialization; failures never publish a completed SDK.
    import fcntl
    with (parent / '.mathdx.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for root in sdk.rglob('nvidia/mathdx/25.12') if sdk.exists() else ():
            if _sdk_valid(root):
                return root.resolve()
        archive = parent / ('mathdx-download-' + uuid.uuid4().hex + '.tar.gz')
        stage = Path(tempfile.mkdtemp(prefix='mathdx-stage-', dir=parent))
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(_MATHDX_URL, timeout=120) as response, archive.open('wb') as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != _MATHDX_SHA:
                raise ValueError('MathDx archive digest differs from the tested official SDK')
            with tarfile.open(archive, 'r:gz') as bundle:
                for member in bundle.getmembers():
                    destination = (stage / member.name).resolve()
                    if not destination.is_relative_to(stage.resolve()) or member.issym() or member.islnk():
                        raise ValueError('Unsafe MathDx archive member: ' + member.name)
                bundle.extractall(stage)
            root = next((p for p in stage.rglob('nvidia/mathdx/25.12') if _sdk_valid(p)), None)
            if root is None:
                raise ValueError('MathDx archive does not contain the required tested SDK layout')
            if sdk.exists():
                raise FileExistsError('Incomplete SDK directory preserved: ' + str(sdk))
            relative = root.relative_to(stage)
            stage.rename(sdk)
            return sdk / relative
        finally:
            archive.unlink(missing_ok=True)
            if stage.exists():
                # Only this invocation's verified newly-created temporary stage.
                shutil.rmtree(stage)


@contextmanager
def product_build():
    """Bind cold constructors to one private arena without changing process env."""
    previous = _CURRENT.get()
    if previous is not None:
        yield previous
        return
    root = build_directory('environment')
    token = _CURRENT.set(root)
    try:
        yield root
    finally:
        _CURRENT.reset(token)


@contextmanager
def toolchain_scope(sdk):
    sdk=Path(sdk).resolve()
    if not _sdk_valid(sdk):
        raise FileNotFoundError('Incomplete MathDx SDK: '+str(sdk))
    token=_SDK.set(sdk)
    try:yield
    finally:_SDK.reset(token)


def selected_mathdx_root(fallback):
    explicit=os.environ.get('LIBERO_MATHDX_ROOT')
    return Path(explicit).expanduser().resolve() if explicit else (_SDK.get() or Path(fallback))


def nvcc_path():
    explicit=os.environ.get('LIBERO_NVCC')
    selected=explicit or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    result=Path(selected).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError('CUDA nvcc is absent; install the CUDA Toolkit or set LIBERO_NVCC.')
    return str(result)
