"""Ordinary offline native products and read-only compatibility with old caches.

No signing key or source-tree authority is required. Runtime selects products by
source specialization, compile options and architecture; the actual loader checks
the C ABI. Compiler dependency/toolchain validation belongs to offline builds.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import base64
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _selection(identity):
    """Machine-code selection, independent of the installed compiler's files."""
    result = json.loads(_canonical(identity))
    build = result.get('build', result)
    for name in ('toolchain', 'dependencies', 'archives'):
        build.pop(name, None)
    archive = build.get('archive')
    if isinstance(archive, dict):
        build['archive'] = archive['path']
    return result


_scope = ContextVar('libero_product_cache', default=None)
_persistence_required = ContextVar('libero_product_persistence_required', default=False)


@contextmanager
def require_product_persistence():
    token = _persistence_required.set(True)
    try:
        yield
    finally:
        _persistence_required.reset(token)


@contextmanager
def product_cache_scope(cache_root, *, max_bytes=None, max_entries=None, lock_timeout=1800):
    cache = ProductCache(cache_root,
        max_bytes=int(os.environ.get('LIBERO_GPU_PRODUCT_CACHE_MAX_BYTES', 4*1024**3)) if max_bytes is None else max_bytes,
        max_entries=int(os.environ.get('LIBERO_GPU_PRODUCT_CACHE_MAX_ENTRIES', 4096)) if max_entries is None else max_entries,
        lock_timeout=lock_timeout)
    token = _scope.set(cache)
    try:
        yield cache
    finally:
        _scope.reset(token)


def current_product_cache():
    return _scope.get()


def compiler_environment_record(environment=None):
    """Offline build provenance; unrelated worker variables are not inputs."""
    names = {'PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH',
        'LD_LIBRARY_PATH', 'LD_PRELOAD', 'COMPILER_PATH', 'GCC_EXEC_PREFIX',
        'SOURCE_DATE_EPOCH', 'CUDA_HOME', 'CUDA_PATH', 'CUDAHOSTCXX', 'CUDACXX',
        'LIBERO_NVCC', 'LIBERO_MATHDX_ROOT', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ'}
    runtime_names = {'CUDA_CACHE_PATH', 'CUDA_CACHE_MAXSIZE', 'CUDA_CACHE_DISABLE'}
    selected = {k: v for k, v in (os.environ if environment is None else environment).items()
                if k not in runtime_names and (k in names or k.startswith(('NVCC_', 'CUDA_', 'LC_')))}
    return {'schema': 'compiler-input-environment-v1', 'sha256': _digest(_canonical(selected))}


@contextmanager
def _lock(path, timeout):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout:
                    raise TimeoutError('native product build lock timeout: ' + str(path))
                time.sleep(.05)
        yield
    finally:
        os.close(fd)


class ProductCache:
    """Offline publication owns writes; loading neither registers nor copies code."""
    def __init__(self, cache_root, *, max_bytes=4*1024**3, max_entries=4096, lock_timeout=1800):
        from .compilation import compilation_allowed, MissingCompiledProduct
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.root = self.cache_root/'native-v1'
        self.legacy_root = self.cache_root/'signed-v1'
        if type(max_bytes) is not int or max_bytes < 1 or type(max_entries) is not int or max_entries < 1:
            raise ValueError('positive native product cache bounds required')
        self.max_bytes, self.max_entries, self.lock_timeout = max_bytes, max_entries, lock_timeout
        if compilation_allowed():
            self.root.mkdir(parents=True, exist_ok=True)
        elif not (self.root.is_dir() or self.legacy_root.is_dir()):
            raise MissingCompiledProduct('Native products are absent; prepare offline.')

    def _find(self, identity, symbols, names, *, offline=False):
        key = _digest(_canonical({'identity': _selection(identity), 'required_symbols': list(symbols)}))
        def candidates():
            yield self.root/key/'receipt.json'
            # Only a miss in the indexed layout needs a legacy-directory scan.
            yield from sorted(self.legacy_root.glob('*/receipt.json'))
        for path in candidates():
            try:
                raw = json.loads(path.read_bytes())
                receipt = raw.get('receipt', raw)  # old signed envelope
                if (receipt.get('completed') is not True or
                        receipt.get('required_symbols') != list(symbols) or
                        _selection(receipt['identity']) != _selection(identity) or
                        (offline and receipt['identity'] != identity)):
                    continue
                legacy = receipt.get('schema') == 'authorized-native-product-v1'
                if legacy:
                    files = {'product.so': dict(sha256=receipt['binary_sha256'], bytes=receipt['binary_bytes'])}
                elif receipt.get('schema') == 'native-products-v1':
                    files = receipt['files']
                    if set(files) != set(names):
                        continue
                else:
                    continue
                if not all((path.parent/name).is_file() for name in files):
                    continue
                if offline and any(_digest((path.parent/name).read_bytes()) != item['sha256']
                                   for name, item in files.items()):
                    continue
                return path.parent, receipt, legacy
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def find_native(self, source, bridges, flags, architecture, archive, required_symbols):
        identity = dict(schema='compiler-dependency-build-v1', source_sha256=_digest(source.encode()),
            bridges_sha256=[_digest(text.encode()) for text in bridges], flags=list(flags),
            architecture=architecture, archive=archive, params_launcher_abi='included-generated-source',
            policy=dict(fma=True, precise_div=True, precise_sqrt=True, ftz=False))
        found = self._find(identity, required_symbols, ('product.so',))
        if found is None:
            return None
        folder, receipt, _ = found
        return folder/'product.so', [folder/'inputs'/('unit%d.cu' % i) for i in range(1+len(bridges))], receipt

    def _publish(self, folder, key, identity, symbols, artifacts):
        receipt = dict(schema='native-products-v1', build_key=key, identity=identity,
            required_symbols=list(symbols), completed=True,
            files={name: dict(bytes=len(raw), sha256=_digest(raw)) for name, raw in artifacts.items()})
        if set(artifacts) == {'product.so'}:
            # Retained diagnostic consumers use these ordinary provenance fields.
            receipt.update(binary_sha256=receipt['files']['product.so']['sha256'],
                           binary_bytes=receipt['files']['product.so']['bytes'])
        encoded = _canonical(receipt)
        with _lock(self.root/'catalog.lock', self.lock_timeout):
            # Budget only at offline publication. Include legacy and interrupted
            # files, but never scan or prune the cache when creating an env.
            files = [p for base in (self.root, self.legacy_root) for p in base.rglob('*') if p.is_file()]
            occupied = sum(p.stat().st_size for p in files)
            replaced = sum((folder/name).stat().st_size for name in (*artifacts, 'receipt.json')
                           if (folder/name).is_file())
            entries = {p.parent for p in files if p.name == 'receipt.json'}
            if (occupied-replaced+sum(map(len, artifacts.values()))+len(encoded) > self.max_bytes or
                    len(entries | {folder}) > self.max_entries):
                if _persistence_required.get():
                    raise OSError(errno.ENOSPC, 'Offline native product cannot be persisted: product cache budget exceeded')
                return receipt, False
            temporary = []
            try:
                for name, raw in {**artifacts, 'receipt.json': encoded}.items():
                    path = folder/('.' + uuid.uuid4().hex + '.tmp')
                    temporary.append((path, folder/name))
                    path.write_bytes(raw)
                # A failed replacement must not leave an old completion marker
                # attached to a partially replaced set of files.
                (folder/'receipt.json').unlink(missing_ok=True)
                for source, target in temporary:
                    os.replace(source, target)
            finally:
                for source, _ in temporary:
                    source.unlink(missing_ok=True)
        return receipt, True

    def obtain_artifacts(self, identity, required_symbols, *, names, build, validate, exports, _single=False):
        from .compilation import compilation_allowed, MissingCompiledProduct
        names = tuple(sorted(names))
        if not names or any(type(n) is not str or Path(n).name != n or n in ('.', '..', '') for n in names):
            raise ValueError('native artifact names must be safe basenames')
        identity = json.loads(_canonical(identity))
        if not _single:
            identity = dict(schema='native-multi-product-v1', build=identity, artifact_names=list(names))
        symbols = tuple(required_symbols)
        key = _digest(_canonical({'identity': _selection(identity), 'required_symbols': list(symbols)}))
        offline = compilation_allowed()
        def load(found):
            folder, receipt, legacy = found
            if legacy and not _single:
                bundle = json.loads((folder/'product.so').read_bytes())
                if bundle.get('schema') != 'native-artifact-bundle-v1' or set(bundle['artifacts']) != set(names):
                    raise ValueError('incomplete legacy native artifact bundle')
                artifacts = {n: base64.b64decode(bundle['artifacts'][n], validate=True) for n in names}
            else:
                artifacts = {n: (folder/n).read_bytes() for n in names}
            exports(artifacts)
            return artifacts, receipt, dict(hit=True, persisted=True, key=folder.name, directory=str(folder))
        if not offline:
            found = self._find(identity, symbols, names)
            if found is None:
                raise MissingCompiledProduct('No completed native product matches the requested source, options and architecture; prepare offline.')
            return load(found)
        folder = self.root/key
        folder.mkdir(exist_ok=True)
        with _lock(folder/'build.lock', self.lock_timeout):
            validate()
            found = self._find(identity, symbols, names, offline=True)
            if found is not None:
                result = load(found)
                validate()
                return result
            artifacts = build()
            if type(artifacts) is not dict or set(artifacts) != set(names) or any(type(v) is not bytes for v in artifacts.values()):
                raise ValueError('incomplete native build artifact set')
            exports(artifacts)
            validate()
            receipt, persisted = self._publish(folder, key, identity, symbols, artifacts)
            return artifacts, receipt, dict(hit=False, persisted=persisted, key=key, directory=str(folder))

    def obtain(self, identity, required_symbols, *, build, validate, exports):
        artifacts, receipt, status = self.obtain_artifacts(identity, required_symbols, names=('product.so',),
            build=lambda: {'product.so': build()}, validate=validate,
            exports=lambda files: exports(files['product.so']), _single=True)
        return artifacts['product.so'], receipt, status
