"""Offline maintenance for the retired runtime storage-v1 catalog.

Environment construction/reset/close must not import this module.
Only explicitly registered files are collected.

flock, rather than a PID or an age heuristic, protects live users. The catalog
lock covers opening leases and removing entries, including their lock files.
Compiler projects and old unregistered directories are never collected here.
"""
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


GiB = 1024 ** 3


def _limit(name, default):
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(name + ' must be nonnegative')
    return value


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path, seen=None):
    # Do not follow symlinks out of a managed directory.
    if Path(path).is_symlink():
        return 0
    seen = set() if seen is None else seen
    total = 0
    for file in Path(path).rglob('*'):
        try:
            if file.is_symlink() or not file.is_file():
                continue
            info = file.stat()
        except FileNotFoundError:
            # Active runtimes can rename/remove temporary files while another
            # process inventories them. Admission rechecks after preparation.
            continue
        inode = info.st_dev, info.st_ino
        if inode not in seen:
            total += info.st_size
            seen.add(inode)
    return total


def catalog_sizes(catalog):
    # Read-only preflight links do not charge another full MJB to run storage.
    seen = set()
    order = {'models': 0, 'runs': 1}
    return {p: directory_bytes(p, seen) for p in sorted(catalog, key=lambda p: order[catalog[p]['kind']])}


class DiskLease:
    def __init__(self, cache, path, stream):
        self.cache, self.path, self._stream = cache, path, stream
        self._closed = False

    def close(self):
        if self._closed:
            return
        with self.cache._catalog() as catalog:
            entry = catalog.get(str(self.path))
            if entry is not None:
                entry['touched'] = time.time_ns()
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self.cache._collect(catalog)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DiskCache:
    """Total/category budgets apply to managed data, not offline build banks.

    Active data is never evicted. If it alone exceeds the budget, admission
    fails explicitly instead of letting a training run fill the filesystem.
    """
    def __init__(self, root=None, *, max_bytes=None, limits=None, min_free=None):
        if root is None:
            from ..runtime.core.product_paths import cache_root
            root = cache_root()
        self.root = Path(root).expanduser().resolve() / 'storage-v1'
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = _limit('LIBERO_GPU_DISK_MAX_BYTES', 64 * GiB) if max_bytes is None else max_bytes
        self.limits = limits if limits is not None else {
            'models': _limit('LIBERO_GPU_MODELS_MAX_BYTES', 32 * GiB),
            'runs': _limit('LIBERO_GPU_RUNS_MAX_BYTES', 2 * GiB)}
        self.min_free = _limit('LIBERO_GPU_DISK_MIN_FREE_BYTES', GiB) if min_free is None else min_free
        self.max_entries = _limit('LIBERO_GPU_DISK_MAX_ENTRIES', 4096)

    def _save(self, catalog):
        temporary = self.root / 'catalog.tmp'
        with temporary.open('w') as stream:
            stream.write(json.dumps(catalog, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.root / 'catalog.json')
        self._sync_directory(self.root)

    @staticmethod
    def _sync_directory(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _marker_record(self, entry):
        record = dict(schema='libero-disk-entry-v1', catalog=str(self.root), kind=entry['kind'])
        if 'ownership' in entry:
            record.update(schema='libero-disk-entry-v2', ownership=entry['ownership'])
        return record

    def _marker_matches(self, path, entry):
        marker = path / '.libero-storage.json'
        if marker.is_symlink() or not marker.is_file():
            return False
        try:
            return json.loads(marker.read_text()) == self._marker_record(entry)
        except (OSError, ValueError):
            return False

    @contextmanager
    def _catalog(self):
        with (self.root / 'catalog.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'catalog.json'
            catalog = json.loads(path.read_text()) if path.exists() else {}
            # Prepared scenes are fixed offline inputs, not runtime cache.
            # Retire old registrations without touching their files/markers.
            catalog = {p: row for p, row in catalog.items() if row['kind'] != 'scenes'}
            try:
                yield catalog
            finally:
                self._save(catalog)

    def _register(self, catalog, path, kind):
        if kind not in self.limits:
            raise ValueError('unknown managed storage category: ' + kind)
        if path.is_symlink():
            raise ValueError('refusing to register a symlink: ' + str(path))
        marker = path / '.libero-storage.json'
        previous = catalog.get(str(path), dict(kind=kind))
        if marker.exists() or marker.is_symlink():
            if previous['kind'] != kind or not self._marker_matches(path, previous):
                raise ValueError('managed directory belongs to another cache or has a damaged marker: ' + str(path))
            # Never truncate a valid marker when another reader registers it.
            catalog[str(path)] = dict(previous, touched=time.time_ns(), state='ready')
            catalog[str(path)].pop('collection_error', None)
            return
        if str(path) in catalog:
            raise ValueError('managed directory belongs to another cache: ' + str(path))
        entry = dict(kind=kind, touched=time.time_ns(), state='pending',
                     ownership=uuid.uuid4().hex, new_directory=not path.exists())
        catalog[str(path)] = entry
        # Durable intent precedes mkdir. A crash cannot leave an unregistered
        # allocation; unmarked recovery may remove only our small metadata.
        self._save(catalog)
        path.mkdir(parents=True, exist_ok=True)
        self._sync_directory(path.parent)
        temporary = path / ('.libero-storage-' + entry['ownership'] + '.tmp')
        try:
            with temporary.open('x') as stream:
                stream.write(json.dumps(self._marker_record(entry)) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker)
            self._sync_directory(path)
        finally:
            temporary.unlink(missing_ok=True)
        entry['state'] = 'ready'
        self._save(catalog)

    def _remove(self, catalog, path, entry):
        if path.is_symlink():
            entry['collection_error'] = 'unowned symlink; directory retained'
            return False
        if not path.exists():
            catalog.pop(str(path), None)
            return True
        if not path.is_dir():
            entry['collection_error'] = 'entry is not a directory; file retained'
            return False
        owned = self._marker_matches(path, entry)
        info = path.stat()
        removal_identity = [info.st_dev, info.st_ino]
        marker = path / '.libero-storage.json'
        resuming_removal = (entry.get('state') == 'removing' and
                            entry.get('removal_identity') == removal_identity and
                            not marker.exists() and not marker.is_symlink())
        owned = owned or resuming_removal
        if not owned:
            # Interrupted registration may predate the marker. The journal
            # authorizes only a newly allocated, otherwise empty directory.
            pending = entry.get('state') == 'pending' and 'ownership' in entry
            if pending and not entry['new_directory'] and not (path / '.libero-storage.json').exists():
                catalog.pop(str(path), None)  # Pre-existing files were never claimed.
                return True
            allowed = {'.lease', '.libero-storage-' + entry.get('ownership', '') + '.tmp'}
            metadata_only = pending and entry['new_directory'] and all(
                p.name in allowed and p.is_file() and not p.is_symlink() for p in path.iterdir())
            if not metadata_only:
                entry['collection_error'] = 'ownership marker missing/damaged/foreign; directory retained'
                return False
        lease_path = path / '.lease'
        if lease_path.is_symlink():
            entry['collection_error'] = 'unowned lease symlink; directory retained'
            return False
        fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            # rmtree may remove the ownership marker before a process dies.
            # Commit the checked directory identity while holding both locks;
            # a later collector can resume, but cannot delete a replacement.
            if entry.get('state') != 'removing':
                entry.update(state='removing', removal_identity=removal_identity)
                self._save(catalog)
            shutil.rmtree(path)
        catalog.pop(str(path))
        return True

    def _collect(self, catalog, *, extra=0, kind=None, require=False):
        # A dead process releases its run lease. Reap even below the budget.
        for name, entry in list(catalog.items()):
            incomplete = entry['kind'] == 'models' and not (Path(name) / 'model.mjb').exists()
            if entry.get('state') in ('pending', 'removing') or entry['kind'] == 'runs' or incomplete or not Path(name).exists():
                self._remove(catalog, Path(name), entry)
        sizes = catalog_sizes(catalog)

        def over():
            return (len(sizes) > self.max_entries or sum(sizes.values()) + extra > self.max_bytes or
                    any(sum(sizes[p] for p in sizes if catalog[p]['kind'] == k) +
                        (extra if kind == k else 0) > limit for k, limit in self.limits.items()) or
                    shutil.disk_usage(self.root).free - extra < self.min_free)

        for name in sorted(catalog, key=lambda p: catalog[p]['touched']):
            if not over():
                break
            if self._remove(catalog, Path(name), catalog[name]):
                sizes.pop(name, None)
        if require and over():
            raise OSError(errno.ENOSPC, 'GPU-LIBERO managed disk budget/free-space floor exceeded; '
                          'live leases are protected. Configure LIBERO_GPU_DISK_MAX_BYTES/category limits, '
                          'or close unused environments. Unmanaged compiler/legacy caches need separate maintenance.')

    def collect(self):
        with self._catalog() as catalog:
            self._collect(catalog)
            sizes = catalog_sizes(catalog)
            return {p: dict(row, bytes=sizes[p]) for p, row in catalog.items()}

    def enforce(self, *, extra=0, kind=None):
        with self._catalog() as catalog:
            self._collect(catalog, extra=extra, kind=kind, require=True)

    def new_run(self):
        with self._catalog() as catalog:
            self._collect(catalog, kind='runs', require=True)
            parent = self.root / 'runs'
            parent.mkdir(exist_ok=True)
            path = parent / ('environment-' + uuid.uuid4().hex)
            self._register(catalog, path, 'runs')
            lock = (path / '.lease').open('a+b')
            fcntl.flock(lock, fcntl.LOCK_SH)
            return DiskLease(self, path, lock)

    def model(self, model, expected_sha256):
        """Serialize actual native bytes in RAM, retain exact identity admission."""
        import mujoco
        import numpy as np
        raw = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, raw)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('BuiltTaskModel native file identity differs')
        engine = hashlib.sha256((mujoco.__version__ + '/' + mujoco.mj_versionString()).encode()).hexdigest()[:16]
        folder = self.root / 'models' / engine / expected_sha256
        path = folder / 'model.mjb'
        with self._catalog() as catalog:
            if not path.exists():
                self._collect(catalog, extra=raw.nbytes, kind='models', require=True)
            self._register(catalog, folder, 'models')
            lock = (folder / '.lease').open('a+b')
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                if not path.exists():
                    temporary = folder / 'model.tmp'
                    try:
                        with temporary.open('wb') as stream:
                            stream.write(memoryview(raw))
                            stream.flush()
                            os.fsync(stream.fileno())
                        temporary.chmod(0o444)
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
                if file_digest(path) != expected_sha256:
                    raise ValueError('shared model file record drift: ' + str(path))
                path.chmod(0o444)
            except BaseException:
                lock.close()
                raise
        return dict(path=str(path), sha256=expected_sha256), DiskLease(self, folder, lock)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect or collect registered GPU-LIBERO caches; legacy caches stay untouched.')
    parser.add_argument('--root')
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    cache = DiskCache(args.root)
    if args.collect:
        rows = cache.collect()
    else:
        with cache._catalog() as catalog:
            sizes = catalog_sizes(catalog)
            rows = {p: dict(row, bytes=sizes[p]) for p, row in catalog.items()}
    print(json.dumps(dict(entries=rows, managed_bytes=sum(r['bytes'] for r in rows.values()),
                         max_bytes=cache.max_bytes, limits=cache.limits,
                         unmanaged_data='excluded; no legacy/project files deleted'), indent=2))


if __name__ == '__main__':
    main()
