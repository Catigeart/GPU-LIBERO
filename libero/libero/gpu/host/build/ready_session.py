"""Explicit readiness reuse for a caller-owned, immutable offline build.

Qualification hashes the complete bank once. Each subsequent construction
checks the source file set and metadata of all qualified inputs. This is a
process-local optimization under an immutable-tree contract, not a filesystem
lock or a defense against forged filesystem metadata. Prepared scene contents
remain separately checked by aot.scene_for(). There is no GPU import here.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
import time

from .compilation import MissingCompiledProduct


_QUALIFIED = object()


def _file_signature(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('ready session requires a regular, non-symlink file: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _directory_signature(path):
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError('ready session requires non-symlink artifact directories: ' + str(path))
    # Adding an unrelated file does not change an already qualified product.
    return value.st_dev, value.st_ino, value.st_mode


def _artifact_paths(root, record):
    files, directories = {}, {root}
    for spelling, digest in record['artifacts'].items():
        path = Path(spelling)
        if (path.is_absolute() or not path.parts or '..' in path.parts or
                not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('invalid unified artifact path or SHA: ' + str(spelling))
        target = root / path
        files[str(target)] = target
        directories.update(target.parents[:len(path.parts)])
    return files, directories


def _snapshot(files, directories):
    return ({name: _file_signature(path) for name, path in files.items()},
            {str(path): _directory_signature(path) for path in directories})


class ReadySession:
    """A qualified in-process build; obtain through qualify_ready_session().

    Keep its source and build tree immutable for the entire session. Ordinary
    writes, replacement, deletion and symlink changes invalidate it permanently.
    A new session must perform full qualification after a change. A session
    neither grants compilation authority nor certifies a newly changed source.
    """

    def __init__(self, token, *, root, num_envs, seed, record, manifest_sha,
                 source_paths, files, directories, signatures, elapsed):
        if token is not _QUALIFIED:
            raise TypeError('use qualify_ready_session to create a ready session')
        self._root, self._num_envs, self._seed = root, num_envs, seed
        self._record = deepcopy(record)
        self._source_paths = source_paths
        self._files, self._directories = files, directories
        self._signatures = signatures
        self._pid, self._lock = os.getpid(), RLock()
        self._invalidated = None
        self._info = dict(root=str(root), num_envs=num_envs, seed=seed,
            source_identity=record['source_identity'], manifest_sha256=manifest_sha,
            tasks=len(record['tasks']), artifacts=len(record['artifacts']),
            source_files=len(source_paths), qualification_seconds=elapsed,
            validation='initial-full-sha256; reuse-source-set-and-file-metadata',
            immutable_tree_required=True)

    @classmethod
    def _qualify(cls, root, *, num_envs, seed):
        from . import aot
        started = time.perf_counter()
        root = aot.project_root(root)
        manifest = root / 'AOT_BUILD.json'
        try:
            manifest_before = _file_signature(manifest)
        except FileNotFoundError as error:
            raise MissingCompiledProduct('Unified LIBERO build is absent; run python -m libero.gpu.build') from error
        raw = manifest.read_bytes()
        record = json.loads(raw)
        source_paths = aot._source_identity_files(include_excluded=True)
        files, directories = _artifact_paths(root, record)
        files.update({str(path): path for path in source_paths.values()})
        files[str(manifest)] = manifest
        before = _snapshot(files, directories)
        if before[0][str(manifest)] != manifest_before:
            raise MissingCompiledProduct('Unified manifest changed during ready session qualification')
        aot._require_ready_header(record, num_envs=num_envs, seed=seed,
            fingerprint=aot.source_identity())
        contract = aot.observation_product_contract()
        if not record['tasks']:
            raise MissingCompiledProduct('Unified LIBERO build has no qualified tasks')
        for key in record['tasks']:
            aot._require_ready_task(record, root, key, observation_options=None,
                debug=False, contract=contract)
        for spelling, digest in record['artifacts'].items():
            if aot._sha(root / spelling) != digest:
                raise ValueError('prebuilt artifact differs: ' + spelling)
        if (aot._source_identity_files(include_excluded=True) != source_paths or
                _snapshot(files, directories) != before):
            raise MissingCompiledProduct('Source or build changed during ready session qualification')
        return cls(_QUALIFIED, root=root, num_envs=num_envs, seed=seed,
            record=record, manifest_sha=hashlib.sha256(raw).hexdigest(), source_paths=source_paths,
            files=files, directories=directories, signatures=before,
            elapsed=time.perf_counter()-started)

    @property
    def info(self):
        return dict(self._info)

    @property
    def record(self):
        """A detached audit copy, never the internal qualification record."""
        with self._lock:
            return deepcopy(self._record)

    def _check(self):
        from . import aot
        if self._invalidated is not None:
            raise MissingCompiledProduct('Ready session is invalid: ' + self._invalidated)
        if os.getpid() != self._pid:
            self._invalidated = 'a ready session cannot cross a process boundary'
        else:
            try:
                if (aot._source_identity_files(include_excluded=True) != self._source_paths or
                        _snapshot(self._files, self._directories) != self._signatures):
                    self._invalidated = 'qualified source or build metadata changed; requalify the immutable tree'
            except (OSError, ValueError) as error:
                self._invalidated = 'qualified source or build is unavailable: ' + str(error)
        if self._invalidated is not None:
            raise MissingCompiledProduct('Ready session is invalid: ' + self._invalidated)

    def require_ready(self, root, bddl, *, num_envs, seed, observation_options=None, debug=False,
                      return_record=True):
        from . import aot
        with self._lock:
            if (aot.project_root(root) != self._root or num_envs != self._num_envs or seed != self._seed):
                raise MissingCompiledProduct('Ready session root, num_envs or seed differs from this request')
            self._check()
            key = Path(bddl).parent.name + '/' + Path(bddl).name
            aot._require_ready_task(self._record, self._root, key,
                observation_options=observation_options, debug=debug,
                contract=self._record['observation_contract'])
            return deepcopy(self._record) if return_record else None

    def close(self):
        with self._lock:
            self._invalidated = 'closed'
            self._record = {}

    def __enter__(self):
        with self._lock:
            self._check()
        return self

    def __exit__(self, *exception):
        self.close()

    def __reduce_ex__(self, protocol):
        raise TypeError('ReadySession is process-local and cannot be serialized')
