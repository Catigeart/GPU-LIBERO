"""Compatibility handle for selecting an offline build.

Each request reads the current manifest and checks the selected observer products.
This handle does not certify source trees, cache validation results, or grant
compilation authority.
"""
from pathlib import Path
import time

from .compilation import MissingCompiledProduct


class ReadySession:
    def __init__(self, root, *, num_envs, seed):
        self._root = Path(root)
        self._num_envs, self._seed = num_envs, seed
        self._closed = False
        self._info = {}

    @classmethod
    def _qualify(cls, root, *, num_envs, seed):
        from . import aot
        started = time.perf_counter()
        root = aot.project_root(root)
        record = aot._read_ready_record(root, num_envs=num_envs, seed=seed)
        session = cls(root, num_envs=num_envs, seed=seed)
        session._info = dict(root=str(root), num_envs=num_envs, seed=seed,
            source_identity=record.get('source_identity'),
            tasks=len(record['tasks']), artifacts=len(record.get('artifacts', {})),
            qualification_seconds=time.perf_counter()-started,
            validation='manifest-format-and-selected-profile-products',
            immutable_tree_required=False)
        return session

    def _check(self):
        if self._closed:
            raise MissingCompiledProduct('Ready session is closed')

    @property
    def info(self):
        return dict(self._info)

    @property
    def record(self):
        from . import aot
        self._check()
        return aot._read_ready_record(self._root, num_envs=self._num_envs, seed=self._seed)

    def require_ready(self, root, bddl, *, num_envs, seed, observation_options=None,
                      debug=False, return_record=True):
        from . import aot
        self._check()
        if (aot.project_root(root), num_envs, seed) != (self._root, self._num_envs, self._seed):
            raise MissingCompiledProduct('Ready session root, num_envs or seed differs from this request')
        return aot.require_ready(root, bddl, num_envs=num_envs, seed=seed,
            observation_options=observation_options, debug=debug, return_record=return_record)

    def close(self):
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exception):
        self.close()
