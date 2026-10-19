"""Optional wall-clock attribution of cold construction; no step instrumentation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter


_REPORT = ContextVar('libero_startup_report', default=None)
_PATH = ContextVar('libero_startup_path', default='')


def profile_construction(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        enabled = kwargs.get('profile_startup', False)
        if type(enabled) is not bool:
            raise ValueError('profile_startup must be bool')
        self.startup_timings = {}
        token = _REPORT.set(self.startup_timings if enabled else None)
        path = _PATH.set('')
        try:
            with startup_phase('constructor'):
                return function(self, *args, **kwargs)
        finally:
            _PATH.reset(path)
            _REPORT.reset(token)
    return wrapped


@contextmanager
def startup_phase(name):
    report = _REPORT.get()
    if report is None:
        yield
        return
    parent = _PATH.get()
    name = parent + '.' + name if parent else name
    token = _PATH.set(name)
    start = perf_counter()
    try:
        yield
    finally:
        report[name] = report.get(name, 0.0) + perf_counter() - start
        _PATH.reset(token)
