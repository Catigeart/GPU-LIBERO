"""Own temporary native loader files until the actual runtime closes."""
from functools import wraps
from contextlib import nullcontext
from threading import RLock
from pathlib import Path
from tempfile import TemporaryDirectory


class BuildLifetime:
    def __init__(self):
        self._temporary = TemporaryDirectory(prefix="flash-libero-")
        self.path = Path(self._temporary.name)
        self._references = 1  # Constructor scope; runtimes acquire their own.
        self.retire_callbacks = []
        self._lock = RLock()

    def retain(self):
        with self._lock:
            if self._references == 0:
                raise RuntimeError('cold build lifetime already closed')
            self._references += 1
        return self

    def close(self):
        with self._lock:
            if self._references:
                self._references -= 1
            if self._references:
                return
            errors, pending = [], []
            for callback in self.retire_callbacks:
                try:
                    callback()
                except BaseException as error:
                    errors.append(error)
                    pending.append(callback)
            self.retire_callbacks[:] = pending
            if errors:
                if len(errors) > 1:
                    errors[0].gpu_cleanup_errors = [repr(error) for error in errors[1:]]
                raise errors[0]
            self._temporary.cleanup()


def cold_build_scope(function=None, *, project_root=None):
    """A transaction may return a runtime that explicitly retained this scope."""
    if function is None:
        return lambda callback: cold_build_scope(callback, project_root=project_root)
    @wraps(function)
    def call(*args, **kwargs):
        from .product_paths import product_build, build_project
        root = project_root() if project_root is not None else None
        project = build_project(root) if root is not None else nullcontext()
        with project, product_build():
            return function(*args, **kwargs)
    return call


def managed_build(function):
    """Close failed constructors; retain files until successful runtime close."""
    @wraps(function)
    def prepare(*args, **kwargs):
        from .product_paths import product_build, current_build_lifetime
        with product_build():
            lifetime = current_build_lifetime()
            bundle = function(*args, **kwargs)
            close_runtime = bundle.close
            lifetime.retain()
            closed = False
            gpu_closed = False

            def close():
                nonlocal closed, gpu_closed
                with lifetime._lock:
                    if closed:
                        return
                    if not gpu_closed:
                        close_runtime()  # Retain disk files until GPU release succeeds.
                        gpu_closed = True
                    lifetime.close()
                    closed = True

            bundle.close = close
            bundle.disk_build_root = lifetime.path
            return bundle
    return prepare
