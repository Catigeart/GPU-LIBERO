"""Explicit offline compilation and load-only runtime boundaries."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import inspect
import threading

_SCOPE = ContextVar('libero_compilation_scope', default=None)
_INSTALL_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_FROZEN_COUNT = 0
_BUILD_SCOPES = []


class MissingCompiledProduct(RuntimeError):
    pass


def current_scope():
    return _SCOPE.get()


def compilation_allowed():
    scope = _SCOPE.get()
    return not _FROZEN_COUNT and (scope is None or scope['allow_compile'])


def require_compilation(kind, identity):
    if not compilation_allowed():
        raise MissingCompiledProduct(
            f'Prebuilt {kind} is absent: {identity}. '
            'Prepare this product with its offline build/preparation entry before creating environments; '
            'runtime compilation is disabled.')


@contextmanager
def compilation_scope(allow_compile):
    global _FROZEN_COUNT
    if type(allow_compile) is not bool:
        raise ValueError('allow_compile must be bool')
    state = dict(allow_compile=allow_compile, events=[])
    token = _SCOPE.set(state)
    with _STATE_LOCK:
        if allow_compile:
            _BUILD_SCOPES.append(state)
        else:
            _FROZEN_COUNT += 1
    try:
        yield state
    finally:
        with _STATE_LOCK:
            if allow_compile:
                index = next(i for i, entry in enumerate(_BUILD_SCOPES) if entry is state)
                _BUILD_SCOPES.pop(index)
            else:
                _FROZEN_COUNT -= 1
        _SCOPE.reset(token)


def record_event(event):
    scope = _SCOPE.get()
    if scope is None and len(_BUILD_SCOPES) == 1:
        scope = _BUILD_SCOPES[0]
    if scope is not None:
        scope['events'].append(event)


def construction_scope(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        supplied = signature.bind(*args, **kwargs).arguments
        mode = supplied.get('allow_compile', False)
        with compilation_scope(mode) as scope:
            result = function(*args, **kwargs)
            owner = args[0]
            if hasattr(owner, 'build_report'):
                owner.build_report['events'] = list(scope['events'])
            return result
    return wrapped


def environment_scope(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        # Constructor permission is not permission for future step/reset calls.
        # Only a surrounding explicit offline scope may compile lazy products.
        scope = current_scope()
        with compilation_scope(False) if scope is None else nullcontext(scope) as scope:
            try:
                return function(self, *args, **kwargs)
            finally:
                if hasattr(self, 'build_report'):
                    self.build_report['events'].extend(scope['events'])
    return wrapped


def load_only_entry(function):
    """Freeze ordinary public calls; reuse explicit offline authority/events.

    The authority belongs to the call, not to the environment's constructor:
    an environment prepared offline must become load-only afterwards too.
    Nested entry points reuse the scope instead of adding per-operation gates.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        scope = current_scope()
        with compilation_scope(False) if scope is None else nullcontext(scope):
            return function(*args, **kwargs)
    return wrapped


def install_warp_guard():
    """Observe Warp's real cache-miss boundary, including NVRTC compilation."""
    import warp._src.context as context
    with _INSTALL_LOCK:
        original = context.Module._compile
        if getattr(original, '_libero_compile_guard', False):
            return
        @wraps(original)
        def compile_(module, *args, **kwargs):
            result = original(module, *args, **kwargs)
            if result:
                from libero.gpu.host.runtime.core.product_paths import record_build
                record_build('warp', module.name, (('warp-compile', module.name),))
            return result
        original_codegen = context.Module._run_codegen
        @wraps(original_codegen)
        def codegen(module, *args, **kwargs):
            require_compilation('Warp module', module.name)
            return original_codegen(module, *args, **kwargs)
        compile_._libero_compile_guard = True
        context.Module._compile = compile_
        context.Module._run_codegen = codegen
