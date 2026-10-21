"""Balance dlopen only when a CDLL and all borrowed function pointers die.

Never force dlclose at environment close: retained plans/functions may still
own the library. Runtime owners synchronize GPU work before dropping them.
"""
import ctypes
import _ctypes
import weakref


def load_cdll(path):
    library = ctypes.CDLL(str(path))
    release = weakref.finalize(library, _ctypes.dlclose, library._handle)
    # CUDA/driver shutdown order is undefined at interpreter exit. The OS
    # releases mappings then; this finalizer is for ordinary live-process GC.
    release.atexit = False
    return library
