"""Scalar policy shared by allocations, Warp kernels and native CUDA products.

The legacy cold assembly imports concrete Warp signatures. Select precision
before constructing a GPU owner; it is fixed once a typed module is imported.
Independent processes may freely use the FP32 and FP64 products in one build.
"""
import os
from threading import RLock


def _normalize(value):
    names = {'fp32': 32, 'float32': 32, 'fp64': 64, 'float64': 64}
    if value not in names:
        raise ValueError('precision must be fp32 or fp64')
    return names[value]


_frozen = False
_lock = RLock()


def _set_attributes(requested):
    global bits, name, itemsize, epsilon, jar_cap, safe_magnitude, norm_limit
    bits, name, itemsize = requested, 'float' + str(requested), requested // 8
    epsilon = 1.1920928955078125e-7 if bits == 32 else 2.220446049250313e-16
    jar_cap = float.fromhex('0x1.fffffep127') if bits == 32 else 1e50
    safe_magnitude = 1e10 if bits == 32 else 1e100
    norm_limit = float.fromhex('0x1.fffffep127') if bits == 32 else 1e300


_set_attributes(_normalize(os.environ.get('LIBERO_GPU_PRECISION', 'fp64')))


def select(value=None):
    """Choose the process scalar before importing typed physics modules."""
    with _lock:
        return _select(value)


def _select(value):
    global _frozen
    requested = bits if value is None else _normalize(value)
    if _frozen and requested != bits:
        raise RuntimeError('GPU precision is already fixed in this process; '
                           'construct the other precision in a separate process')
    _set_attributes(requested)
    os.environ['LIBERO_GPU_PRECISION'] = 'fp' + str(bits)
    _frozen = True
    return 'fp' + str(bits)


def __getattr__(attribute):
    with _lock:
        return _resolve(attribute)


def _resolve(attribute):
    global _frozen
    if attribute in ('wp_real', 'wp_bits', 'vec3_real', 'vec4_real', 'mat33_real', 'quat_real'):
        import warp as wp
        _frozen = True
        types = {'wp_real': ('float32', 'float64'),
                 'wp_bits': ('uint32', 'uint64'),
                 'vec3_real': ('vec3f', 'vec3d'),
                 'vec4_real': ('vec4f', 'vec4d'),
                 'mat33_real': ('mat33f', 'mat33d'),
                 'quat_real': ('quatf', 'quatd')}
        return getattr(wp, types[attribute][bits == 64])
    if attribute == 'ct_real':
        import ctypes
        _frozen = True
        return ctypes.c_float if bits == 32 else ctypes.c_double
    raise AttributeError(attribute)


def native_flags():
    return ['-DFLASH_REAL_BITS=' + str(bits)]


def solver_tolerance(original):
    return max(float(original), 1e-6) if bits == 32 else float(original)
