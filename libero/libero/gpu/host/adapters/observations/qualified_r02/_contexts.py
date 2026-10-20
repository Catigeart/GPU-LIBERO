"""Native context methods copied from independently qualified source."""
import libero.gpu.precision as scalar
import ctypes as ct
import numpy as np

class MatrixContext:
    def __init__(self, owner, worlds):
        self.owner = owner;self.lib = owner.library;self.worlds = worlds;self.pointer = ct.c_void_p()
        self.lib.create.argtypes = [ct.c_int, ct.POINTER(ct.c_void_p)]
        self.lib.upload.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
        self.lib.step.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.read.argtypes = [ct.c_void_p] + [ct.c_void_p] * 4
        self.lib.capture.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.lib.measure.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
        self.lib.destroy.argtypes = [ct.c_void_p]
        try:self.checked(self.lib.create(worlds, ct.byref(self.pointer)))
        except BaseException:
            if self.pointer.value:self.lib.destroy(self.pointer)
            raise
    def checked(self, code):
        if code:raise RuntimeError("CUDA leaf error " + str(code))
    def live(self):
        if not self.pointer.value:raise RuntimeError("released private matrix owner")
    def upload(self, value, mask):
        self.live()
        value = np.ascontiguousarray(value, dtype=np.float32);mask = np.ascontiguousarray(mask, dtype=np.int32)
        self.checked(self.lib.upload(self.pointer, value.ctypes.data, mask.ctypes.data))
    def step(self, reset=0):
        self.live();self.checked(self.lib.step(self.pointer, reset))
    def read(self):
        self.live()
        out = np.zeros((self.worlds, 4), dtype=np.float64)
        vectors = [np.zeros(self.worlds, dtype=np.int32) for _ in range(3)]
        self.checked(self.lib.read(self.pointer, out.ctypes.data, *(x.ctypes.data for x in vectors)))
        return (out, *vectors)
    def capture(self, batch, reset):
        self.live();self.checked(self.lib.capture(self.pointer, batch, reset))
    def measure(self):
        self.live()
        result = ct.c_float();self.checked(self.lib.measure(self.pointer, ct.byref(result)));return float(result.value)
    def close(self):
        if self.pointer.value:
            ptr = self.pointer;self.pointer = ct.c_void_p();self.checked(self.lib.destroy(ptr))

class SamplerContext:
    def __init__(self, owner, layout):
        self.owner = owner;self.lib = owner.library;self.layout = layout;self.pointer = ct.c_void_p()
        self.lib.create.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_void_p]
        self.lib.upload.argtypes = [ct.c_void_p] * 4;self.lib.mutate.argtypes = [ct.c_void_p] * 2
        self.lib.step.argtypes = [ct.c_void_p, ct.c_int, scalar.ct_real]
        self.lib.read.argtypes = [ct.c_void_p] * 4
        self.lib.capture.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, scalar.ct_real]
        self.lib.measure.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)];self.lib.destroy.argtypes = [ct.c_void_p]
        try:self.check(self.lib.create(ct.byref(self.pointer), layout["integers"].ctypes.data, layout["periods"].ctypes.data))
        except BaseException:
            if self.pointer.value:self.lib.destroy(self.pointer)
            raise
    def check(self, code):
        if code:raise RuntimeError("sampling CUDA leaf error " + str(code))
    def live(self):
        if not self.pointer.value:raise RuntimeError("released private sampling owner")
    def upload(self, raw, state, flags):
        self.live()
        self.check(self.lib.upload(self.pointer, raw.ctypes.data, state.ctypes.data, flags.ctypes.data))
    def step(self, force, dt):
        self.live();self.check(self.lib.step(self.pointer, force, dt))
    def read(self):
        self.live()
        state = np.zeros(self.layout["NSTATE"], dtype=np.float64);flags = np.zeros(self.layout["NFLAGS"], dtype=np.int32);samples = np.zeros(1, dtype=np.uint64)
        self.check(self.lib.read(self.pointer, state.ctypes.data, flags.ctypes.data, samples.ctypes.data));return state, flags, int(samples[0])
    def capture(self, batch, force, dt):
        self.live();self.check(self.lib.capture(self.pointer, batch, force, dt))
    def measure(self):
        self.live()
        value = ct.c_float();self.check(self.lib.measure(self.pointer, ct.byref(value)));return float(value.value)
    def mutate(self, values):
        self.live();self.check(self.lib.mutate(self.pointer, values.ctypes.data))
    def close(self):
        if self.pointer.value:
            ptr = self.pointer;self.pointer = ct.c_void_p();self.check(self.lib.destroy(ptr))

class SceneContext:
    def __init__(self, owner, integer, local):
        self.owner = owner;self.lib = owner.library;self.pointer = ct.c_void_p()
        self.lib.create.argtypes = [ct.c_int] * 3 + [ct.c_void_p] * 2 + [ct.POINTER(ct.c_void_p)]
        self.lib.upload.argtypes = [ct.c_void_p] * 3 + [ct.c_int]
        self.lib.step.argtypes = [ct.c_void_p] + [ct.c_int] * 3
        self.lib.read.argtypes = [ct.c_void_p] * 6
        self.lib.capture.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.lib.measure.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)];self.lib.destroy.argtypes = [ct.c_void_p]
        try:self.check(self.lib.create(26, 7, 2, integer.ctypes.data, local.ctypes.data, ct.byref(self.pointer)))
        except BaseException:
            if self.pointer.value:self.lib.destroy(self.pointer)
            raise
    def check(self, code):
        if code:raise RuntimeError("scene CUDA error " + str(code))
    def live(self):
        if not self.pointer.value:raise RuntimeError("released private scene owner")
    def upload(self, a, b, publish=1):
        self.live();self.check(self.lib.upload(self.pointer, a.ctypes.data, b.ctypes.data, publish))
    def step(self, mask=1, valid=1, reset=0):
        self.live();self.check(self.lib.step(self.pointer, mask, valid, reset))
    def read(self):
        self.live()
        output = np.zeros(7 * 12 + 2 * 6, dtype=np.float64)
        reused = np.zeros(9, dtype=np.int32);computed = np.zeros(9, dtype=np.int32)
        generation = np.zeros(1, dtype=np.int64);valid = np.zeros(1, dtype=np.int32)
        self.check(self.lib.read(self.pointer, output.ctypes.data, reused.ctypes.data, computed.ctypes.data, generation.ctypes.data, valid.ctypes.data))
        return output, reused, computed, int(generation[0]), int(valid[0])
    def capture(self, batch, changing):
        self.live();self.check(self.lib.capture(self.pointer, batch, changing))
    def measure(self):
        self.live()
        value = ct.c_float();self.check(self.lib.measure(self.pointer, ct.byref(value)));return float(value.value)
    def close(self):
        if self.pointer.value:
            pointer = self.pointer;self.pointer = ct.c_void_p();self.check(self.lib.destroy(pointer))
