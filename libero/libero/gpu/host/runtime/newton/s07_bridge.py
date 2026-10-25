"""S07 device bridge: compile the frozen reference/newton/newton_linesearch/kernel.cu device
prefix with NVRTC and launch search_kernel through the CUDA driver API.

The frozen source file is read byte-exact and SHA-checked before use.  Only
the trailing host launch wrappers (source bytes after `extern "C" int
s07_launch`) are outside the JIT unit; their semantics are reproduced exactly
here (grid=(nworld+127)//128, block=128, stream arg, launch-status return).

Compile inputs are fully pinned (R01-F18): the device prefix bytes, the
complete option list, and the embedded shim header contents (hashed into the
build record and part of the cache key).  The produced PTX is gated against an
expected SHA256 and the source-level FP64 instruction audit; any mismatch
fails closed.  NVRTC return codes are checked and failed compiles are never
cached (R01-F19).  The build directory must resolve under the caller's
private root (env P06_S07_BUILD_ROOT, else TMPDIR) (R01-F19).
"""
from __future__ import annotations
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import ctypes as ct
import hashlib
import json
import os
import re
import tempfile
import uuid
import traceback
from pathlib import Path

NV = 43
NM = 153
FROZEN_SOURCE_SHA256 = '6f7b56af23828be60561a12555337714e089ea2e65f0a817bfea5235c4a5038b'
FROZEN_SOURCE_BYTES = 11531
DEVICE_PREFIX_BYTES = 10842
DEVICE_PREFIX_SHA256 = '7f7dc6fb4c6cca165c3106b286f05ba20423ab6475acb264064c578b8688ffc0'
HOST_MARKER = b'extern "C" int s07_launch'
EXPECTED_PTX_SHA256 = '1592dfd1c12faad8db7d83ee391954d43da4118c401b0bb8bd9cf006f364efb6'
EXPECTED_PTX_BYTES = 223960
EXPECTED_PTX_INSTRUCTIONS = {'div_rn_f64': 27, 'sqrt_rn_f64': 8, 'div_approx_f64': 0, 'sqrt_approx_f64': 0, 'fma_f64': 435}
ARCH = 'sm_120'
FIXED_OPTIONS = (b'--fmad=true', b'--prec-div=true', b'--prec-sqrt=true',
                 b'--ftz=false')
# Shim headers: byte-exact content, embedded so the translation-unit inputs
# are part of the artifact (hashes below are pinned; regenerated per build).
SHIM_FILES = {
    'cuda_runtime.h': (
        b'// device-only compile shim: host CUDA runtime is not needed for '
        b'GPU compilation\n#pragma once\n'),
    'math.h': b'#pragma once\n',
    'stdint.h': (
        b'#pragma once\ntypedef signed char int8_t; typedef unsigned char '
        b'uint8_t; typedef short int16_t; typedef unsigned short uint16_t; '
        b'typedef int int32_t; typedef unsigned int uint32_t; typedef long '
        b'long int64_t; typedef unsigned long long uint64_t;\n'),
}
EXPECTED_SHIM_SHA256 = {
    'cuda_runtime.h': 'd28063ec6f04bdcbc5499fa0f70976de55ea3740ae52764ed5574186dd10d478',
    'math.h': 'b3adf106d95b8934f690943f2921308b90be5b766c4c8ba250c792de56400201',
    'stdint.h': '8626c454ee1965ec64e526058f47be12698bc0ed5bf8fcd3cce205a8628698d7',
}

# Frozen S07Args layout (kernel.cu:8-17): 34 pointers, declaration order.
_S07_FIELDS = (
    'search', 'Ma', 'Jaref', 'quad0', 'smooth', 'qM', 'J',
    'D', 'R', 'loss', 'cone_mu', 'cone_friction',
    'efc_type', 'efc_id', 'cone_dim',
    'madr', 'parent', 'simplenum',
    'nefc', 'ne', 'nf', 'active', 'model_scalars',
    'Mv', 'Jv', 'quadGauss', 'quad', 'alpha', 'slope', 'eval_trace',
    'LSiter', 'LSresult', 'eval_tag', 'status')


class S07Args(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in _S07_FIELDS]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def device_prefix(source: Path | bytes) -> tuple[bytes, bytes, int]:
    """Return (frozen_bytes, device_prefix_bytes, cut_offset) with SHA gates."""
    raw = source.read_bytes() if isinstance(source, Path) else source
    if len(raw) != FROZEN_SOURCE_BYTES or sha256_bytes(raw) != FROZEN_SOURCE_SHA256:
        raise AssertionError(('frozen S07 source drift', len(raw), sha256_bytes(raw)))
    idx = raw.find(HOST_MARKER)
    if idx != DEVICE_PREFIX_BYTES:
        raise AssertionError(('frozen S07 host boundary drift', idx))
    prefix = raw[:idx]
    if sha256_bytes(prefix) != DEVICE_PREFIX_SHA256:
        raise AssertionError(('frozen S07 device prefix drift', sha256_bytes(prefix)))
    return raw, prefix, idx


def default_kernel_path() -> Path:
    """Frozen kernel.cu inside the running worktree (R01-F07)."""
    here = (GPU_ROOT / 'host/runtime/newton/s07_bridge.py')
    return (GPU_ROOT / 'reference/newton/newton_linesearch/kernel.cu')


def default_nvrtc() -> str:
    """NVRTC shared library: env override, system search, then this role's
    private wheel install (a toolchain input, path recorded in the build
    record; override with S07_NVRTC for a different toolchain)."""
    env = os.environ.get('S07_NVRTC')
    if env:
        return env
    import ctypes.util
    found = ctypes.util.find_library('nvrtc') or ctypes.util.find_library('nvrtc.so.12')
    if found:
        return found
    for cand in (
            '/usr/local/cuda-13.1/lib64/libnvrtc.so.12',
            '/usr/lib/wsl/lib/libnvrtc.so.12',
            '/srv/libero-split-receive-20261004-r01/actors/user-forward-r01'
            '/P06-N/execution/native-lib/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'):
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        'libnvrtc not found; set S07_NVRTC to the shared library path')


def default_build_root() -> Path:
    root = os.environ.get('P06_S07_BUILD_ROOT') or os.environ.get('TMPDIR')
    return Path(root) if root else Path(tempfile.gettempdir())


class S07DeviceBridge:
    """One compiled device module plus exact s07_launch-compatible launcher."""

    def __init__(self, *, kernel_source: Path | None = None,
                 nvrtc_path: str | None = None,
                 build_dir: Path | None = None,
                 arch: str = ARCH,
                 force_rebuild: bool = False,
                 defer_compile: bool = False):
        self.kernel_source = Path(kernel_source) if kernel_source else default_kernel_path()
        self.nvrtc_path = str(nvrtc_path) if nvrtc_path else None
        self.arch = arch
        build_root = default_build_root().resolve()
        if build_dir is None:
            build_dir = build_root / 'p06-n-s07-device'
        build_dir = Path(build_dir).resolve()
        try:
            build_dir.relative_to(build_root)
        except ValueError:
            raise ValueError(
                ('build_dir must resolve under the private build root',
                 str(build_dir), str(build_root)))
        self.build_dir = build_dir
        self.frozen_sha256 = None
        self.device_prefix_sha256 = None
        self.device_prefix_bytes = None
        self.host_cut_offset = None
        self.compile_record = {}
        self.ptx_sha256 = None
        self.ptx_bytes = None
        self._force_rebuild = force_rebuild
        # Native persistent Newton consumes the frozen source directly. Its
        # workspace still needs S07Args, but never loads this standalone module.
        # Any actual legacy load retains the original full PTX/source gates.
        self._ptx = None
        self._module = None
        self._function = None
        self._cuda = None
        if not defer_compile:
            self._prepare_ptx()

    def _prepare_ptx(self):
        """Prepare the standalone reference only when it is actually needed."""
        raw, prefix, cut = device_prefix(self.kernel_source)
        self.frozen_sha256 = sha256_bytes(raw)
        self.device_prefix_sha256 = sha256_bytes(prefix)
        self.device_prefix_bytes = len(prefix)
        self.host_cut_offset = cut
        if self.nvrtc_path is None:
            self.nvrtc_path = default_nvrtc()
        self._ptx = self._compile(prefix, self._force_rebuild)

    # ---------------- pinned compile inputs --------------------------------

    def _shim_dir(self) -> tuple[Path, dict]:
        shim = self.build_dir / 'shim'
        shim.mkdir(parents=True, exist_ok=True)
        digests = {}
        for name, data in SHIM_FILES.items():
            digest = sha256_bytes(data)
            if digest != EXPECTED_SHIM_SHA256[name]:
                raise AssertionError(('embedded shim drift', name, digest))
            target = shim / name
            target.write_bytes(data)
            digests[name] = digest
        return shim, digests

    def _option_list(self, shim_dir: Path) -> list[bytes]:
        return [('--gpu-architecture=' + self.arch).encode()] + list(FIXED_OPTIONS) + [
            b'-I' + str(shim_dir).encode()]

    def _cache_key(self, prefix_sha: str, options: list[bytes],
                   shim_digests: dict) -> dict:
        return {
            'device_prefix_sha256': prefix_sha,
            'arch': self.arch,
            'options': [x.decode() for x in options],
            'shim_sha256': shim_digests,
            'nvrtc_path': self.nvrtc_path,
        }

    # ---------------- compilation (NVRTC; no host compiler required) --------

    def _compile(self, prefix: bytes, force_rebuild: bool) -> bytes:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        shim_dir, shim_digests = self._shim_dir()
        options = self._option_list(shim_dir)
        key = self._cache_key(self.device_prefix_sha256, options, shim_digests)
        ptx_path = self.build_dir / 's07_device.ptx'
        record_path = self.build_dir / 's07_device_build.json'
        if not force_rebuild and ptx_path.exists() and record_path.exists():
            try:
                record = json.loads(record_path.read_text())
            except ValueError:
                record = {}
            if record.get('cache_key') == key:
                ptx = ptx_path.read_bytes()
                if (sha256_bytes(ptx) == record.get('ptx_sha256')
                        and sha256_bytes(ptx) == EXPECTED_PTX_SHA256):
                    self.compile_record = record
                    self.ptx_sha256 = record['ptx_sha256']
                    self.ptx_bytes = len(ptx)
                    return ptx
        # Every native attempt is immutable, including unsuccessful products.
        # Cache replacement may not erase full compiler logs or an unexpected PTX.
        self._compile_attempt = self.build_dir / ("compile-attempt-" + uuid.uuid4().hex)
        self._compile_attempt.mkdir(exist_ok=False)
        with (self._compile_attempt / "original_device_prefix.cu").open("xb") as f:
            f.write(prefix)
        self.compile_record = {"cache_key": key,
            "frozen_source_sha256": self.frozen_sha256,
            "device_prefix_sha256": self.device_prefix_sha256,
            "device_prefix_bytes": len(prefix), "nvrtc_path": self.nvrtc_path,
            "options": [x.decode() for x in options], "shim_sha256": shim_digests,
            "arch": self.arch, "attempt_root": str(self._compile_attempt)}
        try:
            ptx = self._nvrtc_compile(prefix, options, shim_digests, key)
        except BaseException:
            with (self._compile_attempt / "original_compile_failure.json").open("x") as f:
                json.dump({"compile_record": self.compile_record,
                           "full_traceback": traceback.format_exc()}, f, indent=2)
                f.write("\n")
            raise
        with (self._compile_attempt / "complete_compile_record.json").open("x") as f:
            json.dump(self.compile_record, f, indent=2)
            f.write("\n")
        ptx_path.write_bytes(ptx)
        record_path.write_text(json.dumps(self.compile_record, indent=1))
        return ptx

    def _nvrtc_compile(self, prefix: bytes, options: list[bytes],
                       shim_digests: dict, key: dict) -> bytes:
        nvrtc = ct.CDLL(self.nvrtc_path)
        for fn, args in (
                ('nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p,
                                        ct.c_char_p, ct.c_int,
                                        ct.POINTER(ct.c_char_p),
                                        ct.POINTER(ct.c_char_p)]),
                ('nvrtcCompileProgram', [ct.c_void_p, ct.c_int,
                                         ct.POINTER(ct.c_char_p)]),
                ('nvrtcGetPTXSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)]),
                ('nvrtcGetPTX', [ct.c_void_p, ct.c_char_p]),
                ('nvrtcGetProgramLogSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)]),
                ('nvrtcGetProgramLog', [ct.c_void_p, ct.c_char_p]),
                ('nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)])):
            f = getattr(nvrtc, fn)
            f.argtypes = args
            f.restype = ct.c_int
        prog = ct.c_void_p()
        rc = nvrtc.nvrtcCreateProgram(ct.byref(prog), prefix, b'kernel.cu', 0,
                                      None, None)
        self.compile_record["nvrtc_create_return"] = rc
        if rc != 0:
            raise RuntimeError(('nvrtcCreateProgram', rc))
        opts = (ct.c_char_p * len(options))(*options)
        rc = nvrtc.nvrtcCompileProgram(prog, len(options), opts)
        logsz = ct.c_size_t()
        log_size_rc = nvrtc.nvrtcGetProgramLogSize(prog, ct.byref(logsz))
        self.compile_record["nvrtc_log_size_return"] = log_size_rc
        if log_size_rc:
            self.compile_record["nvrtc_destroy_return"] = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
            raise RuntimeError(("nvrtcGetProgramLogSize", log_size_rc, "compile return", rc))
        log = ct.create_string_buffer(max(logsz.value, 1))
        log_rc = nvrtc.nvrtcGetProgramLog(prog, log)
        self.compile_record["nvrtc_log_return"] = log_rc
        if log_rc:
            self.compile_record["nvrtc_destroy_return"] = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
            raise RuntimeError(("nvrtcGetProgramLog", log_rc, "compile return", rc))
        complete_log = log.raw[:logsz.value]
        with (self._compile_attempt / "complete_nvrtc_log.raw").open("xb") as f:
            f.write(complete_log)
        self.compile_record = {
            'tool': 'NVRTC (libnvrtc), device-only JIT of the frozen prefix',
            'nvrtc_path': self.nvrtc_path,
            'arch': self.arch,
            'options': [x.decode() for x in options],
            'frozen_source_sha256': self.frozen_sha256,
            'device_prefix_bytes': self.device_prefix_bytes,
            'device_prefix_sha256': self.device_prefix_sha256,
            'host_cut_offset': self.host_cut_offset,
            'host_marker': HOST_MARKER.decode(),
            'shim_sha256': shim_digests,
            'cache_key': key,
            'compile_exit': rc,
            'nvrtc_create_return': 0,
            'nvrtc_log_size_return': log_size_rc,
            'nvrtc_log_return': log_rc,
            'compile_log': log.value.decode(errors='replace'),
            'complete_log_bytes': len(complete_log),
            'complete_log_sha256': sha256_bytes(complete_log),
            'attempt_root': str(self._compile_attempt),
            'expected_ptx_sha256': EXPECTED_PTX_SHA256,
            'expected_ptx_bytes': EXPECTED_PTX_BYTES,
            'expected_instructions': dict(EXPECTED_PTX_INSTRUCTIONS),
        }
        if rc != 0:
            self.compile_record['nvrtc_destroy_return'] = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
            raise RuntimeError(('NVRTC compile failed', rc, log.value.decode(errors='replace')))
        ptxsz = ct.c_size_t()
        rc = nvrtc.nvrtcGetPTXSize(prog, ct.byref(ptxsz))
        self.compile_record["nvrtc_ptx_size_return"] = rc
        if rc != 0 or ptxsz.value < 2:
            self.compile_record["nvrtc_destroy_return"] = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
            raise RuntimeError(('nvrtcGetPTXSize', rc, ptxsz.value))
        buf = ct.create_string_buffer(ptxsz.value)
        rc = nvrtc.nvrtcGetPTX(prog, buf)
        self.compile_record["nvrtc_ptx_return"] = rc
        destroy_rc = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
        self.compile_record["nvrtc_destroy_return"] = destroy_rc
        if rc != 0:
            raise RuntimeError(('nvrtcGetPTX', rc, 'destroy return', destroy_rc))
        if destroy_rc != 0:
            raise RuntimeError(('nvrtcDestroyProgram', destroy_rc))
        ptx = buf.raw[:ptxsz.value - 1]  # strip the NUL terminator
        if not ptx:
            raise RuntimeError('empty PTX refused')
        # Retain the actual product even if the unchanged frozen gate refuses it.
        with (self._compile_attempt / "actual_product_before_gate.ptx").open("xb") as f:
            f.write(ptx)
        self._gate_ptx(ptx)
        self.compile_record['ptx_sha256'] = sha256_bytes(ptx)
        self.compile_record['ptx_bytes'] = len(ptx)
        return ptx

    def _gate_ptx(self, ptx: bytes):
        digest = sha256_bytes(ptx)
        if digest != EXPECTED_PTX_SHA256 or len(ptx) != EXPECTED_PTX_BYTES:
            raise AssertionError(('unexpected PTX product (fail closed)',
                                  digest, len(ptx)))
        text = ptx.decode()
        counts = {name: len(re.findall(pattern, text))
                  for name, pattern in {
                      'div_rn_f64': r'\bdiv\.rn\.f64\b',
                      'sqrt_rn_f64': r'\bsqrt\.rn\.f64\b',
                      'div_approx_f64': r'\bdiv\.approx\.f64\b',
                      'sqrt_approx_f64': r'\bsqrt\.approx\.f64\b',
                      'fma_f64': r'\bfma(?:\.rn)?\.f64\b'}.items()}
        if counts != EXPECTED_PTX_INSTRUCTIONS:
            raise AssertionError(('PTX instruction audit failed', counts))
        self.compile_record['instruction_counts'] = counts

    # ---------------- driver load + exact launch semantics -----------------

    def _driver(self):
        if self._cuda is not None:
            return self._cuda
        cuda = ct.CDLL('libcuda.so.1')
        cuda.cuInit.argtypes = [ct.c_uint]
        cuda.cuInit.restype = ct.c_int
        cuda.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        cuda.cuModuleLoadData.restype = ct.c_int
        cuda.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p),
                                             ct.c_void_p, ct.c_char_p]
        cuda.cuModuleGetFunction.restype = ct.c_int
        cuda.cuLaunchKernel.argtypes = [
            ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
            ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
            ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
        cuda.cuLaunchKernel.restype = ct.c_int
        # NOTE: this WSL libcuda.so.1 does not export cuGetLastError /
        # cuPeekAtLastError (verified via nm -D).  The launch result is
        # reported from cuLaunchKernel's own status, which is the frozen
        # s07_launch error contract in practice (its callers raise on
        # nonzero).  Recorded as a documented environment deviation.
        cuda.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
        cuda.cuCtxGetCurrent.restype = ct.c_int
        rc = cuda.cuInit(0)
        if rc != 0:
            raise RuntimeError(('cuInit', rc))
        self._cuda = cuda
        return cuda

    def load(self):
        if self._module is not None:
            return
        if self._ptx is None:
            self._prepare_ptx()
        cuda = self._driver()
        ctx = ct.c_void_p()
        rc = cuda.cuCtxGetCurrent(ct.byref(ctx))
        if rc != 0 or not ctx.value:
            raise RuntimeError(('cuCtxGetCurrent', rc, 'no active CUDA context'))
        module = ct.c_void_p()
        buf = ct.create_string_buffer(self._ptx)
        rc = cuda.cuModuleLoadData(ct.byref(module), ct.cast(buf, ct.c_void_p))
        if rc != 0:
            raise RuntimeError(('cuModuleLoadData', rc))
        function = ct.c_void_p()
        rc = cuda.cuModuleGetFunction(ct.byref(function), module,
                                      b'_Z13search_kernel7S07Argsiiiii')
        if rc != 0:
            raise RuntimeError(('cuModuleGetFunction search_kernel', rc))
        self._module = module
        self._function = function

    def launch(self, args: S07Args, nworld: int, nv: int, nM: int,
               rows: int, tracecap: int, stream_ptr: int) -> int:
        """Exact s07_launch semantics from frozen kernel.cu:297-303."""
        if not isinstance(args, S07Args) or not stream_ptr or nworld <= 0 \
                or nv <= 0 or nM <= 0 or rows < 0:
            return -1
        self.load()
        cuda = self._driver()
        grid = (nworld + 127) // 128
        kern_args = (ct.c_void_p * 6)(
            ct.addressof(args),
            _stack_int(nworld), _stack_int(nv), _stack_int(nM),
            _stack_int(rows), _stack_int(tracecap))
        extra = (ct.c_void_p * 2)(ct.c_void_p(0), ct.c_void_p(0))
        rc = cuda.cuLaunchKernel(self._function,
                                 grid, 1, 1, 128, 1, 1, 0,
                                 ct.c_void_p(stream_ptr),
                                 kern_args, extra)
        return int(rc)

    def corrupt_eval_cost(self, eval_trace_ptr: int, stream_ptr: int) -> int:
        """Verification-only route kept available; production never calls it."""
        from .s07_fault import S07FaultModule
        if not hasattr(self, "_fault_module"):
            self._fault_module = S07FaultModule(self)
        return self._fault_module.launch(eval_trace_ptr, stream_ptr)


_INT_SLOTS: list = []


def _stack_int(value: int) -> int:
    box = ct.c_int(value)
    _INT_SLOTS.append(box)  # keep alive through cuLaunchKernel
    if len(_INT_SLOTS) > 4096:
        del _INT_SLOTS[:2048]
    return ct.addressof(box)
