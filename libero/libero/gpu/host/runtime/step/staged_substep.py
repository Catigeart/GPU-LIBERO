"""Coarse FP64 forward/complete-Newton/Post Graph baseline; load-only by default."""
import ctypes as ct
import numpy as np
import warp as wp
from libero.gpu import precision as scalar
from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from .fused_substep import FusedSubstep


class StagedArguments(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ('arguments', 'phase', 'smooth_reuse_status', 'qacc')]


class StagedSubstep(FusedSubstep):
    """Reuse binding, observer policy, mixed-world recovery and prefix callbacks."""
    def __init__(self, runtime, build_root, **options):
        if scalar.bits != 64 or options.get('single_solve') or options.get('profile_newton'):
            raise ValueError('staged substeps require FP64, three solves and no fused profiler')
        super().__init__(runtime, build_root, **options)
        self.fused_library = self.library  # retains selector/ABI and the original entry
        prepared = substep_native.prepare_staged(GPU_ROOT / 'cuda/step/fused_substep.cu',
            build_root, self.manifest['config']['sdk'], self.stream.device.arch,
            **{k: options[k] for k in ('fast_compile', 'device_lto', 'atomic_reductions') if k in options},
            nv=self.variant.nv)
        self.manifest = substep_native.finish(prepared, allow_build=options.get('allow_build', False))
        self.library = lib = ct.CDLL(self.manifest['library'])
        expected = {'arguments_size': ct.sizeof(StagedArguments), 'scalar_bytes': 8,
                    'nv': self.variant.nv, 'threads': 64}
        for suffix, value in expected.items():
            query = getattr(lib, 'flash_libero_staged_' + suffix)
            query.argtypes = []; query.restype = ct.c_size_t if suffix == 'arguments_size' else ct.c_int
            if query() != value:
                raise RuntimeError('staged CUDA ABI/shape differs: ' + suffix)
        configure = lib.flash_libero_staged_configure
        configure.argtypes = [ct.c_void_p, ct.c_void_p]; configure.restype = ct.c_int
        registers, shared = (ct.c_int * 4)(), (ct.c_int * 4)()
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(registers, shared), 'configure staged')
        self.phase = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.smooth_reuse_status = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.qacc_handoff = wp.empty((self.worlds, self.variant.nv), dtype=scalar.wp_real,
                                    device=self.stream.device)
        self.staged_arguments = StagedArguments(self.device_arguments.ptr, self.phase.ptr,
            self.smooth_reuse_status.ptr, self.qacc_handoff.ptr)
        self.staged_device_arguments = wp.array(np.frombuffer(bytes(self.staged_arguments), np.uint8).copy(),
                                                dtype=wp.uint8, device=self.stream.device)
        self.staged_launch = lib.flash_libero_launch_staged_substep
        self.staged_launch.argtypes = [ct.c_void_p] + [ct.c_int] * 4 + [ct.c_void_p]
        self.staged_launch.restype = ct.c_int
        self.forward_launch = lib.flash_libero_launch_staged_forward
        self.forward_launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
        self.forward_launch.restype = ct.c_int
        self.resources = dict(self.resources, execution='staged-ABC',
            stage_resources={name: dict(registers=registers[i], static_shared_bytes=shared[i], threads=64)
                             for i, name in enumerate(('prepare', 'newton', 'post_tail', 'reset_tail'))},
            handoff_bytes_per_world=self.variant.nv * 8 + 8)

    def _launch(self, policy_step, entry, publish_post=True):
        self._check(self.staged_launch(self.staged_device_arguments.ptr, self.worlds,
            int(policy_step), entry, int(publish_post), self.stream.cuda_stream), 'launch staged')

    def enqueue_forward(self, *, publish_post=True):
        """Regional first-forward entry on already prepared position inputs."""
        self._check(self.forward_launch(self.staged_device_arguments.ptr, self.worlds,
            int(publish_post), self.stream.cuda_stream), 'launch staged forward')
