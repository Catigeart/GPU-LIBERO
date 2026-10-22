"""Opt-in FP64 init/remainder split, retaining accepted motion/force owners."""
import ctypes as ct
import numpy as np
import warp as wp
from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from .peeled_force import PeeledForceSubstep
from .staged_substep import StagedArguments


def selected_for_control(nv, atomic_reductions=None):
    """Measured admission, independent of task ID; other shapes retain force fusion.

    NV21 and NV57 lost in complete controls. NV39 passed; intermediate shapes
    have not yet passed the catalogue gate. Explicit regional experiments may
    still construct AlignedSubstep directly to evaluate those shapes.
    """
    return nv == 39 and atomic_reductions is not False


class AlignedArguments(ct.Structure):
    _fields_ = [('stages', StagedArguments), ('force_ready', ct.c_void_p),
                ('newton_initial', ct.c_void_p)]


class AlignedSubstep(PeeledForceSubstep):
    """Actual separate init256/resume64 kernels; no change to the default path."""
    def __init__(self, runtime, build_root, *, initial_threads=256, **options):
        if type(initial_threads) is not int or initial_threads not in (64, 128, 256):
            raise ValueError('initial_threads must be 64, 128 or 256')
        if options.get('atomic_reductions') is False:
            raise ValueError('aligned prototype requires the normal reduction product')
        super().__init__(runtime, build_root, **options)
        if not self.manifest['config']['atomic_reductions']:
            raise ValueError('aligned prototype requires the normal reduction product')
        self.force_library = self.library
        prepared = substep_native.prepare_staged(GPU_ROOT/'cuda/step/fused_substep.cu',
            build_root, self.manifest['config']['sdk'], self.stream.device.arch,
            nv=self.variant.nv, **{k: options[k] for k in
            ('fast_compile', 'device_lto', 'atomic_reductions') if k in options})
        self.manifest = substep_native.finish(prepared, allow_build=options.get('allow_build', False))
        self.library = lib = ct.CDLL(self.manifest['library'])
        lib.flash_libero_aligned_arguments_size.restype = ct.c_size_t
        lib.flash_libero_aligned_checkpoint_bytes.restype = ct.c_size_t
        lib.flash_libero_staged_nv.restype = ct.c_int
        if (lib.flash_libero_aligned_arguments_size() != ct.sizeof(AlignedArguments) or
                lib.flash_libero_staged_nv() != self.variant.nv):
            raise RuntimeError('aligned CUDA ABI/shape differs; rebuild offline')
        limit = lib.flash_libero_aligned_initial_thread_limit()
        if initial_threads > limit:
            raise ValueError('this offline GEMV product permits at most '+str(limit)+' initializer threads')
        configure = lib.flash_libero_aligned_configure
        configure.argtypes = [ct.c_void_p]*4+[ct.c_int]; configure.restype = ct.c_int
        regs, shared, local, dynamic = ((ct.c_int*5)() for _ in range(4))
        bodies = self.arguments.body_velocity.nb
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(regs, shared, local, dynamic, bodies), 'configure aligned')
        self.phase = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.smooth_reuse_status = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.qacc_handoff = wp.empty((self.worlds, self.variant.nv), dtype=wp.float64,
                                     device=self.stream.device)
        checkpoint_bytes = lib.flash_libero_aligned_checkpoint_bytes()
        self.newton_initial = wp.empty(self.worlds*checkpoint_bytes, dtype=wp.uint8,
                                       device=self.stream.device)
        stages = StagedArguments(self.device_arguments.ptr, self.phase.ptr,
            self.smooth_reuse_status.ptr, self.qacc_handoff.ptr, self.ready.ptr, self.plan.ptr)
        self.aligned_arguments = AlignedArguments(stages, self.force_ready.ptr, self.newton_initial.ptr)
        self.aligned_device_arguments = wp.array(np.frombuffer(bytes(self.aligned_arguments), np.uint8).copy(),
                                                 dtype=wp.uint8, device=self.stream.device)
        self.initial_threads = initial_threads
        self.post_shared_bytes = dynamic[3]
        self.aligned_launch = lib.flash_libero_launch_aligned_substep
        self.aligned_launch.argtypes = [ct.c_void_p]+[ct.c_int]*8+[ct.c_void_p]
        self.aligned_launch.restype = ct.c_int
        self.resources = dict(self.resources, execution='motion-force-prepare-init-resume-post',
            checkpoint_bytes_per_world=checkpoint_bytes, initial_threads=initial_threads,
            aligned_resources={name: dict(registers=regs[i], static_shared_bytes=shared[i],
                dynamic_shared_bytes=dynamic[i], local_bytes_per_thread=local[i])
                for i, name in enumerate(('prepare', 'initialize', 'resume', 'post_tail', 'reset_tail'))})

    def _launch(self, policy_step, entry, publish_post=True):
        produce = bool(self.motion_threads and entry != self.ACCELERATION_RESET)
        if produce:
            self.enqueue_motion(entry)
            if self.force_threads:
                self.enqueue_force(entry)
        self._check(self.aligned_launch(self.aligned_device_arguments.ptr, self.worlds,
            int(policy_step), entry, int(publish_post), int(produce),
            int(produce and bool(self.force_threads)), self.initial_threads,
            self.post_shared_bytes, self.stream.cuda_stream), 'launch aligned substep')


class EulerAlignedSubstep(AlignedSubstep):
    """Separate Euler for admitted shapes, with the original reset continuation."""
    @staticmethod
    def supports(euler):
        return (0 < euler.nv <= 57 and 0 < euler.nq <= 65 and
                0 < euler.nm <= 212 and 0 <= euler.nj <= 18)

    def __init__(self, runtime, build_root, **options):
        super().__init__(runtime, build_root, **options)
        e = self.arguments.euler
        if not self.supports(e):
            raise ValueError('Euler experiment exceeds the offline catalogue arena')
        configure = self.library.flash_libero_aligned_euler_configure
        configure.argtypes = [ct.POINTER(ct.c_int), ct.c_int]
        configure.restype = ct.c_int
        values = (ct.c_int * 8)()
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(values, self.arguments.body_velocity.nb), 'configure aligned Euler')
        self.aligned_launch = self.library.flash_libero_launch_aligned_euler_substep
        self.aligned_launch.argtypes = [ct.c_void_p]+[ct.c_int]*8+[ct.c_void_p]
        self.aligned_launch.restype = ct.c_int
        self.resources = dict(self.resources, execution='motion-force-init-resume-post-euler',
            euler_split=True, euler_threads=64,
            euler_resources={name:dict(registers=values[4*i],static_shared_bytes=values[4*i+1],
                local_bytes_per_thread=values[4*i+2],dynamic_shared_bytes=values[4*i+3])
                for i,name in enumerate(('euler','deferred_post_tail'))})
