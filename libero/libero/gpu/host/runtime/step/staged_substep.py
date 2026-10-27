"""Regional FP64 motion/forward/Newton/Post Graph; load-only by default."""
import ctypes as ct
import numpy as np
import warp as wp
from libero.gpu import precision as scalar
from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from .fused_substep import FusedSubstep
from .staged_motion_plan import build_motion_plan, motion_threads as select_motion_threads


class StagedArguments(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ('arguments', 'phase', 'smooth_reuse_status', 'qacc', 'motion_reuse', 'motion_plan')]


class StagedSubstep(FusedSubstep):
    """Reuse binding, observer policy, mixed-world recovery and prefix callbacks."""
    def __init__(self, runtime, build_root, *, motion_threads=None, **options):
        if scalar.bits != 64 or options.get('single_solve') or options.get('profile_newton'):
            raise ValueError('staged substeps require FP64, three solves and no fused profiler')
        super().__init__(runtime, build_root, **options)
        self.fused_library = self.library  # retains selector/ABI and the original entry
        prepared = substep_native.prepare_staged(GPU_ROOT / 'cuda/step/fused_substep.cu',
            build_root, self.manifest['config']['sdk'], self.stream.device.arch,
            **{k: options[k] for k in ('fast_compile', 'device_lto', 'atomic_reductions') if k in options},
            nv=self.variant.nv, nM=self.variant.nM, nq=self.variant.nq, njnt=self.variant.njnt, nu=self.variant.nu, for_build=options.get('allow_build', False))
        self.manifest = substep_native.finish(prepared, allow_build=options.get('allow_build', False))
        self.library = lib = ct.CDLL(self.manifest['library'])
        expected = {'arguments_size': ct.sizeof(StagedArguments), 'scalar_bytes': 8,
                    'nv': self.variant.nv, 'threads': 64}
        for suffix, value in expected.items():
            query = getattr(lib, 'flash_libero_staged_' + suffix)
            query.argtypes = []; query.restype = ct.c_size_t if suffix == 'arguments_size' else ct.c_int
            if query() != value:
                raise RuntimeError('staged CUDA ABI/shape differs: ' + suffix)
        bodies = self.arguments.body_velocity.nb
        parent = runtime.forward.d.acceleration.static['body_parentid']
        post = runtime.forward.n.cuda_entry_core.params
        if len(parent) != bodies or post.scalar_226 != bodies or post.scalar_245 != 6:
            raise ValueError('compact Post requires matching body extents and six sensor outputs')
        plan = build_motion_plan(parent)
        selected = select_motion_threads(plan, bodies)
        if motion_threads is not None:
            if type(motion_threads) is not int or motion_threads not in (0, 64, 128, 256):
                raise ValueError('motion threads must be 0, 64, 128 or 256')
            if motion_threads and not selected:
                raise ValueError('branch motion is outside the admitted duplication budget')
            selected = motion_threads
        self.motion_threads = selected
        configure = lib.flash_libero_staged_configure
        configure.argtypes = [ct.c_void_p] * 4 + [ct.c_int]; configure.restype = ct.c_int
        registers, shared, local, dynamic = ((ct.c_int * 5)() for _ in range(4))
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(registers, shared, local, dynamic, bodies), 'configure staged')
        self.post_shared_bytes = dynamic[2]
        self.phase = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.smooth_reuse_status = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.qacc_handoff = wp.empty((self.worlds, self.variant.nv), dtype=scalar.wp_real,
                                    device=self.stream.device)
        self.motion_reuse = wp.empty(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.motion_plan = wp.array(plan, dtype=wp.int32, device=self.stream.device)
        self.staged_arguments = StagedArguments(self.device_arguments.ptr, self.phase.ptr,
            self.smooth_reuse_status.ptr, self.qacc_handoff.ptr, self.motion_reuse.ptr, self.motion_plan.ptr)
        self.staged_device_arguments = wp.array(np.frombuffer(bytes(self.staged_arguments), np.uint8).copy(),
                                                dtype=wp.uint8, device=self.stream.device)
        self.staged_launch = lib.flash_libero_launch_staged_substep
        self.staged_launch.argtypes = [ct.c_void_p] + [ct.c_int] * 6 + [ct.c_void_p]
        self.staged_launch.restype = ct.c_int
        self.forward_launch = lib.flash_libero_launch_staged_forward
        self.forward_launch.argtypes = [ct.c_void_p] + [ct.c_int] * 4 + [ct.c_void_p]
        self.forward_launch.restype = ct.c_int
        self.prepare_launch = lib.flash_libero_launch_staged_prepare
        self.prepare_launch.argtypes = [ct.c_void_p] + [ct.c_int] * 4 + [ct.c_void_p]
        self.prepare_launch.restype = ct.c_int
        self.resources = dict(self.resources, execution='staged-motion-ABC',
            stage_resources={name: dict(registers=registers[i], static_shared_bytes=shared[i],
                dynamic_shared_bytes=dynamic[i], local_bytes_per_thread=local[i],
                threads=selected if i == 4 else 64)
                for i, name in enumerate(('prepare', 'newton', 'post_tail', 'reset_tail', 'motion'))},
            motion_branches=int(plan[0]), motion_path_bodies=int(plan[1 + plan[0]]),
            motion_topology_bytes=plan.nbytes, handoff_bytes_per_world=self.variant.nv * 8 + 12)

    def _launch(self, policy_step, entry, publish_post=True):
        self._check(self.staged_launch(self.staged_device_arguments.ptr, self.worlds,
            int(policy_step), entry, int(publish_post), self.motion_threads,
            self.post_shared_bytes, self.stream.cuda_stream), 'launch staged')

    def enqueue_forward(self, *, publish_post=True):
        """Regional first-forward entry on already prepared position inputs."""
        self._check(self.forward_launch(self.staged_device_arguments.ptr, self.worlds,
            int(publish_post), self.motion_threads, self.post_shared_bytes,
            self.stream.cuda_stream), 'launch staged forward')

    def enqueue_prepare(self, *, entry=FusedSubstep.START, phase=0):
        """Local preparation boundary on already produced position inputs."""
        self._check(self.prepare_launch(self.staged_device_arguments.ptr, self.worlds,
            entry, phase, self.motion_threads, self.stream.cuda_stream), 'launch staged prepare')
