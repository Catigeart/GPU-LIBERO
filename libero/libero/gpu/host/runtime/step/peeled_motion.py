"""FP32-organized FP64 motion followed by the remaining native fused body."""
import ctypes as ct
import numpy as np
import warp as wp
from libero.gpu import precision as scalar
from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from .fused_substep import FusedSubstep
from .staged_motion_plan import passive_motion_plan as motion_plan


class PeeledMotionSubstep(FusedSubstep):
    def __init__(self, runtime, build_root, *, motion_threads=512, **options):
        if scalar.bits != 64 or options.get('single_solve') or options.get('profile_newton'):
            raise ValueError('peeled motion requires FP64, three solves and no fused profiler')
        if type(motion_threads) is not int or motion_threads not in (0, 128, 256, 512):
            raise ValueError('motion threads must be 0, 128, 256 or 512')
        super().__init__(runtime, build_root, **options)
        self.fused_library = self.library
        manifest = substep_native.prepare_motion(GPU_ROOT/'cuda/step/fused_substep.cu', build_root,
            self.manifest['config']['sdk'], self.stream.device.arch, nv=self.variant.nv,
            for_build=options.get('allow_build', False),
            **{k: options[k] for k in ('fast_compile', 'device_lto', 'atomic_reductions') if k in options})
        self.manifest = substep_native.finish(manifest, allow_build=options.get('allow_build', False))
        self.library = lib = ct.CDLL(self.manifest['library'])
        lib.flash_libero_motion_arguments_size.restype = ct.c_size_t
        if (lib.flash_libero_motion_arguments_size() != ct.sizeof(self.arguments) or
                lib.flash_libero_motion_nv() != self.variant.nv):
            raise RuntimeError('peeled motion ABI/shape differs')
        parent = runtime.forward.d.acceleration.static['body_parentid']
        plan = motion_plan(parent)
        if len(parent) != self.arguments.body_velocity.nb:
            raise ValueError('motion topology and native body extents differ')
        # Borrow exactly the qualified native work owners. The passive chain
        # accumulates into the joint output; disjoint owners would lose it.
        p, f, v, b = (getattr(self.arguments, n) for n in
                      ('joint_passive', 'fluid_passive', 'body_velocity', 'bias_forces'))
        self.eligible = bool(plan is not None and p.joint_disjoint and not f.diagnostic and
            p.out == f.qfrc and v.nv == p.nv == f.nv == b.nv and
            v.nb == f.nbody == b.nb and v.cvel == f.cvel == b.cvel and
            v.cdof == f.cdof == b.cdof and v.cdot == b.cdot and v.cinert == b.cinert and
            v.qvel == p.qvel == b.qvel and v.cache_acc == b.cache_acc and
            v.cache_force == b.cache_force)
        self.motion_threads = motion_threads if self.eligible else 0
        self.plan = wp.array(plan if plan is not None else np.zeros(1, np.int32),
                             dtype=wp.int32, device=self.stream.device)
        self.ready = wp.zeros(self.worlds, dtype=wp.int32, device=self.stream.device)
        configure = lib.flash_libero_motion_configure
        configure.argtypes = [ct.c_void_p]*3; configure.restype = ct.c_int
        regs, shared, local = ((ct.c_int*2)() for _ in range(3))
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(regs, shared, local), 'configure peeled motion')
        self.launch_motion = lib.flash_libero_launch_motion
        self.launch_motion.argtypes = [ct.c_void_p]*3+[ct.c_int]*3+[ct.c_void_p]
        self.launch_motion.restype = ct.c_int
        self.launch_peeled = lib.flash_libero_launch_motion_substep
        self.launch_peeled.argtypes = [ct.c_void_p]*3+[ct.c_int]*5+[ct.c_void_p]
        self.launch_peeled.restype = ct.c_int
        self.resources = dict(self.resources, execution='motion-residual', motion_eligible=self.eligible,
            motion_threads=self.motion_threads, topology_bytes=self.plan.size*4,
            handoff_bytes_per_world=4, stage_resources={name: dict(registers=regs[i],
                static_shared_bytes=shared[i], local_bytes_per_thread=local[i])
                for i, name in enumerate(('motion_passive', 'residual'))})

    def _launch(self, policy_step, entry, publish_post=True):
        self._check(self.launch_peeled(self.device_arguments.ptr, self.plan.ptr, self.ready.ptr,
            self.worlds, int(policy_step), entry, int(publish_post), self.motion_threads,
            self.stream.cuda_stream), 'peeled motion substep')

    def enqueue_motion(self, entry=FusedSubstep.START):
        """Local region entry on already admitted position inputs; no solve."""
        if not self.motion_threads:
            raise ValueError('motion-only replay requires an admitted cooperative topology')
        self._check(self.launch_motion(self.device_arguments.ptr, self.plan.ptr, self.ready.ptr,
            self.worlds, entry, self.motion_threads, self.stream.cuda_stream), 'motion/passive region')
