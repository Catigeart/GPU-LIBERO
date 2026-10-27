"""Load-only entry motion/actuation producers and native forward remainder."""
import ctypes as ct
import warp as wp
from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from .peeled_motion import PeeledMotionSubstep


class PeeledForceSubstep(PeeledMotionSubstep):
    def __init__(self, runtime, build_root, *, force_threads=256, **options):
        if type(force_threads) is not int or force_threads not in (0, 64, 128, 256):
            raise ValueError('force threads must be 0, 64, 128 or 256')
        super().__init__(runtime, build_root, **options)
        self.motion_library = self.library
        manifest = substep_native.prepare_force(GPU_ROOT/'cuda/step/fused_substep.cu',
            build_root, self.manifest['config']['sdk'], self.stream.device.arch,
            nv=self.variant.nv, nM=self.variant.nM, nq=self.variant.nq, njnt=self.variant.njnt, nu=self.variant.nu, for_build=options.get('allow_build', False), **{k: options[k] for k in
            ('fast_compile', 'device_lto', 'atomic_reductions') if k in options})
        self.manifest = substep_native.finish(manifest, allow_build=options.get('allow_build', False))
        self.library = lib = ct.CDLL(self.manifest['library'])
        if lib.flash_libero_force_handoff_version()!=2:
            raise RuntimeError('force handoff ABI differs; rebuild offline')
        lib.flash_libero_force_arguments_size.restype = ct.c_size_t
        if (lib.flash_libero_force_arguments_size() != ct.sizeof(self.arguments) or
                lib.flash_libero_force_nv() != self.variant.nv):
            raise RuntimeError('peeled force ABI/shape differs')
        a = self.arguments
        u, s, t = a.actuation, a.smooth_acceleration, a.transmission
        self.force_eligible = bool(self.eligible and u.nu <= 32 and s.nv <= 128 and
            u.nv == s.nv == t.nv and u.nu == t.nu == a.control_copy.width and
            u.moment == t.moment and u.velocity == t.velocity and
            t.qvel == a.body_velocity.qvel and u.ctrl == a.control_copy.target and
            u.qfrc_actuator == s.actuator and a.bias_forces.nv == s.nv and
            a.forward_workspace.mass_plan)
        # A speculative producer must retain canonical ctrl and the two
        # public smooth outputs until the original residual admission point.
        # Unusual aliases keep the original native implementation.
        width = self.worlds*8
        outputs = [(u.force,u.nu),(u.force_pre,u.nu),(u.ctrl_used,u.nu),
                   (u.qfrc_actuator,s.nv)]
        outputs += [(getattr(s,k),n) for k,n in
                    (('smooth',s.nv),('seed',s.nv),('scratch',s.nv),('qfrc',s.nv),
                     ('qacc_smooth',s.nv),('ld',s.nm),('inv',s.nv),
                     ('sqrt_inv',s.nv),('rhs',4*s.nv),('solution',4*s.nv))]
        spans = [(int(p),int(p)+n*width) for p,n in outputs]
        retained = [(int(p),int(p)+n*width) for p,n in
                    ((a.control_copy.source,u.nu),(a.smooth_commit.pf,s.nv),
                     (a.smooth_commit.pa,s.nv))]
        overlap = lambda x,y: x[0] < y[1] and y[0] < x[1]
        self.force_eligible &= not any(overlap(x,y) for x in spans for y in retained)
        self.force_eligible &= not any(overlap(x,y) for i,x in enumerate(spans) for y in spans[i+1:])
        self.force_threads = force_threads if self.force_eligible and self.motion_threads else 0
        self.force_ready = wp.zeros(self.worlds, dtype=wp.int32, device=self.stream.device)
        configure = lib.flash_libero_force_configure
        configure.argtypes = [ct.c_void_p]*3
        configure.restype = ct.c_int
        regs, shared, local = ((ct.c_int*2)() for _ in range(3))
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(regs, shared, local), 'configure peeled force')
        self.launch_force = lib.flash_libero_launch_force
        self.launch_force.argtypes = [ct.c_void_p]*3+[ct.c_int]*3+[ct.c_void_p]
        self.launch_force.restype = ct.c_int
        self.launch_force_substep = lib.flash_libero_launch_force_substep
        self.launch_force_substep.argtypes = [ct.c_void_p]*4+[ct.c_int]*6+[ct.c_void_p]
        self.launch_force_substep.restype = ct.c_int
        self.resources = dict(self.resources, execution='force-residual',
            force_eligible=self.force_eligible, force_threads=self.force_threads,
            force_after_controller=True, controller_force_threads=64,
            handoff_bytes_per_world=8, force_resources={name:dict(registers=regs[i],
                static_shared_bytes=shared[i],local_bytes_per_thread=local[i])
                for i,name in enumerate(('actuation_solve','residual'))})

    def _launch(self, policy_step, entry, publish_post=True):
        self._check(self.launch_force_substep(self.device_arguments.ptr,self.plan.ptr,
            self.ready.ptr,self.force_ready.ptr,self.worlds,int(policy_step),entry,
            int(publish_post),self.motion_threads,self.force_threads,self.stream.cuda_stream),
            'peeled force substep')

    def enqueue_force(self, entry=PeeledMotionSubstep.START):
        """Local force replay after an admitted entry motion producer."""
        if not self.force_threads:
            raise ValueError('force-only replay requires an admitted producer')
        self._check(self.launch_force(self.device_arguments.ptr,self.ready.ptr,
            self.force_ready.ptr,self.worlds,entry,self.force_threads,self.stream.cuda_stream),
            'actuation/smooth region')
