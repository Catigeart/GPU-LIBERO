"""One physical substep per CUDA kernel, with explicit prefix continuations."""
import libero.gpu.precision as scalar
import ctypes as ct

import numpy as np
import warp as wp

from libero.gpu.host.build import substep_native
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.substep_catalog import select_variant
from .substep_binding import bind
from .newton_binding import (NewtonArguments, NewtonWorldData, NewtonModelData, NewtonPostData,
                             device_workspace_bytes)


class FusedSubstep:
    START = 0
    CONTROLLER_POSITION = 1
    PHYSICS_POSITION = 2
    ACCELERATION_RESET = 3
    POSITION_CHECKS = 5

    def __init__(self, runtime, build_root, *, allow_build=False,
                 fast_compile=False, device_lto=False, profile_newton=False, atomic_reductions=None,
                 single_solve=False):
        if type(single_solve) is not bool or single_solve and scalar.bits != 32:
            raise ValueError('single_solve requires an explicit FP32 candidate')
        self.single_solve = single_solve
        self.runtime = runtime
        self.stream = runtime.stream
        self.worlds = runtime.euler.capacity
        if self.stream.device.arch != 80:
            raise ValueError("the complete substep reference/controller path currently targets SM80")
        euler = runtime.euler
        self.variant = select_variant(euler.nv,euler.nM,nq=euler.nq,njnt=euler.njnt,
            nu=euler.ctrl.shape[1],rows=runtime.forward.n.ws.rows,contacts=runtime.forward.n.ws.contacts)
        self.continuation = wp.zeros(self.worlds, dtype=wp.int32, device=self.stream.device)
        self.predicate = wp.zeros(1, dtype=wp.int32, device=self.stream.device)
        self.arguments, self.owners, core = bind(runtime, self.continuation, single_solve=single_solve)
        data = np.frombuffer(bytes(self.arguments), dtype=np.uint8).copy()
        self.device_arguments = wp.array(data, dtype=wp.uint8, device=self.stream.device)
        self.manifest = substep_native.prepare(GPU_ROOT / "cuda/step/fused_substep.cu",
            build_root, core.manifest["mathdx_root"], self.stream.device.arch,
            fast_compile=fast_compile, device_lto=device_lto, profile_newton=profile_newton,
            nv=self.variant.nv, atomic_reductions=atomic_reductions, single_solve=single_solve,
            for_build=allow_build)
        self._graphs = {}
        self.load(allow_build=allow_build)

    def load(self, *, allow_build=False):
        self.manifest = substep_native.finish(self.manifest, allow_build=allow_build)
        self.library = ct.CDLL(self.manifest["library"])
        scalar_size = self.library.flash_libero_scalar_bytes
        scalar_size.argtypes = []; scalar_size.restype = ct.c_int
        if scalar_size() != scalar.itemsize:
            raise RuntimeError('loaded substep precision differs from its buffer owners')
        prefix = 'flash_libero_substep_single' if self.single_solve else 'flash_libero_substep'
        solve_count = getattr(self.library, prefix + '_solves_per_substep')
        solve_count.argtypes = []; solve_count.restype = ct.c_int
        if solve_count() != self.manifest['config']['solves_per_substep']:
            raise RuntimeError('loaded substep uses a different physical schedule')
        for suffix, kind in (("arguments", NewtonArguments), ("world", NewtonWorldData),
                             ("model", NewtonModelData), ("post", NewtonPostData)):
            try:
                newton_size = getattr(self.library, "flash_libero_newton_" + suffix + "_size")
            except AttributeError as error:
                raise RuntimeError("single-substep Newton ABI requires an offline rebuild") from error
            newton_size.argtypes = []
            newton_size.restype = ct.c_size_t
            if newton_size() != ct.sizeof(kind):
                raise RuntimeError("single-substep Newton layout differs: " + suffix)
        size = self.library.flash_libero_substep_arguments_size
        size.argtypes = []
        size.restype = ct.c_size_t
        if size() != ct.sizeof(self.arguments):
            raise RuntimeError("single-substep Python/CUDA argument layout differs")
        dimension = self.library.flash_libero_substep_dimension
        dimension.argtypes = [ct.c_int]
        dimension.restype = ct.c_int
        v = self.variant
        expected = (v.nv,v.nq,v.nM,v.njnt,v.nu,v.rows,v.contacts)
        if tuple(dimension(i) for i in range(7)) != expected:
            raise RuntimeError("loaded substep library has different model dimensions")
        try:
            reduction_mode = self.library.flash_libero_newton_atomic_reductions
            workspace_size = self.library.flash_libero_newton_device_workspace_bytes
        except AttributeError as error:
            raise RuntimeError("Newton reduction/workspace ABI requires an offline rebuild") from error
        reduction_mode.argtypes = []; reduction_mode.restype = ct.c_int
        workspace_size.argtypes = []; workspace_size.restype = ct.c_size_t
        if bool(reduction_mode()) != self.manifest['config']['atomic_reductions']:
            raise RuntimeError("loaded Newton reduction mode differs from the requested build")
        if workspace_size() != device_workspace_bytes(v.nv, v.rows):
            raise RuntimeError("loaded Newton device workspace pitch differs")
        configure = getattr(self.library, prefix + '_configure')
        configure.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        configure.restype = ct.c_int
        registers, shared = ct.c_int(), ct.c_int()
        with wp.ScopedDevice(self.stream.device):
            self._check(configure(ct.byref(registers), ct.byref(shared)), "configure")
        shared_bytes = self.library.flash_libero_substep_shared_bytes
        shared_bytes.argtypes = []
        shared_bytes.restype = ct.c_size_t
        self.resources = dict(real_bits=scalar.bits, registers=registers.value, static_shared_bytes=shared.value,
                              schedule=self.manifest['config']['schedule'],
                              solves_per_substep=solve_count(),
                              dynamic_shared_bytes=shared_bytes(), threads_per_world=dimension(7),
                              nv=v.nv,hessian_tile_rows=v.hessian_tile_rows,
                              atomic_reductions=bool(reduction_mode()),
                              newton_device_workspace_bytes_per_world=workspace_size())
        launch_prefix = 'flash_libero_launch_substep' + ('_single' if self.single_solve else '')
        self.launch_entry = getattr(self.library, launch_prefix)
        self.launch_entry.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p]
        self.launch_entry.restype = ct.c_int
        self.launch_post_entry = getattr(self.library, launch_prefix + '_with_post', None)
        if self.launch_post_entry is not None:
            self.launch_post_entry.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p]
            self.launch_post_entry.restype = ct.c_int
        self.select_entry = self.library.flash_libero_select_substep_continuation
        self.select_entry.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        self.select_entry.restype = ct.c_int

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(("single-substep CUDA " + operation, status))

    def _launch(self, policy_step, entry, publish_post=True):
        if publish_post:
            status = self.launch_entry(self.device_arguments.ptr, self.worlds,
                int(policy_step), entry, self.stream.cuda_stream)
        else:
            if self.launch_post_entry is None:
                raise RuntimeError('Post boundary scheduling requires an offline substep rebuild')
            status = self.launch_post_entry(self.device_arguments.ptr, self.worlds,
                int(policy_step), entry, 0, self.stream.cuda_stream)
        self._check(status, 'launch')

    def _select(self, entry):
        self._check(self.select_entry(self.device_arguments.ptr, entry, self.worlds,
            self.predicate.ptr, self.stream.cuda_stream), "select continuation")

    def _refresh_position(self):
        """Rebuild only selected worlds through the existing parallel prefix."""
        runtime = self.runtime
        preservation = getattr(runtime, "_reset_preserver", None)
        if preservation is not None:
            preservation.save(self.stream)
        forward = runtime.forward
        forward._reset_preservation_scope = True
        try:
            forward._dataflow_reuse.miss_body()
        finally:
            forward._reset_preservation_scope = False
        if preservation is not None:
            preservation.restore(self.stream)

    def _enqueue(self, policy_step, publish_post=True):
        """Capture-time schedule. All conditions below execute on the device."""
        runtime = self.runtime
        forward = runtime.forward
        runtime._recovery_validate()

        if self.single_solve:
            self._launch(policy_step, self.POSITION_CHECKS, publish_post)

        # The first forward's admission and position/collision prefix remain
        # the accepted multi-CTA implementation. Its suffix starts in CUDA.
        wp.copy(forward.mask, runtime.intent, stream=self.stream)
        forward.f.begin_forward_with_plan(forward.mask, stream=self.stream,
                                          plan=forward.expected_forward)
        forward._dataflow_reuse.enqueue()
        self._launch(policy_step, self.START, publish_post)

        # There are exactly two possible position-refresh suspension points.
        # Earlier phases and their Newton/warmstart writes are never replayed.
        def resume_controller():
            self._refresh_position()
            self._launch(policy_step, self.CONTROLLER_POSITION, publish_post)

        if not self.single_solve:
            self._select(self.CONTROLLER_POSITION)
            wp.capture_if(self.predicate, on_true=resume_controller, stream=self.stream)

        def resume_physics():
            self._refresh_position()
            self._launch(policy_step, self.PHYSICS_POSITION, publish_post)

        if not self.single_solve:
            self._select(self.PHYSICS_POSITION)
            wp.capture_if(self.predicate, on_true=resume_physics, stream=self.stream)

        # mj_checkAcc requires a real reset-only full forward before Euler.
        # The original callback owns mixed-world save/restore and Post work.
        def resume_after_acceleration_reset():
            runtime._reset_forward()
            self._launch(policy_step, self.ACCELERATION_RESET, publish_post)

        self._select(self.ACCELERATION_RESET)
        wp.capture_if(self.predicate, on_true=resume_after_acceleration_reset, stream=self.stream)
        runtime.observe()

    def enqueue_substep(self, *, policy_step=False, publish_post=True):
        """Standalone substeps publish forces/sensors; a qualified batch may defer them."""
        if self.stream.is_capturing:
            self._enqueue(bool(policy_step), bool(publish_post))
            return
        key = bool(policy_step), bool(publish_post)
        if key not in self._graphs:
            with wp.ScopedCapture(stream=self.stream) as capture:
                self._enqueue(*key)
            self._graphs[key] = capture.graph
        wp.capture_launch(self._graphs[key], stream=self.stream)
