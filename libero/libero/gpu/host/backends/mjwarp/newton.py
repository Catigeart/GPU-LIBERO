# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned Newton host orchestration with explicit CUDA stage substitutions.

Adapted from solver._solve/_solver_iteration/_update_gradient at MJWarp commit
c384ac51df67eb92ab7657cc472ff7494dc036bf. The cumulative cuda_newton path
replaces the admitted numerical stages; allocation and conditional Graphs
remain upstream. No monkey patch, generated CUDA, altered warmstart publication
or persistent reuse of per-solve scratch is involved.
"""
from contextlib import contextmanager

import warp as wp
from mujoco_warp._src import solver, types

from libero.gpu.host.runtime.newton.mjwarp_hessian import HessianPlan
from libero.gpu.host.runtime.newton.mjwarp_direction import DirectionPlan, unsupported_reason as direction_reason
from libero.gpu.host.runtime.newton.mjwarp_constraint import ConstraintPlan
from libero.gpu.host.runtime.newton.mjwarp_solver_state import SolverStatePlan
from libero.gpu.host.runtime.newton.mjwarp_linesearch import LinesearchPlan, unsupported_reason as linesearch_reason


def hessian_product(root, *, fast_compile=False, component='hessian'):
    """Locate an offline product; the numerical binding validates its ABI."""
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    from libero.gpu.host.build.project_paths import project_paths
    folder = project_paths(root).cmake / ('sm80-dev' if fast_compile else 'sm80-release')
    if component not in ('hessian', 'direction', 'constraint', 'linesearch', 'solver_state'):
        raise ValueError('unknown Newton component')
    library = folder / ('mjwarp_' + component + '_fp32.so')
    target = 'flash_mjwarp_' + component + '_fp32'
    if not library.is_file():
        raise MissingCompiledProduct('Build ' + target + ' offline in ' + str(folder))
    return library


def unsupported_reason(model, data):
    """Domains outside the CUDA leaf retain the complete upstream solver."""
    if model.opt.solver != types.SolverType.NEWTON:
        return 'solver is not Newton'
    if model.is_sparse or model.opt.cone != types.ConeType.ELLIPTIC:
        return 'requires dense elliptic constraints'
    if model.opt.integrator == types.IntegratorType.DISCRETE:
        return 'discrete effective metric'
    if model.opt.enableflags & types.EnableBit.SLEEP:
        return 'active DOF compaction'
    if model.nv == 0 or data.njmax == 0:
        return 'empty solver dimension'
    expected = ((model.nv + 3) // 4) * 4 if model.nv <= 32 else ((model.nv + 15) // 16) * 16
    if model.nv_pad != expected or model.nv_pad not in (24, 28, 32, 48, 64):
        return 'unqualified padded NV'
    return None


class Newton:
    """Explicit cumulative CUDA substitutions with visible fallback metadata.

    Static CUDA preparation happens in __init__, outside capture. Each solve
    creates fresh upstream scratch. Captures retain every context/binding in
    an ownership list returned by capture_owners(); callers retain that list
    together with the graph, rather than extending a process-global cache.
    """

    def __init__(self, model, data, stream, *, implementation='upstream', library=None,
                 direction_library=None, constraint_library=None, linesearch_library=None, state_library=None):
        if implementation not in ('upstream', 'cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton', 'staged_upstream'):
            raise ValueError('invalid Newton implementation')
        self.model, self.data, self.stream = model, data, stream
        self.requested = implementation
        self.fallback = unsupported_reason(model, data) if implementation != 'upstream' else None
        if implementation in ('cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') and self.fallback is None:
            self.fallback = direction_reason(model)
        if implementation in ('cuda_linesearch', 'cuda_newton') and self.fallback is None:
            self.fallback = linesearch_reason(model)
        self.selected = 'upstream' if self.fallback else implementation
        self.plan = (HessianPlan(model, data, library, stream)
                     if self.selected in ('cuda_hessian', 'cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') else None)
        self.direction_plan = (DirectionPlan(model, data, direction_library, stream)
                               if self.selected in ('cuda_direction', 'cuda_constraint', 'cuda_linesearch', 'cuda_newton') else None)
        self.constraint_plan = (ConstraintPlan(model, data, constraint_library, stream)
                                if self.selected in ('cuda_constraint', 'cuda_linesearch', 'cuda_newton') else None)
        self.linesearch_plan = (LinesearchPlan(model, data, linesearch_library, stream)
                                if self.selected in ('cuda_linesearch', 'cuda_newton') else None)
        self.state_plan = (SolverStatePlan(model, data, state_library, stream)
                           if self.selected == 'cuda_newton' else None)
        self._capture_owners = None

    @property
    def identity(self):
        result = dict(requested=self.requested, selected=self.selected, fallback=self.fallback)
        if self.plan is not None:
            import hashlib
            from libero.gpu.host.runtime.newton.mjwarp_hessian import source_id
            result.update(hessian_source_id=source_id(),
                hessian_library_sha256=hashlib.sha256(self.plan.library_path.read_bytes()).hexdigest())
        if self.direction_plan is not None:
            import hashlib
            from libero.gpu.host.runtime.newton.mjwarp_direction import source_id
            result.update(direction_source_id=source_id(),
                direction_library_sha256=hashlib.sha256(self.direction_plan.library_path.read_bytes()).hexdigest(),
                handwritten=['gradient', 'hessian', 'factor_direction'],
                upstream=['initialization', 'constraint_update', 'linesearch', 'stopping', 'noslip'])
        if self.constraint_plan is not None:
            import hashlib
            from libero.gpu.host.runtime.newton.mjwarp_constraint import source_id
            result.update(constraint_source_id=source_id(),
                constraint_library_sha256=hashlib.sha256(self.constraint_plan.library_path.read_bytes()).hexdigest(),
                handwritten=['constraint_update', 'constraint_force', 'gradient', 'hessian', 'factor_direction'],
                upstream=['initialization_prefix', 'linesearch', 'stopping', 'noslip'])
        if self.linesearch_plan is not None:
            import hashlib
            from libero.gpu.host.runtime.newton.mjwarp_linesearch import source_id
            result.update(linesearch_source_id=source_id(),
                linesearch_library_sha256=hashlib.sha256(self.linesearch_plan.library_path.read_bytes()).hexdigest(),
                handwritten=['constraint_update', 'constraint_force', 'gradient', 'hessian',
                             'factor_direction', 'linesearch_products', 'linesearch'],
                upstream=['initialization_prefix', 'stopping', 'noslip'])
        if self.state_plan is not None:
            import hashlib
            from libero.gpu.host.runtime.newton.mjwarp_solver_state import source_id
            result.update(state_source_id=source_id(),
                state_library_sha256=hashlib.sha256(self.state_plan.library_path.read_bytes()).hexdigest(),
                handwritten=['initialization', 'constraint_update', 'constraint_force', 'gradient',
                             'hessian', 'factor_direction', 'linesearch_products', 'linesearch', 'stopping'],
                upstream=['optional_noslip'], infrastructure=['context_allocation', 'conditional_graph'])
        return result

    @contextmanager
    def capture_owners(self):
        if self._capture_owners is not None:
            raise RuntimeError('nested Newton ownership capture')
        owners = []
        self._capture_owners = owners
        try:
            yield owners
        finally:
            self._capture_owners = None

    def _gradient(self, ctx, assembly, direction=None):
        m, d = self.model, self.data
        if assembly is None:  # validation control: qualify host dispatch alone
            solver._update_gradient(m, d, ctx, compact=False)
            return
        if direction is not None:
            direction.enqueue(stage=1)
        else:
            wp.launch(solver._update_gradient_zero_grad_dot(False), dim=d.nworld,
                inputs=[d.nefc, ctx.alpha, ctx.done],
                outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged])
            wp.launch(solver._update_gradient_grad(False), dim=(d.nworld, m.nv),
                inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done],
                outputs=[ctx.grad, ctx.grad_dot])
        assembly.enqueue()
        if direction is not None:
            direction.enqueue(stage=2)
        else:
            solver._cholesky_factorize_solve(m, d, ctx)

    def _constraint(self, ctx, constraint):
        if constraint is None:
            solver._update_constraint(self.model, self.data, ctx, track_changes=False, stable_fast=False)
        else:
            constraint.enqueue()

    def _initialize_context(self, ctx, constraint):
        """Pinned dense/full-DOF init prefix, with exactly one constraint call.

        Only the host enqueue order is adapted from solver.init_context.
        Reset/Jaref/Ma mathematics stay upstream. The selected leaf is used
        here as well as after every line search, without an upstream duplicate.
        """
        m, d = self.model, self.data
        if constraint is None:
            solver.init_context(m, d, ctx, grad=False, compact=False)
            return
        wp.launch(solver._solve_init_efc, dim=d.nworld,
            outputs=[d.solver_niter, ctx.search_dot, ctx.done])
        dofs_per_thread = 20 if m.nv > 50 else 50
        threads_per_row = (m.nv + dofs_per_thread - 1) // dofs_per_thread
        if threads_per_row > 1:
            ctx.Jaref.zero_()
        wp.launch(solver._solve_init_jaref_kernel(False, m.nv, dofs_per_thread, False),
            dim=(d.nworld, d.njmax, threads_per_row),
            inputs=[d.nefc, d.qacc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind,
                    d.efc.J, d.efc.aref, d.dof_cdof], outputs=[ctx.Jaref])
        solver._mul_m_compact_aware(m, d, ctx, d.efc.Ma, d.qacc, ctx.done)
        self._constraint(ctx, constraint)

    def _iteration(self, ctx, assembly, nsolving, direction=None, constraint=None, linesearch=None, solver_state=None):
        m, d = self.model, self.data
        if linesearch is None:
            solver._linesearch(m, d, ctx)
        else:
            linesearch.enqueue()
        # The admitted elliptic domain never uses incremental/stable-fast H.
        self._constraint(ctx, constraint)
        self._gradient(ctx, assembly, direction)
        if solver_state is not None:
            solver_state.stop()
        else:
            wp.launch(solver._solve_done(int(m.opt.warn_overflow)), dim=d.nworld,
                inputs=[m.nv, m.opt.tolerance, m.opt.iterations, m.stat.meaninertia,
                    ctx.grad_dot, ctx.newton_decrement, ctx.improvement, ctx.alpha, ctx.done],
                outputs=[d.solver_niter, d.overflow, nsolving, ctx.done])

    def solve(self):
        m, d = self.model, self.data
        if self.selected == 'upstream':
            with wp.ScopedStream(self.stream):
                solver.solve(m, d)
            return None
        with wp.ScopedStream(self.stream):
            # Same allocation/zeroing semantics as upstream solver.solve.
            ctx = solver._create_solver_context(m, d)
            assembly = self.plan.bind(ctx) if self.plan is not None else None
            direction = self.direction_plan.bind(ctx) if self.direction_plan is not None else None
            constraint = self.constraint_plan.bind(ctx) if self.constraint_plan is not None else None
            linesearch = self.linesearch_plan.bind(ctx) if self.linesearch_plan is not None else None
            state = self.state_plan.bind(ctx) if self.state_plan is not None else None
            if state is not None:
                state.initialize()
                self._constraint(ctx, constraint)
            else:
                warmstart = not (m.opt.disableflags & types.DisableBit.WARMSTART)
                wp.launch(solver._solve_init_dof(warmstart, False), dim=(d.nworld, m.nv),
                    inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
                    outputs=[d.qacc, d.qfrc_constraint])
                self._initialize_context(ctx, constraint)
            self._gradient(ctx, assembly, direction)
            nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)
            if state is not None:
                state.bind_stopping(nsolving)
            owner = (ctx, assembly, nsolving, direction, constraint, linesearch, state)
            if self._capture_owners is not None:
                self._capture_owners.append(owner)
            if m.opt.iterations != 0 and m.opt.graph_conditional:
                wp.capture_while(nsolving, while_body=self._iteration,
                    ctx=ctx, assembly=assembly, nsolving=nsolving, direction=direction, constraint=constraint, linesearch=linesearch, solver_state=state)
            else:
                for _ in range(m.opt.iterations):
                    self._iteration(ctx, assembly, nsolving, direction, constraint, linesearch, state)
            if m.opt.noslip_iterations > 0:
                solver.noslip(m, d)
            return owner
