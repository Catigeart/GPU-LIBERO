"""Device workspace for the complete fwdConstraint + device Newton loop.

One owner allocates every device buffer.  Frozen S04-S08 kernels and the S07
device bridge are launched against these buffers through typed aliases, so
the production hot loop never allocates, uploads, downloads or transfers
between stages.  Shapes are static: nworld worlds, nv=43, nM=153, rows/contacts
capacities fixed at prepare time.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

from pathlib import Path

import numpy as np
import warp as wp

from libero.gpu.host.runtime.newton.s07_bridge import S07Args, S07DeviceBridge

NV = 43
NM = 153
MAXITER = 100
NSOLVER = 1000
TRACECAP = 52


class NewtonFwdWorkspace:
    """Prepared buffers plus typed kernel aliases for every stage."""

    def __init__(self, nworld: int, rows: int, contacts: int, *,
                 device: str = 'cuda:0', maxiter: int = MAXITER,
                 s07_build_dir: Path | None = None,
                 s07_kernel_source: Path | None = None,
                 warmstart_enabled: bool = True):
        if nworld < 1 or rows < 1 or contacts < 1:
            raise ValueError('positive capacities required')
        limit = np.iinfo(np.int32).max
        if nworld * rows * NV > limit or nworld * contacts * 36 > limit:
            raise ValueError('flattened device index exceeds int32 capacity')
        if maxiter < 0 or maxiter > 1000000:
            raise ValueError('maxiter out of range')
        self.nworld = int(nworld)
        self.rows = int(rows)
        self.contacts = int(contacts)
        self.maxiter = int(maxiter)
        self.device = device
        self.warmstart_enabled = bool(warmstart_enabled)
        self.stream = wp.get_stream(device)
        d = device

        def f2(shape):
            return wp.zeros(shape, dtype=scalar.wp_real, device=d)

        def i32(shape):
            return wp.zeros(shape, dtype=wp.int32, device=d)

        # ---- row-outer constraint inputs (S04/S05/S07 layout) -------------
        self.aref = f2((nworld, rows))
        self.jar = f2((nworld, rows))            # Newton Jaref (loop state)
        self.jar_warm = f2((nworld, rows))       # warmstart evaluator jar
        self.efc_b = f2((nworld, rows))
        self.D = f2((nworld, rows))
        self.R = f2((nworld, rows))
        self.loss = f2((nworld, rows))
        self.efc_type = i32((nworld, rows))
        self.efc_id = i32((nworld, rows))
        self.cone_dim = i32((nworld, rows))
        self.cone_mu = f2((nworld, rows))
        self.cone_friction = f2((nworld, rows, 5))
        self.J = f2((nworld, rows, NV))
        self.force = f2((nworld, rows))
        self.state = i32((nworld, rows))
        self.oldstate = i32((nworld, rows))
        self.Jv = f2((nworld, rows))

        self.contact_H = f2((nworld, contacts, 36))
        self.contact_dim = i32((nworld, contacts))
        self.contact_mu = f2((nworld, contacts))
        self.contact_friction = f2((nworld, contacts, 5))

        # ---- nv-sized buffers ---------------------------------------------
        self.qacc = f2((nworld, NV))
        self.qacc_smooth = f2((nworld, NV))
        self.qacc_warmstart = f2((nworld, NV))
        self.qfrc_smooth = f2((nworld, NV))
        self.qfrc_constraint = f2((nworld, NV))
        self.Ma = f2((nworld, NV))
        self.Mv = f2((nworld, NV))
        self.Ma_warm = f2((nworld, NV))
        self.grad = f2((nworld, NV))
        self.Mgrad = f2((nworld, NV))
        self.candidate = f2((nworld, NV))
        self.scratch_grad = f2((nworld, NV))
        self.scratch_x = f2((nworld, NV))
        self.scratch_fwd = f2((nworld, NV))
        self.search = f2((nworld, NV))
        # S08 A reads the two qacc candidates; they are exactly the two
        # workspace candidates (aliasing keeps them filled with no transfer).
        self.warm_qacc = self.qacc_warmstart
        self.smooth_qacc = self.qacc_smooth

        # ---- Hessian scratch (S05 layout) ---------------------------------
        self.H = f2((nworld, NV, NV))
        self.Hcone = f2((nworld, NV, NV))
        self.H_local = f2((nworld, 36))
        self.H_ltj = f2((nworld, 6, NV))
        self.H_vec = f2((nworld, NV))

        # ---- topology / static --------------------------------------------
        self.qM = f2((nworld, NM))
        self.madr = wp.zeros((NV,), dtype=wp.int32, device=d)
        self.parentid = wp.zeros((NV,), dtype=wp.int32, device=d)
        self.simplenum = wp.zeros((NV,), dtype=wp.int32, device=d)
        self.model_scalars = f2((2,))            # tolerance, meaninertia

        # ---- per-world scalars --------------------------------------------
        self.nefc = i32((nworld,))
        self.ne = i32((nworld,))
        self.nf = i32((nworld,))
        self.ncon = i32((nworld,))
        self.count = i32((nworld,))              # S04's name for nefc
        self.active = i32((nworld,))
        self.empty = i32((nworld,))
        self.newton_world = i32((nworld,))
        self.status = i32((nworld,))
        self.cost = f2((nworld,))
        self.oldcost = f2((nworld,))
        self.quad0 = f2((nworld,))
        self.warm_cost = f2((nworld,))
        self.smooth_cost = f2((nworld,))
        self.choice = i32((nworld,))
        self.nactive = i32((nworld,))
        self.ncone = i32((nworld,))
        self.nupdate = i32((nworld,))
        self.nnz = i32((nworld,))
        self.route = i32((nworld,))
        self.iteration = i32((nworld,))
        self.committed = i32((nworld,))
        self.stage = i32((nworld,))
        self.solver_iter = i32((nworld,))
        self.solver_iter_base = i32((nworld,))
        self.solver_nnz = i32((nworld,))
        self.solver_exit = i32((nworld,))
        self.maxiter_arr = i32((nworld,))
        self.alpha = f2((nworld,))
        self.slope = f2((nworld,))
        self.lineslope = f2((nworld,))
        self.meaninertia = f2((nworld,))
        self.tolerance = f2((nworld,))
        self.out_nchange = i32((nworld,))
        self.out_improvement = f2((nworld,))
        self.out_gradient = f2((nworld,))
        self.out_stop = i32((nworld,))
        self.out_reason = i32((nworld,))
        self.stop_reason = i32((nworld,))
        self.err_code = i32((nworld,))
        self.error_item = i32((nworld,))
        self.solver_stats_f = f2((nworld, NSOLVER, 3))
        self.solver_stats_i = i32((nworld, NSOLVER, 4))

        self.flag = i32((nworld,))               # S04 Hessian-write flag

        self.err_base_s08b = i32((nworld,))
        self.err_base_s04 = i32((nworld,))
        self.err_base_s05 = i32((nworld,))
        self.err_base_s06 = i32((nworld,))
        self.add_to_cost = i32((nworld,))

        # ---- S07 buffers --------------------------------------------------
        self.eval_trace = f2((nworld, TRACECAP, 4))
        self.eval_tag = i32((nworld, TRACECAP))
        self.LSiter = i32((nworld,))
        self.LSresult = i32((nworld,))
        self.s07_status = i32((nworld,))
        self.quad = f2((nworld, rows, 3))
        self.quadGauss = f2((nworld, 3))

        # ---- S08 stage status ---------------------------------------------
        self.s08a_status = i32((nworld,))
        self.s08b_status = i32((nworld,))
        self.s08c_status = i32((nworld,))
        self.s08d_status = i32((nworld,))

        self.bind_topology(
            np.zeros((NV,), np.int32), np.full((NV,), -1, np.int32),
            np.zeros((NV,), np.int32), np.zeros((2,), np.float64))
        self.s07 = S07DeviceBridge(build_dir=s07_build_dir,
                                   kernel_source=s07_kernel_source, defer_compile=True)
        self._build_s07_args()
        self.s07_stream_ptr = int(self.stream.cuda_stream)
        self.reset_run_state()

    def reset_run_state(self):
        """Zero every per-run loop-state buffer (F15 reuse contract).

        Output buffers that the source writes-or-preserves (qacc, warmstart,
        force/state/H tails, solver_iter/nnz/stats, qfrc_constraint) are
        caller-owned and are NOT touched here; callers seed them as needed.
        """
        for arr in (self.iteration, self.committed, self.choice,
                    self.stop_reason, self.err_code, self.s07_status,
                    self.s08a_status, self.s08b_status, self.s08c_status,
                    self.s08d_status, self.LSiter, self.LSresult,
                    self.out_stop, self.out_reason, self.out_nchange,
                    self.solver_exit, self.nactive, self.ncone, self.nupdate,
                    self.nnz, self.route):
            arr.zero_()
        self.error_item.assign(np.full(self.nworld, -1, np.int32))
        for arr in (self.alpha, self.slope, self.cost, self.oldcost,
                    self.quad0, self.warm_cost, self.smooth_cost,
                    self.out_improvement, self.out_gradient):
            arr.zero_()

    # ------------------------------------------------------------------
    def bind_topology(self, madr, parentid, simplenum, model_scalars):
        """Prepare-time host upload of the frozen main-model topology."""
        for name, value in (('madr', madr), ('parentid', parentid),
                            ('simplenum', simplenum)):
            arr = np.ascontiguousarray(np.asarray(value))
            if arr.dtype != np.int32 or arr.shape != (NV,):
                raise TypeError((name, arr.shape, arr.dtype))
            getattr(self, name).assign(arr)
        scalars = np.ascontiguousarray(np.asarray(model_scalars))
        if scalars.dtype != np.float64 or scalars.shape != (2,):
            raise TypeError(('model_scalars', scalars.shape, scalars.dtype))
        scalars = scalars.copy()
        scalars[0] = scalar.solver_tolerance(scalars[0])
        self.model_scalars.assign(scalars)
        from libero.gpu.model.topology import build_packed_mass_plan
        try:
            plan=build_packed_mass_plan(np.asarray(madr),np.asarray(parentid),np.asarray(simplenum),NV,NM)
        except ValueError:
            plan=None
        for name,data in zip(('mass_plan_offsets','mass_plan_coefficient','mass_plan_source','mass_plan_assign'),plan or (None,)*4):
            setattr(self,name,None if data is None else wp.array(data,dtype=wp.int32,device=self.device))

        for name,data in (('mass_plan_madr_snapshot',madr),('mass_plan_parent_snapshot',parentid),('mass_plan_simple_snapshot',simplenum)):
            setattr(self,name,wp.array(np.asarray(data),dtype=wp.int32,device=self.device))

    def _build_s07_args(self):
        a = self
        self._s07_args = S07Args(**{
            'search': a.search.ptr,
            'Ma': a.Ma.ptr,
            'Jaref': a.jar.ptr,
            'quad0': a.quad0.ptr,
            'smooth': a.qfrc_smooth.ptr,
            'qM': a.qM.ptr,
            'J': a.J.ptr,
            'D': a.D.ptr,
            'R': a.R.ptr,
            'loss': a.loss.ptr,
            'cone_mu': a.cone_mu.ptr,
            'cone_friction': a.cone_friction.ptr,
            'efc_type': a.efc_type.ptr,
            'efc_id': a.efc_id.ptr,
            'cone_dim': a.cone_dim.ptr,
            'madr': a.madr.ptr,
            'parent': a.parentid.ptr,
            'simplenum': a.simplenum.ptr,
            'nefc': a.nefc.ptr,
            'ne': a.ne.ptr,
            'nf': a.nf.ptr,
            'active': a.active.ptr,
            'model_scalars': a.model_scalars.ptr,
            'Mv': a.Mv.ptr,
            'Jv': a.Jv.ptr,
            'quadGauss': a.quadGauss.ptr,
            'quad': a.quad.ptr,
            'alpha': a.alpha.ptr,
            'slope': a.slope.ptr,
            'eval_trace': a.eval_trace.ptr,
            'LSiter': a.LSiter.ptr,
            'LSresult': a.LSresult.ptr,
            'eval_tag': a.eval_tag.ptr,
            'status': a.s07_status.ptr,
        })

    def launch_s07(self, tracecap: int = TRACECAP):
        error = self.s07.launch(self._s07_args, self.nworld, NV, NM,
                                self.rows, tracecap, self.s07_stream_ptr)
        if error:
            raise RuntimeError(f'S07 device launch error {error}')
