"""Complete fwdConstraint and device-resident Newton loop orchestration.

Source order (MuJoCo 2.3.7, SHA-bound engine_forward.c/engine_solver.c):
  engine_forward.c:388-506   warmstart selection, empty fast path, efc_b
  engine_solver.c:1499-1622  mj_solCGNewton init + while loop + exit
  engine_solver.c:750-835    CGallocate / CGupdateConstraint / CGupdateGradient

The Newton loop is a pre-fixed masked schedule of `maxiter` (production 100)
iterations: the host never reads alpha/count/stop to steer the loop.  Gates
replicate the source breaks exactly: alpha==0 exits before commit/stats/
iter++; the tolerance stop (reason 1) exits BEFORE the direction write while
the maxiter while-guard exit (reason 2) happens AFTER it, preserving the
source's observable direction write set.

Stages are the frozen S04-S08 kernels plus source-order glue kernels from
this package.  No CPU intermediate value is fed into the device chain.
run_batch() ends with a stream synchronization and returns a host-visible
status summary (R01-F20); the loop body itself performs no device->host
read.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from libero.gpu.host.runtime.newton import kernels as K
from libero.gpu.reference.constraints.constraint_update import kernel as s04
from libero.gpu.reference.newton.newton_hessian import kernel as s05
from libero.gpu.reference.newton.newton_direction import kernel as s06
from libero.gpu.reference.newton.newton_state import kernel as s08
from libero.gpu.host.runtime.newton.workspace import NewtonFwdWorkspace, NV, NM, TRACECAP

# stop_reason codes recorded on device
REASON_ALPHA0 = 10
REASON_TOL = 1   # S08 out_reason
REASON_MAXITER = 2
TAPE_SLOT_CAP = 256


def _f64_1(arr):
    return wp.array(ptr=arr.ptr, dtype=scalar.wp_real, shape=(arr.size,),
                    device=arr.device)


def _i32_1(arr):
    return wp.array(ptr=arr.ptr, dtype=wp.int32, shape=(arr.size,),
                    device=arr.device)


class FwdConstraintSolver:
    """One prepared workspace plus the complete stage schedule."""

    def __init__(self, workspace: NewtonFwdWorkspace, *, diagnostic_profile=None):
        self.ws = workspace
        w = workspace
        # typed aliases for frozen flat-indexed kernel signatures
        self._J_flat = _f64_1(w.J)
        self._aref_flat = _f64_1(w.aref)
        self._qacc_flat = _f64_1(w.qacc)
        self._qacc_smooth_flat = _f64_1(w.qacc_smooth)
        self._qacc_warmstart_flat = _f64_1(w.qacc_warmstart)
        self._D_flat = _f64_1(w.D)
        self._R_flat = _f64_1(w.R)
        self._loss_flat = _f64_1(w.loss)
        self._type_flat = _i32_1(w.efc_type)
        self._id_flat = _i32_1(w.efc_id)
        self._mu_flat = _f64_1(w.contact_mu)
        self._fr_flat = _f64_1(w.contact_friction)
        self._dim_flat = _i32_1(w.contact_dim)
        self._cone_dim_flat = _i32_1(w.cone_dim)
        self._cone_mu_flat = _f64_1(w.cone_mu)
        self._cone_fr_flat = _f64_1(w.cone_friction)
        self._force_flat = _f64_1(w.force)
        self._state_flat = _i32_1(w.state)
        self._oldstate_flat = _i32_1(w.oldstate)
        self._H_contact_flat = _f64_1(w.contact_H)
        self._qfrc_flat = _f64_1(w.qfrc_constraint)
        self._jar_flat = _f64_1(w.jar)
        self._jar_warm_flat = _f64_1(w.jar_warm)
        self._efc_b_flat = _f64_1(w.efc_b)
        self._Jv_flat = _f64_1(w.Jv)
        self._qM_flat = _f64_1(w.qM)
        self._H_flat = _f64_1(w.H)
        self._Hcone_flat = _f64_1(w.Hcone)
        self._H_local_flat = _f64_1(w.H_local)
        self._H_ltj_flat = _f64_1(w.H_ltj)
        self._H_vec_flat = _f64_1(w.H_vec)

        self._kind_direct = wp.full((w.nworld,), 1, dtype=wp.int32,
                                    device=w.device)
        self._kind_incr = wp.full((w.nworld,), 3, dtype=wp.int32,
                                  device=w.device)
        self._flag_on = wp.full((w.nworld,), 1, dtype=wp.int32, device=w.device)
        self._flag_off = wp.zeros((w.nworld,), dtype=wp.int32, device=w.device)
        self._add_cost_on = wp.full((w.nworld,), 1, dtype=wp.int32,
                                    device=w.device)
        self._err_base_s08b = wp.full((w.nworld,), 2000, dtype=wp.int32,
                                      device=w.device)
        self._err_base_s04 = wp.full((w.nworld,), 4000, dtype=wp.int32,
                                     device=w.device)
        self._err_base_s05 = wp.full((w.nworld,), 5000, dtype=wp.int32,
                                     device=w.device)
        self._err_base_s06 = wp.full((w.nworld,), 6000, dtype=wp.int32,
                                     device=w.device)
        self._err_base_s08d = wp.full((w.nworld,), 7000, dtype=wp.int32,
                                      device=w.device)
        self._finalize_active = wp.zeros((w.nworld,), dtype=wp.int32,
                                         device=w.device)
        # witness tape (device-resident; downloaded once outside the loop).
        # Bounded slots: min(maxiter, TAPE_SLOT_CAP) keeps the allocation
        # finite for pathological maxiter values (R01-F21).
        n = w.nworld
        profile=diagnostic_profile or getattr(w,'diagnostic_profile','compatible')
        if profile=='compatibility':profile='compatible'
        if profile not in ('compatible','fast'):raise ValueError('Newton diagnostic profile')
        self.diagnostic_profile=profile
        self.tape_slots = min(w.maxiter, TAPE_SLOT_CAP) if profile=='compatible' else 0
        m = self.tape_slots
        self.tape_f = wp.zeros((n, m, 8), dtype=scalar.wp_real, device=w.device)
        self.tape_i = wp.zeros((n, m, 12), dtype=wp.int32, device=w.device)
        self.tape_status = wp.zeros((n, m, 6), dtype=wp.int32, device=w.device)
        self.tape_eval = wp.zeros((n, m, TRACECAP, 4), dtype=scalar.wp_real,
                                  device=w.device)
        self.tape_tag = wp.zeros((n, m, TRACECAP), dtype=wp.int32,
                                 device=w.device)
        # Typed device snapshots of the actual S07 buffers. The host never
        # supplies an oracle value to these producer/consumer witnesses.
        self.tape_Mv = wp.zeros((n, m, NV), dtype=scalar.wp_real, device=w.device)
        self.tape_Jv = wp.zeros((n, m, w.rows), dtype=scalar.wp_real, device=w.device)
        self.tape_quad = wp.zeros((n, m, w.rows, 3), dtype=scalar.wp_real, device=w.device)
        self.tape_quadGauss = wp.zeros((n, m, 3), dtype=scalar.wp_real, device=w.device)
        self.tape_ls_quad0_input = wp.zeros((n, m), dtype=scalar.wp_real, device=w.device)

    # ================= glue launchers ====================================
    def _glue(self, kernel, inputs, dim=None):
        ws = self.ws
        wp.launch(kernel, dim or ws.nworld, inputs=inputs, device=ws.device,
                  stream=ws.stream)

    def _jacvec_sub(self, active, qacc, out):
        """out = J*qacc - aref (aref is the single workspace aref buffer)."""
        ws = self.ws
        flat = {ws.qacc.ptr: self._qacc_flat,
                ws.qacc_smooth.ptr: self._qacc_smooth_flat,
                ws.qacc_warmstart.ptr: self._qacc_warmstart_flat}
        self._glue(K._jacvec_sub,
                   [active, ws.nefc, self._J_flat, flat[qacc.ptr], self._aref_flat, out,
                    ws.rows])

    def _mulM(self, active, qacc, out):
        ws = self.ws
        self._glue(K._mulM,
                   [active, qacc, self._qM_flat, ws.madr, ws.parentid,
                    ws.simplenum, out])

    def _gauss(self, active, Ma, qacc, cost, add_flag):
        ws = self.ws
        self._glue(K._gauss_compute,
                   [active, Ma, ws.qfrc_smooth, qacc, ws.qacc_smooth, cost,
                    ws.quad0, add_flag])

    def _count(self, active):
        ws = self.ws
        self._glue(K._count_active_cone,
                   [active, self._state_flat, ws.nefc, ws.nactive, ws.ncone,
                    ws.rows])

    # ================= frozen stage launchers ============================
    def _s04_update(self, jar, flag):
        ws = self.ws
        wp.launch(s04._update, ws.nworld,
                  inputs=[ws.active, ws.count, ws.ne, ws.nf, ws.ncon, flag,
                          jar, self._D_flat, self._R_flat, self._loss_flat,
                          self._type_flat, self._id_flat, self._J_flat,
                          self._mu_flat, self._fr_flat, self._dim_flat,
                          self._force_flat, self._state_flat,
                          self._H_contact_flat, ws.cost, ws.status,
                          ws.rows, ws.contacts, NV],
                  device=ws.device, stream=ws.stream)

    def _s04_transpose(self):
        ws = self.ws
        wp.launch(s04._transpose, ws.nworld * NV,
                  inputs=[ws.active, ws.count, self._J_flat, self._force_flat,
                          self._qfrc_flat, ws.status, ws.rows, NV],
                  device=ws.device, stream=ws.stream)

    def _s05_launch(self, kind_arr):
        ws = self.ws
        wp.launch(s05.validate, ws.nworld,
                  inputs=[ws.count, ws.ncon, ws.active, kind_arr,
                          self._state_flat, self._oldstate_flat,
                          self._D_flat, self._J_flat, self._qM_flat,
                          self._H_flat, self._id_flat, self._dim_flat,
                          self._H_contact_flat, ws.ncone, ws.nnz,
                          ws.status, ws.route, ws.rows, ws.contacts],
                  device=ws.device, stream=ws.stream)
        wp.launch(s05.assemble, ws.nworld * NV * NV,
                  inputs=[ws.count, ws.active, kind_arr, self._J_flat,
                          self._D_flat, self._state_flat, self._qM_flat,
                          ws.madr, ws.parentid, ws.simplenum, self._H_flat,
                          ws.status, ws.rows],
                  device=ws.device, stream=ws.stream)
        wp.launch(s05.factor_update, ws.nworld,
                  inputs=[ws.count, ws.ncon, ws.active, kind_arr,
                          self._J_flat, self._D_flat, self._state_flat,
                          self._oldstate_flat, self._qM_flat, ws.madr,
                          ws.parentid, ws.simplenum, self._id_flat,
                          self._dim_flat, self._H_contact_flat, ws.ncone,
                          self._H_flat, self._Hcone_flat, ws.nnz, ws.nupdate,
                          self._H_local_flat, self._H_ltj_flat,
                          self._H_vec_flat, ws.status, ws.route,
                          ws.rows, ws.contacts],
                  device=ws.device, stream=ws.stream)

    def _s06_launch(self, upstream_status):
        ws = self.ws
        wp.launch(s06._direction, ws.nworld,
                  inputs=[ws.Ma, ws.qfrc_smooth, ws.qfrc_constraint,
                          ws.H, ws.Hcone, ws.ncone, ws.nefc, ws.active,
                          upstream_status],
                  outputs=[ws.grad, ws.Mgrad, ws.candidate, ws.status,
                           ws.scratch_grad, ws.scratch_x, ws.scratch_fwd],
                  device=ws.device, stream=ws.stream)

    def _s08a(self):
        ws = self.ws
        wp.launch(s08._warm_select, ws.nworld,
                  inputs=[ws.newton_world, ws.nefc, ws.warm_cost,
                          ws.smooth_cost, ws.warm_qacc, ws.smooth_qacc,
                          ws.force, ws.state, ws.qfrc_constraint,
                          ws.qacc, ws.force, ws.state, ws.qfrc_constraint,
                          ws.choice, ws.s08a_status, NV, ws.rows],
                  device=ws.device, stream=ws.stream)

    def _s08b(self):
        ws = self.ws
        wp.launch(s08._commit_step, ws.nworld,
                  inputs=[ws.active, ws.nefc, ws.alpha, ws.search, ws.Mv,
                          ws.Jv, ws.qacc, ws.Ma, ws.jar,
                          ws.committed, ws.s08b_status, NV, ws.rows],
                  device=ws.device, stream=ws.stream)

    def _s08c(self):
        ws = self.ws
        # out_iter is aliased to iteration: S08 C reads iteration[w] then
        # writes out_iter[w]=it+1 into the same slot (single thread per world).
        # lineslope is the S07 slope output (ctx.LSslope), same buffer.
        wp.launch(s08._stats_stop, ws.nworld,
                  inputs=[ws.active, ws.nefc, ws.stage, ws.committed,
                          ws.oldstate, ws.state, ws.grad,
                          ws.oldcost, ws.cost, ws.meaninertia, ws.tolerance,
                          ws.slope, ws.nactive, ws.nupdate, ws.LSiter,
                          ws.iteration, ws.solver_iter_base, ws.maxiter_arr,
                          ws.solver_stats_f, ws.solver_stats_i,
                          ws.out_nchange, ws.out_improvement,
                          ws.out_gradient, ws.out_stop, ws.out_reason,
                          ws.iteration, ws.s08c_status, NV, ws.rows],
                  device=ws.device, stream=ws.stream)

    def _s08d(self):
        ws = self.ws
        wp.launch(s08._finalize, ws.nworld,
                  inputs=[self._finalize_active, ws.solver_exit, ws.iteration,
                          ws.qacc, ws.qacc_warmstart, ws.solver_iter,
                          ws.solver_nnz, ws.s08d_status, NV],
                  device=ws.device, stream=ws.stream)

    # ================= prepare ===========================================
    def prepare_inputs(self, batch: dict):
        """Host-side prepare: validate, upload, classify worlds.  Not hot path."""
        ws = self.ws
        required = {
            'J', 'aref', 'D', 'R', 'loss', 'efc_type', 'efc_id', 'cone_dim',
            'cone_mu', 'cone_friction', 'qM', 'qfrc_smooth', 'qacc_smooth',
            'qacc_warmstart', 'nefc', 'ne', 'nf', 'ncon',
            'contact_dim', 'contact_mu', 'contact_friction',
        }
        missing = required - set(batch)
        extra = set(batch) - required
        if missing or extra:
            raise ValueError(('input keys', sorted(missing), sorted(extra)))
        shapes = {
            'J': (ws.nworld, ws.rows, NV), 'aref': (ws.nworld, ws.rows),
            'D': (ws.nworld, ws.rows), 'R': (ws.nworld, ws.rows),
            'loss': (ws.nworld, ws.rows),
            'efc_type': (ws.nworld, ws.rows), 'efc_id': (ws.nworld, ws.rows),
            'cone_dim': (ws.nworld, ws.rows), 'cone_mu': (ws.nworld, ws.rows),
            'cone_friction': (ws.nworld, ws.rows, 5),
            'qM': (ws.nworld, NM), 'qfrc_smooth': (ws.nworld, NV),
            'qacc_smooth': (ws.nworld, NV), 'qacc_warmstart': (ws.nworld, NV),
            'nefc': (ws.nworld,), 'ne': (ws.nworld,), 'nf': (ws.nworld,),
            'ncon': (ws.nworld,),
            'contact_dim': (ws.nworld, ws.contacts),
            'contact_mu': (ws.nworld, ws.contacts),
            'contact_friction': (ws.nworld, ws.contacts, 5),
        }
        int_keys = {'efc_type', 'efc_id', 'cone_dim', 'nefc', 'ne', 'nf',
                    'ncon', 'contact_dim'}
        for key in sorted(required):
            value = np.ascontiguousarray(np.asarray(batch[key]))
            want = np.int32 if key in int_keys else np.float64
            if value.shape != shapes[key] or value.dtype != want:
                raise TypeError((key, value.shape, str(value.dtype),
                                 shapes[key], str(want)))
            getattr(ws, key if key != 'aref' else 'aref').assign(value)
        ws.count.assign(np.ascontiguousarray(np.asarray(batch['nefc'], np.int32)))

        nefc_host = np.asarray(batch['nefc'], np.int32)
        empty = (nefc_host == 0).astype(np.int32)
        newton = (nefc_host > 0).astype(np.int32)
        ws.empty.assign(empty)
        ws.newton_world.assign(newton)
        ws.active.assign(newton.copy())

        ws.maxiter_arr.assign(np.full(ws.nworld, ws.maxiter, np.int32))
        if int(ws.maxiter_arr.numpy()[0]) != int(ws.maxiter):
            raise AssertionError(('host/device maxiter mismatch (R01-F21)',
                                  ws.maxiter, int(ws.maxiter_arr.numpy()[0])))
        scalars = ws.model_scalars.numpy()
        self._tol = float(scalars[0])
        self._mean = float(scalars[1])
        ws.tolerance.assign(np.full(ws.nworld, self._tol, np.float64))
        ws.meaninertia.assign(np.full(ws.nworld, self._mean, np.float64))
        ws.solver_iter_base.assign(np.zeros(ws.nworld, np.int32))
        ws.stage.assign(np.ones(ws.nworld, np.int32))
        # per-run loop-state reset so one workspace can serve repeated runs
        ws.reset_run_state()

    # ================= warmstart selection ===============================
    def _warmstart_select(self):
        """engine_forward.c:393-444 warm branch, non-PGS.

        Warm evaluator (jar from qacc_warmstart) then smooth evaluator
        (efc_b); the smooth evaluator's force/state/qfrc side effects are the
        ones that survive even when warm wins (S08 A contract).  `active`
        already equals the Newton-world mask from prepare.
        """
        ws = self.ws
        nw = ws.newton_world
        # qacc = qacc_warmstart on entry (mju_copy)
        self._glue(K._copy_qacc, [nw, ws.qacc_warmstart, ws.qacc])
        # jar_warm = J*qacc_warmstart - aref
        self._jacvec_sub(nw, ws.qacc_warmstart, self._jar_warm_flat)
        # warm evaluator: constraintUpdate(jar, &cost_warm, flag=0)
        self._s04_update(self._jar_warm_flat, self._flag_off)
        self._s04_transpose()
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s04, ws.err_code])
        # warm Gauss (engine_forward.c:427-432) added to cost_warm
        self._mulM(nw, ws.qacc_warmstart, ws.Ma_warm)
        self._gauss(nw, ws.Ma_warm, ws.qacc_warmstart, ws.cost,
                    self._add_cost_on)
        # keep warm total for the comparison
        self._glue(K._copy_cost, [nw, ws.cost, ws.warm_cost])
        # smooth evaluator: constraintUpdate(efc_b, &cost_smooth, flag=0)
        self._s04_update(self._efc_b_flat, self._flag_off)
        self._s04_transpose()
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s04, ws.err_code])
        # cost_smooth is the constraint cost only (Gauss at qacc_smooth is 0)
        self._glue(K._copy_cost, [nw, ws.cost, ws.smooth_cost])
        # S08 A: selection; smooth side effects preserved (aliasing enforces it)
        self._s08a()
        self._glue(K._gate_status,
                   [ws.active, ws.s08a_status, self._err_base_s08b, ws.err_code])

    # ================= full batch run ====================================
    def run_batch(self):
        """Enqueue the complete schedule, then return the public batch status."""
        self.launch_device()
        return self.status()

    def launch_device(self):
        """Enqueue full fwdConstraint without sync/download or a host verdict.

        Preparation/allocation/input uploads are separate. F may record this
        exact hot schedule with all actual roots, typed commands and first
        error origins. Graph/residency is still subject to real validation.
        """
        ws = self.ws
        nw = ws.newton_world

        # Count validation precedes classification side effects, including
        # invalid negative nefc and malformed nominally-empty worlds.
        self._glue(K._validate_entry_counts,
                   [ws.nefc, ws.ne, ws.nf, ws.ncon, ws.empty, nw, ws.active,
                    ws.err_code, ws.error_item, ws.rows, ws.contacts])
        self._glue(K._validate_contact_rows,
                   [ws.nefc, ws.ne, ws.nf, ws.ncon, self._type_flat,
                    self._id_flat, self._cone_dim_flat, self._cone_mu_flat,
                    self._cone_fr_flat, self._dim_flat, self._mu_flat,
                    self._fr_flat, nw, ws.active, ws.err_code, ws.error_item,
                    ws.rows, ws.contacts])
        # ---- empty fast path (engine_forward.c:462-468) ------------------
        self._glue(K._empty_fast_path,
                   [ws.empty, ws.qacc_smooth, ws.qacc, ws.qacc_warmstart,
                    ws.qfrc_constraint, ws.solver_iter])

        # ---- efc_b = J*qacc_smooth - aref (engine_forward.c:472-474) ----
        self._jacvec_sub(nw, ws.qacc_smooth, self._efc_b_flat)

        # ---- warmstart selection / cold branch ---------------------------
        if ws.warmstart_enabled:
            self._warmstart_select()
        else:
            self._glue(K._copy_qacc, [nw, ws.qacc_smooth, ws.qacc])
            self._glue(K._zero_rows,
                       [nw, ws.nefc, self._force_flat, ws.rows])

        # ---- d->solver_iter = 0 (engine_forward.c:477) ------------------
        self._glue(K._solver_iter_reset, [nw, ws.solver_iter])

        # ---- Newton init (engine_solver.c:1513-1528) ---------------------
        self._mulM(ws.active, ws.qacc, ws.Ma)
        self._jacvec_sub(ws.active, ws.qacc, self._jar_flat)
        self._s04_update(self._jar_flat, self._flag_on)
        self._s04_transpose()
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s04, ws.err_code])
        self._gauss(ws.active, ws.Ma, ws.qacc, ws.cost, self._add_cost_on)
        self._count(ws.active)
        self._s05_launch(self._kind_direct)
        self._glue(K._gate_status, [ws.active, ws.status, self._err_base_s05,
                                    ws.err_code])
        self._s06_launch(ws.status)
        self._glue(K._gate_status, [ws.active, ws.status, self._err_base_s06,
                                    ws.err_code])
        self._glue(K._init_direction, [ws.active, ws.candidate, ws.search])

        # ---- masked schedule: source while(iter<maxiter) -----------------
        for k in range(ws.maxiter):
            self._loop_body(k)

        # ---- exit (engine_solver.c:1597-1610; engine_forward.c:447-448) --
        self._glue(K._finalize_exit,
                   [nw, ws.err_code, ws.solver_exit, self._finalize_active])
        self._s08d()
        # S08 D status joins the first-error channel like every other stage
        self._glue(K._gate_status,
                   [self._finalize_active, ws.s08d_status, self._err_base_s08d,
                    ws.err_code])

    def status(self):
        """Synchronize once after the schedule and expose every world.

        err_code is a batch failure channel: the error of the lowest-index
        failing world (zero only when all worlds succeeded).  per_world
        retains every world's first error and final stage statuses.  Scalar
        legacy fields remain scalars for one world and become complete lists
        for a batch; they never silently project world zero.
        """
        ws = self.ws
        wp.synchronize_stream(ws.stream)
        names = ('err_code', 'stop_reason', 'iteration', 'solver_iter',
                 'solver_nnz', 'choice', 's07_status', 's08d_status',
                 's08a_status', 's08b_status', 's08c_status', 'status',
                 'active', 'solver_exit', 'route', 'error_item')
        values = {name: getattr(ws, name).numpy().astype(np.int64).tolist()
                  for name in names}
        worlds = [{name: int(values[name][world]) for name in names}
                  for world in range(ws.nworld)]
        failed = [i for i, world in enumerate(worlds) if world['err_code']]
        result = {name: values[name][0] if ws.nworld == 1 else values[name]
                  for name in names}
        result.update(err_code=worlds[failed[0]]['err_code'] if failed else 0,
                      ok=not failed, nworld=ws.nworld, failed_worlds=failed,
                      first_error_world=failed[0] if failed else None,
                      per_world=worlds)
        return result

    def configure_diagnostic_profile(self,profile):
        """Cold-only optional-witness retirement before native roots are sealed.

        The fused fast core never binds these arrays. Empty slot dimensions
        remove their resident allocation; physics stats and final events remain.
        A fresh compatible solver is required to turn full witnesses back on.
        """
        if profile=='compatibility':profile='compatible'
        if profile not in ('compatible','fast'):raise ValueError('Newton diagnostic profile')
        if profile==self.diagnostic_profile:return
        if profile!='fast':raise ValueError('compatible history requires cold solver reconstruction')
        if hasattr(self,'persistent_cuda_core'):
            if self.launch_device!=self.persistent_cuda_core.original:raise ValueError('cannot retire sealed native witness roots')
            del self.persistent_cuda_core
        for name in ('tape_f','tape_i','tape_status','tape_eval','tape_tag','tape_Mv','tape_Jv',
                     'tape_quad','tape_quadGauss','tape_ls_quad0_input'):
            array=getattr(self,name);shape=(array.shape[0],0,*array.shape[2:])
            setattr(self,name,wp.empty(shape,dtype=array.dtype,device=array.device))
        self.tape_slots=0;self.diagnostic_profile='fast'

    def download_tape(self):
        """Witness tape download, outside the production loop only."""
        ws = self.ws
        if self.diagnostic_profile=='fast':raise RuntimeError('Newton witness history is disabled in the explicit fast profile')
        wp.synchronize_stream(ws.stream)
        return {
            'tape_f': self.tape_f.numpy().copy(),
            'tape_i': self.tape_i.numpy().copy(),
            'tape_status': self.tape_status.numpy().copy(),
            'tape_eval': self.tape_eval.numpy().copy(),
            'tape_tag': self.tape_tag.numpy().copy(),
            'tape_Mv': self.tape_Mv.numpy().copy(),
            'tape_Jv': self.tape_Jv.numpy().copy(),
            'tape_quad': self.tape_quad.numpy().copy(),
            'tape_quadGauss': self.tape_quadGauss.numpy().copy(),
            'tape_ls_quad0_input': self.tape_ls_quad0_input.numpy().copy(),
        }

    def _loop_body(self, k: int):
        """One masked iteration in exact source order."""
        ws = self.ws
        # 0. seed the LS diagnostics with the untouched markers (S07
        #    acceptance discipline; keeps the tape tail non-stale, R01-F17)
        self._glue(K._seed_eval, [ws.eval_trace, ws.eval_tag])
        # Capture the actual scalar consumed by this S07 call before later
        # Gauss evaluation can update quad0 in the same Newton iteration.
        self._glue(K._tape_ls_input,
                   [k, self.tape_slots, ws.quad0, self.tape_ls_quad0_input])
        # 1. search / CGsearch (engine_solver.c:1533)
        ws.launch_s07()
        self._glue(K._tape_ls_buffers,
                   [k, self.tape_slots, ws.rows, ws.Mv, ws.Jv, ws.quad,
                    ws.quadGauss, self.tape_Mv, self.tape_Jv,
                    self.tape_quad, self.tape_quadGauss])
        # 2. alpha==0 break before commit/stats/iter++ (engine_solver.c:1535-1538)
        self._glue(K._gate_ls,
                   [ws.active, ws.alpha, ws.s07_status, ws.stop_reason,
                    ws.err_code])
        # 3. advance three states (engine_solver.c:1540-1543) — S08 B
        self._s08b()
        self._glue(K._gate_status,
                   [ws.active, ws.s08b_status, self._err_base_s08b, ws.err_code])
        # 4. save old (engine_solver.c:1548-1551)
        self._glue(K._save_old,
                   [ws.active, ws.nefc, self._state_flat, ws.cost,
                    self._oldstate_flat, ws.oldcost, ws.rows])
        # 5. CGupdateConstraint (engine_solver.c:1556) — S04 + Gauss + counts
        self._s04_update(self._jar_flat, self._flag_on)
        self._s04_transpose()
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s04, ws.err_code])
        self._gauss(ws.active, ws.Ma, ws.qacc, ws.cost, self._add_cost_on)
        self._count(ws.active)
        # 6. HessianIncremental (engine_solver.c:1558) — S05 kind=3
        self._s05_launch(self._kind_incr)
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s05, ws.err_code])
        # 7. CGupdateGradient (engine_solver.c:1560) — S06 (grad/Mgrad/candidate)
        self._s06_launch(ws.status)
        self._glue(K._gate_status,
                   [ws.active, ws.status, self._err_base_s06, ws.err_code])
        # 8. saveStats + tolerance break (engine_solver.c:1564-1582) — S08 C
        self._s08c()
        self._glue(K._gate_stats,
                   [ws.active, ws.out_stop, ws.out_reason, ws.s08c_status,
                    ws.stop_reason, ws.err_code])
        # 9. direction write (engine_solver.c:1585): happens for every world
        #    still looping, including those whose following while guard will
        #    end the loop on maxiter
        self._glue(K._direction_commit, [ws.active, ws.candidate, ws.search])
        # 10. while(iter<maxiter) guard after the direction write
        self._glue(K._gate_maxiter,
                   [ws.active, ws.out_stop, ws.out_reason, ws.stop_reason])
        # 11. witness tape for this slot
        self._glue(K._tape_commit,
                   [k, self.tape_slots, ws.active, ws.alpha, ws.slope,
                    ws.slope, ws.out_improvement, ws.out_gradient, ws.cost,
                    ws.oldcost, ws.quad0, ws.LSiter, ws.LSresult, ws.committed,
                    ws.route, ws.nupdate, ws.nactive, ws.ncone, ws.out_nchange,
                    ws.iteration, ws.stop_reason, ws.err_code,
                    ws.s07_status, ws.s08b_status, ws.status, ws.s08c_status,
                    self.tape_f, self.tape_i, self.tape_status,
                    ws.eval_trace, ws.eval_tag, self.tape_eval, self.tape_tag])
