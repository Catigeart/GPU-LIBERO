"""Fixed-model device-resident Newton entry for the production step scheduler.

Cold binding precedes creation of solver aliases and S07 pointer arguments.
The hot entry performs no allocation, upload, download or synchronization.
Canonical public outputs are committed only for successful worlds and retain
the accepted source's per-field write/hold semantics;
``written`` means the successful fwdConstraint entry completed, not that every
byte (notably constraint tails and solver statistics) was overwritten.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp
from libero.gpu.host.runtime.dynamics.rules import FixedModelRules
from libero.gpu.host.runtime.newton.workspace import NewtonFwdWorkspace, NV, NM
from libero.gpu.host.runtime.newton.solver import FwdConstraintSolver


@wp.kernel(enable_backward=False)
def _fill_int(a: wp.array(dtype=wp.int32), value: int):
    a[wp.tid()] = value


@wp.kernel(enable_backward=False)
def _scalars(model: wp.array(dtype=scalar.wp_real), tolerance: wp.array(dtype=scalar.wp_real),
             meaninertia: wp.array(dtype=scalar.wp_real)):
    w = wp.tid()
    tolerance[w] = model[0]
    meaninertia[w] = model[1]


@wp.kernel(enable_backward=False)
def _entry(upstream: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32),
           nefc: wp.array(dtype=wp.int32), empty: wp.array(dtype=wp.int32),
           newton: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
           count: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
           valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int32)):
    w = wp.tid()
    empty[w] = 0
    newton[w] = 0
    active[w] = 0
    count[w] = nefc[w]
    error[w] = upstream[w]
    valid[w] = 0
    written[w] = 0
    if enabled[w] != 0 and upstream[w] == 0:
        if nefc[w] == 0:
            empty[w] = 1
        elif nefc[w] > 0:
            newton[w] = 1
            active[w] = 1


@wp.kernel(enable_backward=False)
def _publish(upstream: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32),
             error: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
             written: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if enabled[w] != 0 and upstream[w] == 0 and error[w] == 0:
        valid[w] = 1
        written[w] = 1




@wp.kernel(enable_backward=False)
def _publication_gate(valid: wp.array(dtype=wp.int32), diagnostic: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if diagnostic[w] != 0:
        valid[w] = 0

@wp.kernel(enable_backward=False)
def _epoch_begin(enabled: wp.array(dtype=wp.int32), plan: wp.array(dtype=wp.int64),
                 attempted: wp.array(dtype=wp.int64), expected: wp.array(dtype=wp.int64),
                 supplied: int):
    w = wp.tid()
    if enabled[w] != 0:
        attempted[w] = attempted[w] + wp.int64(1)
        if supplied != 0:
            expected[w] = plan[w]
        else:
            expected[w] = attempted[w]


@wp.kernel(enable_backward=False)
def _epoch_commit(valid: wp.array(dtype=wp.int32), expected: wp.array(dtype=wp.int64),
                  written: wp.array(dtype=wp.int64)):
    w = wp.tid()
    if valid[w] != 0:
        written[w] = expected[w]

INPUT_FIELDS = (
    'J', 'aref', 'D', 'R', 'loss', 'efc_type', 'efc_id', 'cone_dim',
    'cone_mu', 'cone_friction', 'qM', 'qfrc_smooth', 'qacc_smooth',
    'qacc_warmstart', 'nefc', 'ne', 'nf', 'ncon',
    'contact_dim', 'contact_mu', 'contact_friction',
)
OUTPUT_FIELDS = ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'force', 'state')

@wp.kernel(enable_backward=False)
def _commit_f64(valid: wp.array(dtype=wp.int32), src: wp.array(dtype=scalar.wp_real),
                dst: wp.array(dtype=scalar.wp_real), stride: int):
    i = wp.tid()
    if valid[i // stride] != 0:
        dst[i] = src[i]


@wp.kernel(enable_backward=False)
def _commit_i32(valid: wp.array(dtype=wp.int32), src: wp.array(dtype=wp.int32),
                dst: wp.array(dtype=wp.int32), stride: int):
    i = wp.tid()
    if valid[i // stride] != 0:
        dst[i] = src[i]


PUBLIC_FIELDS = OUTPUT_FIELDS + ('solver_iter', 'solver_nnz', 'solver_stats_f', 'solver_stats_i')
RESET_INT = (
    'iteration', 'committed', 'choice', 'stop_reason', 'err_code', 's07_status',
    's08a_status', 's08b_status', 's08c_status', 's08d_status', 'LSiter',
    'LSresult', 'out_stop', 'out_reason', 'out_nchange', 'solver_exit',
    'nactive', 'ncone', 'nupdate', 'nnz', 'route',
)
RESET_FLOAT = (
    'alpha', 'slope', 'cost', 'oldcost', 'quad0', 'warm_cost', 'smooth_cost',
    'out_improvement', 'out_gradient',
)


class _NewtonRuntimeAdapter:
    """Bind once, seal once, enqueue repeatedly against live GPU buffers.

    ``rules`` must be loaded with ``host.runtime.dynamics.rules.load_rules`` from
    the real MJB. Same dimensions alone never qualify a different model.
    A supplied solver must reference the supplied workspace and already have
    all live bindings installed; rebinding it would leave stale aliases.
    """

    def __init__(self, workspace: NewtonFwdWorkspace, *, rules: FixedModelRules,
                 solver: FwdConstraintSolver | None = None):
        rules._validate()
        self.rules = rules
        self.ws = workspace
        self.solver = solver
        if solver is not None and solver.ws is not workspace:
            raise ValueError('solver/workspace ownership mismatch')
        self._sealed = solver is not None
        self._owners = []
        self._canonical = {}
        self._commit_aliases = {}
        self._all_enabled = wp.full((workspace.nworld,), 1, dtype=wp.int32,
                                    device=workspace.device)
        self.valid = wp.zeros((workspace.nworld,), dtype=wp.int32,
                              device=workspace.device)
        self.written = wp.zeros((workspace.nworld,), dtype=wp.int32,
                                device=workspace.device)
        self._no_publication_error = wp.zeros(workspace.nworld, dtype=wp.int32, device=workspace.device)
        self.publication_error = wp.zeros(workspace.nworld, dtype=wp.int32, device=workspace.device)
        self.n_attempt = wp.zeros(workspace.nworld, dtype=wp.int64, device=workspace.device)
        self.n_expected = wp.zeros(workspace.nworld, dtype=wp.int64, device=workspace.device)
        self.n_written = wp.zeros(workspace.nworld, dtype=wp.int64, device=workspace.device)
        # Cache a dedicated stream handle; zero_ uses Warp's current stream.
        self.stream = workspace.stream
        if solver is not None:
            self._prepare_public()

    def _check_array(self, name, arr, template):
        if (tuple(arr.shape) != tuple(template.shape) or
                arr.dtype != template.dtype or arr.device != template.device):
            raise TypeError((name, 'shape/dtype/device mismatch',
                             arr.shape, arr.dtype, arr.device,
                             template.shape, template.dtype, template.device))
        if not getattr(arr, 'is_contiguous', False):
            raise ValueError((name, 'contiguous device array required'))

    def _bind(self, mapping, allowed):
        if self._sealed:
            raise RuntimeError('bindings are sealed; create a new adapter to rebind')
        unknown = set(mapping) - set(allowed)
        if unknown:
            raise ValueError(('unknown bindings', sorted(unknown)))
        for name, arr in mapping.items():
            self._check_array(name, arr, getattr(self.ws, name))
        for name, arr in mapping.items():
            if name == 'qacc_warmstart':
                self._canonical[name] = arr
            else:
                setattr(self.ws, name, arr)
            self._owners.append(arr)
        # S08 warm/smooth names alias exactly the selected input roots.
        self.ws.warm_qacc = self.ws.qacc_warmstart
        self.ws.smooth_qacc = self.ws.qacc_smooth

    def bind_inputs(self, mapping):
        """Cold bind live S/D/F input roots; no upload or numeric conversion."""
        self._bind(mapping, INPUT_FIELDS)
        return self

    def bind_outputs(self, mapping):
        """Cold bind independent canonical outputs, never Newton work scratch."""
        if self._sealed:
            raise RuntimeError('bindings are sealed')
        unknown = set(mapping) - set(PUBLIC_FIELDS)
        if unknown:
            raise ValueError(('unknown canonical outputs', sorted(unknown)))
        for name, arr in mapping.items():
            self._check_array(name, arr, getattr(self.ws, name))
        self._canonical.update(mapping)
        self._owners.extend(mapping.values())
        return self

    def _prepare_public(self):
        w = self.ws
        def bounds(arr):
            return arr.ptr, arr.ptr + arr.size * (scalar.itemsize if arr.dtype == scalar.wp_real else 4)
        work_bounds = [bounds(getattr(w, name)) for name in PUBLIC_FIELDS]
        public_bounds = []
        for name in PUBLIC_FIELDS:
            work = getattr(w, name)
            if name not in self._canonical:
                self._canonical[name] = wp.clone(work)
            public = self._canonical[name]
            p0, p1 = bounds(public)
            if any(p0 < b1 and b0 < p1 for b0, b1 in work_bounds + public_bounds):
                raise ValueError((name, 'canonical output overlaps work or another public field'))
            public_bounds.append((p0, p1))
            dtype = work.dtype
            self._commit_aliases[name] = (
                wp.array(ptr=work.ptr, dtype=dtype, shape=(work.size,), device=work.device),
                wp.array(ptr=public.ptr, dtype=dtype, shape=(public.size,), device=public.device),
            )


    def seal(self):
        if not self._sealed:
            self._prepare_public()
            self.ws._build_s07_args()
            self.solver = FwdConstraintSolver(self.ws)
            self._sealed = True
        return self

    @property
    def public_arrays(self):
        if not self._sealed:
            raise RuntimeError('seal bindings before reading canonical arrays')
        w, c = self.ws, self._canonical
        return dict(qacc=c['qacc'], qacc_warmstart=c['qacc_warmstart'],
                    warmstart=c['qacc_warmstart'], qfrc_constraint=c['qfrc_constraint'],
                    efc_force=c['force'], efc_state=c['state'], valid=self.valid,
                    written=self.written, firsterror=w.err_code, status=w.err_code,
                    publication_status=self.publication_error,
                    n_attempt=self.n_attempt, n_expected=self.n_expected, n_written=self.n_written,
                    iteration=w.iteration, stop_reason=w.stop_reason,
                    solver_iter=c['solver_iter'], solver_nnz=c['solver_nnz'],
                    solver_stats_f=c['solver_stats_f'], solver_stats_i=c['solver_stats_i'])

    def enqueue(self, upstream_status, active=None, expected_epoch=None, publication_status=None):
        """Run source fwdConstraint; blocked/inactive worlds hold public bytes.

        Inputs are producer-owned buffers on the workspace stream. The caller
        must order other producer streams before enqueueing. No host status
        inspection occurs here. N-internal failure can leave partial work writes; canonical public
        arrays and warmstart remain unchanged because success commits are masked.
        Optional publication_status is a bridge-only diagnostic: it blocks public
        commit without hiding or replacing original N validation/error precedence.
        """
        if not self._sealed:
            raise RuntimeError('seal bindings before enqueue')
        w = self.ws
        enabled = self._all_enabled if active is None else active
        self._check_array('upstream_status', upstream_status, w.err_code)
        self._check_array('active', enabled, w.active)
        publication = self._no_publication_error if publication_status is None else publication_status
        self._check_array('publication_status', publication, self.publication_error)
        epoch_plan = self.n_attempt if expected_epoch is None else expected_epoch
        self._check_array('expected_epoch', epoch_plan, self.n_expected)
        protected = {getattr(w, name).ptr for name in RESET_INT}
        if upstream_status.ptr in protected:
            raise ValueError('upstream status must not alias reset-owned scratch')
        if enabled.ptr in protected | {w.active.ptr, w.empty.ptr, w.newton_world.ptr}:
            raise ValueError('active mask must not alias N-owned scratch')
        with wp.ScopedStream(self.stream):
            wp.copy(self.publication_error, publication, stream=self.stream)
            wp.launch(_epoch_begin, w.nworld,
                      inputs=[enabled, epoch_plan, self.n_attempt, self.n_expected,
                              int(expected_epoch is not None)],
                      device=w.device, stream=self.stream)
            for name in PUBLIC_FIELDS:
                wp.copy(getattr(w, name), self._canonical[name], stream=self.stream)
            for name in RESET_INT + RESET_FLOAT:
                getattr(w, name).zero_()
            for arr, value in ((w.error_item, -1), (w.stage, 1),
                               (w.maxiter_arr, w.maxiter), (w.solver_iter_base, 0)):
                wp.launch(_fill_int, w.nworld, inputs=[arr, value],
                          device=w.device, stream=self.stream)
            wp.launch(_scalars, w.nworld,
                      inputs=[w.model_scalars, w.tolerance, w.meaninertia],
                      device=w.device, stream=self.stream)
            wp.launch(_entry, w.nworld,
                      inputs=[upstream_status, enabled, w.nefc, w.empty,
                              w.newton_world, w.active, w.count, w.err_code,
                              self.valid, self.written],
                      device=w.device, stream=self.stream)
            self.solver.launch_device()
            wp.launch(_publish, w.nworld,
                      inputs=[upstream_status, enabled, w.err_code,
                              self.valid, self.written],
                      device=w.device, stream=self.stream)
            wp.launch(_publication_gate, w.nworld,
                      inputs=[self.valid, self.publication_error],
                      device=w.device, stream=self.stream)
            # ``written`` is the current successful public entry mask, not just work success.
            wp.copy(self.written, self.valid, stream=self.stream)
            wp.launch(_epoch_commit, w.nworld,
                      inputs=[self.valid, self.n_expected, self.n_written],
                      device=w.device, stream=self.stream)
            for name in PUBLIC_FIELDS:
                src, dst = self._commit_aliases[name]
                kernel = _commit_f64 if src.dtype == scalar.wp_real else _commit_i32
                wp.launch(kernel, src.size,
                          inputs=[self.valid, src, dst, src.size // w.nworld],
                          device=w.device, stream=self.stream)
        return self.public_arrays


GENERALIZATION_REQUIREMENTS = {
    'fixed': {'nv': NV, 'nM': NM, 'binding': 'exact MJB and topology'},
    'other_models': 'not supported by this adapter',
    'required_specialization': [
        'NV/NM kernels, S05/S06 and S07 compiled bridge with source certificate',
        'actual matrix topology, row/contact capacity and enabled solver options',
        'new-model residual/stop/write acceptance and nonempty Newton witness',
    ],
}






class FixedNewtonRuntimeAdapter(_NewtonRuntimeAdapter):
    """Strict fixed-MJB public entry; generic entry uses registry admission."""
    def __init__(self, workspace, *, rules, solver=None):
        if not isinstance(rules, FixedModelRules):
            raise TypeError('fixed entry requires actual FixedModelRules')
        super().__init__(workspace, rules=rules, solver=solver)

