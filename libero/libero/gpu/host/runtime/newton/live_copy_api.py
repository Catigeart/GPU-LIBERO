"""Cold native-model live Newton candidates, separate from frozen fixed runs.

Diagnostic execution is explicit and carries no production grant. Native MJB
identity is checked before topology/scalars are uploaded; live device buffers
are bound before the retained generated solver makes pointer aliases.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from ..core.model_bytes import load_native_model
from types import SimpleNamespace
import numpy as np
import mujoco
from libero.gpu.host.runtime.newton.specialization import load_candidate, _AdmittedModel
from libero.gpu.host.runtime.newton.runtime_adapter import _NewtonRuntimeAdapter
from libero.gpu.host.runtime.newton.parallel_tape import parallel_tape_solver_type


class ColdNewtonRuntimeAdapter(_NewtonRuntimeAdapter):
    def __init__(self, workspace, *, rules, solver_factory):
        self._solver_factory = solver_factory
        super().__init__(workspace, rules=rules)

    def seal(self):
        if not self._sealed:
            self._prepare_public()
            self.ws._build_s07_args()
            self.solver = self._solver_factory(self.ws)
            if self.solver.ws is not self.ws:
                raise ValueError('generated solver/workspace ownership mismatch')
            self._sealed = True
        return self


def _validate_native(model):
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise ValueError('candidate requires Newton solver')
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError('candidate requires elliptic cone')
    if int(model.opt.jacobian) != int(mujoco.mjtJacobian.mjJAC_DENSE) and not (
        int(model.opt.jacobian) == int(mujoco.mjtJacobian.mjJAC_AUTO) and model.nv < 60):
        raise ValueError('sparse Jacobian not qualified')
    if model.neq or model.ntendon or model.nplugin or int(model.opt.noslip_iterations):
        raise ValueError('equality/tendon/plugin/noslip not implemented')
    if int(model.opt.enableflags):
        raise ValueError('optional enabled solver features not qualified')
    warm_disabled = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    if int(model.opt.disableflags) & ~warm_disabled:
        raise ValueError('other disableflags not qualified')
    return warm_disabled


def prepare_live_copy_runtime(*, model_path=None, model=None, model_spec, rows, contacts, nworld,
                         build_root, live_inputs, canonical_outputs=None,
                         registry=None, registered_candidate_id=None,
                         mode='diagnostic', device='cuda:0', diagnostic_profile='compatible',
                         candidate_loader=None, adapter_type=None):
    """Cold load exact native MJB and seal a live component-runtime candidate.

    The native model must satisfy the implemented Newton domain. An optional
    registry additionally checks component-runtime scope; numerical review
    records are not part of runtime execution.
    Returned adapter exposes source_manifest and exact model provenance.
    """
    if mode not in ('diagnostic', 'production'):
        raise ValueError('unsupported execution mode')
    if diagnostic_profile == 'compatibility':
        diagnostic_profile = 'compatible'
    if diagnostic_profile not in ('compatible', 'fast'):
        raise ValueError('Newton diagnostic profile')
    model, path = load_native_model(model=model, model_path=model_path,
                                    expected_sha256=model_spec.model_sha256)
    model_sha = model_spec.model_sha256
    if (int(model.nv), int(model.nM)) != (model_spec.dimensions.nv, model_spec.dimensions.nM):
        raise ValueError('native dimensions differ from ModelSpec')
    warm_disabled = _validate_native(model)
    if registry is None:
        if registered_candidate_id is not None:
            raise ValueError('candidate id requires an explicit capability registry')
        admission = SimpleNamespace(execution_allowed=True,
                                    reasons=(),
                                    production_allowed=True)
    else:
        admission = registry.admit(registered_candidate_id, model_spec, mode=mode,
                                   required_scope='component-runtime')
    rules = _AdmittedModel(admission)
    rules._validate()
    candidate_loader = load_candidate if candidate_loader is None else candidate_loader
    adapter_type = ColdNewtonRuntimeAdapter if adapter_type is None else adapter_type
    package, manifest = candidate_loader(int(model.nv), int(model.nM), Path(build_root))
    ws = package.workspace.NewtonFwdWorkspace(
        nworld, rows, contacts, device=device, maxiter=int(model.opt.iterations),
        warmstart_enabled=not bool(int(model.opt.disableflags) & warm_disabled))
    # The retained solver reads this before allocating witness history. Native
    # fast execution never consumes that history; avoid allocating it first
    # and retiring it only after the rest of the arena has been constructed.
    ws.diagnostic_profile = diagnostic_profile
    ws.bind_topology(np.asarray(model.dof_Madr, np.int32),
                     np.asarray(model.dof_parentid, np.int32),
                     np.asarray(model.dof_simplenum, np.int32),
                     np.asarray([model.opt.tolerance, model.stat.meaninertia], np.float64))
    outputs = {} if canonical_outputs is None else dict(canonical_outputs)
    if ('qacc_warmstart' in live_inputs and 'qacc_warmstart' in outputs and
            live_inputs['qacc_warmstart'] is not outputs['qacc_warmstart']):
        raise ValueError('warm input and canonical output must retain the same owner')
    adapter = adapter_type(ws, rules=rules,
                                      solver_factory=parallel_tape_solver_type(package.solver.FwdConstraintSolver))
    adapter.bind_inputs(live_inputs)
    adapter.bind_outputs(outputs)
    adapter.seal()
    adapter.model_spec = model_spec
    adapter.admission = admission
    adapter.source_manifest = dict(manifest, model_sha256=model_sha,
        model_identity=model_spec.identity_key, mode=mode, diagnostic_profile=diagnostic_profile,
        execution_scope='component-runtime', acceptance='diagnostic-pending',
        runtime=dict(rows=rows, contacts=contacts, nworld=nworld,
                     iterations=int(model.opt.iterations), tolerance=float(model.opt.tolerance),
                     meaninertia=float(model.stat.meaninertia)))
    adapter.source_manifest['optimization'] = 'cold-parallel-tape-only; independent model runtime review pending'
    adapter.native_model = model
    adapter.native_model_path = path
    return adapter
