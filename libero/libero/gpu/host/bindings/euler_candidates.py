"""Retained Euler recipes; default production installer remains unchanged.

Pruned CUDA source/ABI requires its own selected-consumer receipt. Public cold
requests require an owner-bound selection; private builder is qualification-only.
"""
import libero.gpu.precision as scalar
from dataclasses import dataclass
from pathlib import Path
import re
import warp as wp
from ..build.native import NativePlan
from .euler import _defines, _joint_values, _values, stage

GPU_ROOT = Path(__file__).resolve().parents[2]
HEADER = GPU_ROOT / "cuda/integration/euler_candidates.cuh"

@dataclass(frozen=True)
class EulerCandidateOptions:
    single_rhs: bool = False
    small_angle: bool = False
    scalar_io: bool = False
    reuse_velocity: bool = False
    factor_cache: bool = False

    def defines(self):
        if any(type(getattr(self, name)) is not bool for name in self.__dataclass_fields__):
            raise TypeError("Euler choices must be bool")
        return dict(CF_EC_RHS_SLOTS=1 if self.single_rhs else 4,
            CF_EC_SMALL_ANGLE=int(self.small_angle), CF_EC_SCALAR_IO=int(self.scalar_io),
            CF_EC_REUSE_VEL=int(self.reuse_velocity), CF_EC_CACHE=int(self.factor_cache))

_BINDINGS = {'qp': 'qpos', 'qv': 'qvel', 'qm': 'qM', 'smooth': 'qfrc_smooth', 'constraint': 'qfrc_constraint', 'acc': 'qacc', 'damping': 'dof_damping', 'madr': 'dof_Madr', 'parent': 'dof_parentid', 'count': 'mass_count', 'ma': 'mass_active', 'damp': 'damped', 'ms': 'mass_status', 'wc': 'mass_warning_count', 'wl': 'mass_warning_last', 'jt': 'jnt_type', 'padr': 'jnt_qposadr', 'vadr': 'jnt_dofadr', 'used': 'used_acc', 'nj': 'njnt', 'qh': 'qH', 'qhi': 'qHDiagInv', 'offsets': 'joint_type_offsets', 'ids': 'joint_type_indices'}

def _prepare_candidate(e, root, options, *, runtime, extra_flags=()):
    """Private new-source qualification builder; never selects production itself."""
    definitions = dict(_defines(e, "fast"), **options.defines())
    values = _joint_values(e, _values(e))
    definitions["CF_EC_JOINT_PARALLEL"] = int(values["joint_quat"] is not None)
    if values["joint_quat"] is None:
        values["joint_quat"] = wp.zeros(1, dtype=wp.int32, device=e.device)
    cache_shapes = dict(key=(e.capacity,e.nM), cached_ld=(e.capacity,e.nM),
        cached_inv=(e.capacity,e.nv), cached_si=(e.capacity,e.nv),
        cached_pivot=(e.capacity,e.nv), cached_parent=(e.capacity,e.nv), cached_madr=(e.capacity,e.nv),
        cached_dt=(e.capacity,), cache_valid=(e.capacity,), cached_warnings=(e.capacity,),
        cached_last=(e.capacity,), cache_hit=(e.capacity,))
    for name, shape in cache_shapes.items():
        dtype = scalar.wp_real if name in ("key","cached_ld","cached_inv","cached_si") else (wp.int64 if name=="cached_dt" else wp.int32)
        if not options.factor_cache and name != "cache_hit":
            shape=(1,)
        values[name]=wp.zeros(shape,dtype=dtype,device=e.device)
    declaration=re.search(r"void evaluate_plain\(int w,(.*?)\) \{",HEADER.read_text(),re.S).group(1)
    parts=[p.strip().rsplit(None,1) for p in declaration.split(",")]
    parameters=[(dtype+("*" if name.startswith("*") else ""),name.lstrip("*")) for dtype,name in parts]
    values.update(intent=runtime.intent,state_epoch=runtime.state_epoch,
        forward_valid=runtime.forward_valid,forward_written=runtime.forward_written,
        forward_error=runtime.forward_error,first_error=runtime.first_error,
        written=runtime.written,valid=runtime.valid,expected=runtime.expected_state_epoch)
    arguments=[values[_BINDINGS.get(name,name)] for _,name in parameters]
    stages=[stage("gate",values),("cf_euler_candidate::evaluate_plain",arguments,"block",[dtype for dtype,_ in parameters]),stage("publish",values)]
    if runtime.expected_state_epoch is not None:
        stages.append(stage("expected",values))
    if hasattr(runtime,"integration_count"):
        values["integration_count"]=runtime.integration_count
        stages.append(stage("count",values))
    plan=NativePlan("euler_retained_"+"_".join(name for name in options.__dataclass_fields__ if getattr(options,name)),
        [HEADER],stages,e.capacity,Path(root),defines=definitions,extra_flags=extra_flags)
    plan.euler_candidate_options=options
    plan.euler_candidate_stage_parameters=tuple(name for _,name in parameters)
    plan.euler_candidate_stages=tuple(stages)
    plan.euler_candidate_sources=(HEADER,)
    plan.euler_candidate_values=values
    plan.euler_candidate_cache_private=True
    def invalidate_cache(stream=None):
        with wp.ScopedStream(stream or wp.get_stream(e.device)):
            values["cache_valid"].zero_()
    plan.invalidate_cache=invalidate_cache
    plan.manifest.update(experimental=True,retained_production_default=True,
        diagnostic_abi="fast-public-usedacc-status-only-v1",exact_factor_key="constructed MhB bits+dt+parent+Madr",
        pruned_source_qualification="pending independent selected receipt")
    return plan

def prepare_candidate(e, root, options=None, *, runtime=None, cold_selection=None, extra_flags=()):
    """Cold owner-bound selection only; any unsupported request retains E01 lean."""
    from .euler_lean import prepare_lean
    from .euler_selected_selection import use_selected
    options=options or EulerCandidateOptions()
    options.defines()
    if not use_selected(cold_selection,e,runtime,"fast",options):
        return prepare_lean(e,root,runtime=runtime,extra_flags=extra_flags)
    return _prepare_candidate(e,root,options,runtime=runtime,extra_flags=extra_flags)
