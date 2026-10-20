"""Handwritten, block/world OSC with device cuBLASDx and cuSOLVERDx calls.

Original runtime callbacks retain their physical refresh boundary. CUDA groups
replace arithmetic, gates, cache refresh, reset, durable copies and publication.
No Warp arithmetic is generated or copied into these sources.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import inspect
import types
import warp as wp
from libero.gpu.host.build.native import NativePlan
from libero.gpu.reference.control.osc import controller as reference
from libero.gpu.host.assembly.pipeline.osc import FusedRuntimeControllerAdapter

HERE=(GPU_ROOT / 'cuda/control')
REFRESH=tuple(inspect.signature(reference.refresh_cache.func).parameters)
CONTROL=tuple(inspect.signature(reference.fused_osc_control.func).parameters)

def refresh_args(c):
    return [getattr(c,k) for k in REFRESH]

def control_args(c,*,emit_diagnostics=None):
    """Bind the native output flag without changing the sealed workspace mode."""
    rename={'raw_tau':'torques_raw','emit_diagnostics':'diagnostics'}
    if getattr(c,'robot_projection',False):rename.update(ctrl='internal_ctrl',gripper='internal_gripper')
    return [int(bool(emit_diagnostics)) if k=='emit_diagnostics' and emit_diagnostics is not None
            else getattr(c,rename.get(k,k)) for k in CONTROL]

def _make(name, stages, c, root):
    heavy=any(s[0] in ('osc_cuda::control','osc_cuda::control_fast','osc_goal_cuda::control') for s in stages)
    sources=[(GPU_ROOT / 'cuda/control/osc.cuh'),(GPU_ROOT / 'cuda/control/osc_runtime.cuh')]
    if getattr(c,'robot_projection',False):sources.append(GPU_ROOT / 'cuda/control/robot_osc.cuh')
    if getattr(c,'position_only',False):sources.append(GPU_ROOT / 'cuda/control/position_osc.cuh')
    if getattr(c,'variable_mode',0):sources.append(GPU_ROOT / 'cuda/control/variable_osc.cuh')
    if getattr(c,'general_goals',False):sources.extend([GPU_ROOT / 'cuda/control/osc_goals.cuh',GPU_ROOT / 'cuda/control/osc_configured_control.cuh'])
    return NativePlan(name,sources,stages,
        c.nworld,root,mathdx=heavy,bridges=[(GPU_ROOT / 'cuda/control/osc_bridge.cu')] if heavy else [],
        defines={'DX_SM':c.device.arch*10,'OSC_UNCOUPLE':int(c.uncouple!=0)},device=c.device)

class _Variants:
    def __init__(self,plans):self.plans=plans
    def launch(self,owner,**values):
        key=tuple(sorted(values.items()));self.plans[key].launch(owner.stream)

def workspace_plans(c,root):
    """Prepare workspace-only calls for component use and independent checks."""
    run=_make('osc_workspace', [('osc_cuda::refresh',refresh_args(c),'lane0'),
        ('osc_cuda::control',control_args(c),'block')],c,root)
    def launch(owner,*,stream=None):run.launch(stream)
    c.launch=types.MethodType(launch,c)
    c._cuda_osc_workspace=run
    return run

def install(owner,root,*,dataflow_profile="compatible",diagnostic_profile="compatibility"):
    if dataflow_profile not in ("compatible","fast"):raise ValueError("invalid OSC dataflow profile")
    if diagnostic_profile not in ("compatibility","fast"):raise ValueError("invalid OSC diagnostic profile")
    """Replace existing r03 runtime groups without rebinding any data owners."""
    gather_report=None
    debug=bool(getattr(owner,'debug',False))
    if hasattr(owner,'dynamics'):
        from .osc_gather import install as install_gather
        bundle=owner
        gather_report=install_gather(bundle.dynamics,Path(root)/'gather',dataflow_profile=dataflow_profile)
        owner=bundle.controller
        # These callbacks were frozen before candidate installation. Replace
        # their declared dispatch together, preserving their resident inputs.
        owner.gather=bundle.dynamics.gather_controller_inputs
        owner.live_bias=bundle.dynamics.gather_live_bias
        owner._callbacks=(owner.forward_refresh,owner.gather,owner.live_bias)
    if not isinstance(owner,FusedRuntimeControllerAdapter):
        raise TypeError('CUDA OSC install requires pipeline runtime controller')
    c=owner.controller
    fast=dataflow_profile=="fast"
    # Input/cache aliasing can make refresh writes change a later live read.
    # Retain the compatible staging order for any overlapping workspace arrays.
    # Runtime binding tokens freeze this cold proof for the plan lifetime.
    from libero.gpu.host.assembly.compose.backup import _overlap
    arrays=[a for a in vars(c).values() if isinstance(a,wp.array)]
    aliases=any(_overlap(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:])
    if aliases:fast=False
    if getattr(c,'general_goals',False):fast=False
    # Only the disjoint, non-debug fast path suppresses optional output stores.
    # c.diagnostics belongs to the sealed reference workspace configuration;
    # pass this cold native selection explicitly rather than mutating it.
    emit_diagnostics=int(bool(c.diagnostics) and
        (debug or not fast or diagnostic_profile=='compatibility'))
    args=control_args(c,emit_diagnostics=emit_diagnostics)
    if fast:
        cache={"valid":wp.zeros(c.nworld,dtype=wp.int32,device=c.device),
            "key":wp.zeros((c.nworld,91),dtype=scalar.wp_real,device=c.device),
            "data":wp.zeros((c.nworld,206),dtype=scalar.wp_real,device=c.device),
            "ranks":wp.zeros((c.nworld,3),dtype=wp.int32,device=c.device),
            "hits":wp.zeros(c.nworld,dtype=wp.int32,device=c.device)}
        owner._cuda_osc_derived_cache=cache
        run_stages=[('osc_cuda::control_fast',args+list(cache.values())+[
            c.phys_mass,c.phys_jacobian,c.phys_q,c.phys_qvel,c.phys_ee_pos,c.phys_ee_rot,
            c.phys_ee_velocity,c.cache_valid],'block')]
    else:
        run_stages=[('osc_cuda::refresh',refresh_args(c),'lane0'),('osc_cuda::control',args,'block')]
    if getattr(c,'general_goals',False):
        from libero.gpu.host.runtime.control.general_osc import goal_stages
        args=list(args)
        args[CONTROL.index('goal_pos')]=c.goal_effective_pos
        args[CONTROL.index('policy_step')]=c.goal_core_policy
        run_stages=[('osc_cuda::refresh',refresh_args(c),'lane0'),*goal_stages(c),('osc_goal_cuda::control',args+[c.goal_ori_error],'block')]
    if getattr(c,'position_only',False):
        run_stages=[('position_cuda::orientation',[c.active,c.policy_step,c.error,c.goal_rot],'lane0')]+run_stages
    if getattr(c,'variable_mode',0):
        run_stages=[('osc_config_cuda::gains',[c.active,c.policy_step,c.error,c.phys_action,c.phys_impedance_limits,c.gains,c.variable_mode,c.raw_action_dim],'lane0')]+run_stages
    if getattr(c,'robot_projection',False):
        from libero.gpu.host.runtime.control.robot_osc import project_stage
        run_stages.append(project_stage(c))
    def gate(mask,policy):return ('osc_cuda::gate',[mask,owner.parents.valid,owner.parents.written,
        owner.parents.expected,owner.first_error,c.error,c.policy_step,policy],'lane0')
    def commits(mask):
        return [('osc_cuda::copy_double',[mask,getattr(c,k),a,a.shape[1]],'lane0') for k,a in owner.public.items()]+[
            ('osc_cuda::copy_int',[mask,getattr(c,k),a],'lane0') for k,a in owner.public_int.items()]
    def reset(mask,clear,update):return ('osc_cuda::reset',[mask,c.phys_ee_pos,c.phys_ee_rot,c.phys_q,
        c.goal_pos,c.goal_rot,c.q0,c.internal_gripper if getattr(c,'robot_projection',False) else c.gripper,c.error,c.cache_valid,c.needs_update,clear,update],'lane0')
    def resets(mask,clear,update):
        stages=[reset(mask,clear,update)]
        if getattr(c,'robot_projection',False):
            from libero.gpu.host.runtime.control.robot_osc import reset_stage
            stages.append(reset_stage(c,mask,clear))
        if getattr(c,'general_goals',False):
            from libero.gpu.host.runtime.control.general_osc import reset_goal_stage
            stages.append(reset_goal_stage(c,mask))
        elif getattr(c,'variable_mode',0):stages.append(('osc_config_cuda::reset_gains',[mask,c.phys_initial_gains,c.gains],'lane0'))
        return stages
    seeds=[('osc_cuda::seed_double',[a,getattr(c,k),a.shape[1]],'lane0') for k,a in owner.public.items()]
    seeds += [('osc_cuda::seed_int',[a,getattr(c,k)],'lane0') for k,a in owner.public_int.items()]
    owner._seed=_make('osc_seed',seeds,c,root)
    owner._run=_Variants({(('policy',policy),):_make('osc_run_policy%d'%policy,[gate(c.active,policy),
        *run_stages,
        ('osc_cuda::commit_mask',[c.active,c.error,owner.first_error,owner.commit_mask,owner.event,owner.valid],'lane0'),
        *commits(owner.commit_mask),('osc_cuda::publish',[owner.commit_mask,owner.event,owner.parents.expected,
            owner.written,owner.consumed_parent,owner.valid],'lane0')],c,root) for policy in (0,1)})
    owner._reset=_Variants({(('clear',clear),('update',update)):_make('osc_reset_%d%d'%(clear,update),
        [gate(owner.call_mask,0),*resets(owner.call_mask,clear,update),*commits(owner.call_mask)],c,root)
        for clear in (0,1) for update in (0,1)})
    owner._bootstrap_begin=_Variants({():_make('osc_bootstrap_begin',
        [gate(owner.call_mask,0),*resets(owner.call_mask,1,1)],c,root)})
    owner._bootstrap_end=_Variants({():_make('osc_bootstrap_end',[
        ('osc_cuda::refresh',refresh_args(c),'lane0'),('osc_cuda::bootstrap_finish',
            [owner.call_mask,c.error,c.cache_valid,c.needs_update,owner.first_error],'lane0'),
        *commits(owner.call_mask)],c,root)})
    # Replace the remaining scalar planning launch with prebound force variants.
    planning={force:_make('osc_plan%d'%force,[('osc_cuda::plan',[owner.active,owner.first_error,
        owner.public_int['needs_update'],owner.refresh_mask,owner.call_mask,force],'lane0')],c,root) for force in (0,1)}
    def enqueue(self,*,policy_step=False,force_refresh=False,stream=None):
        self.validate(stream)
        if self._world_alias_fallback:raise ValueError('cross-world OSC aliases require an explicitly prepared fallback')
        planning[int(bool(force_refresh))].launch(self.stream)
        self.forward_refresh(self.refresh_mask,self.stream);self.gather(self.refresh_mask,self.stream)
        self.live_bias(self.call_mask,self.stream);self._seed.launch(self.stream)
        wp.copy(c.active,self.call_mask,stream=self.stream);wp.copy(c.needs_update,self.refresh_mask,stream=self.stream)
        self._run.launch(self,policy=int(bool(policy_step)))
    owner.enqueue=types.MethodType(enqueue,owner)
    init_mask=wp.empty(owner.capacity,dtype=wp.int32,device=owner.device)
    init_nu=getattr(c,'actual_nu',9)
    init_ctrl=wp.empty((owner.capacity,init_nu),dtype=scalar.wp_real,device=owner.device)
    init_args=[init_mask,owner.first_error,init_ctrl,owner.public['ctrl'],owner.valid,owner.written,owner.event]
    init_fn='osc_cuda::reset_public'
    if getattr(c,'robot_projection',False):init_fn='osc_robot_cuda::reset_public';init_args.append(init_nu)
    init_plan=_make('osc_reset_public',[(init_fn,init_args,'lane0')],c,root)
    def initialize(self,mask,initial_ctrl,*,stream=None):
        from libero.gpu.host.runtime.dynamics.runtime_adapter import _array
        self.validate(stream)
        _array(mask,(self.capacity,),wp.int32,self.device)
        _array(initial_ctrl,(self.capacity,init_nu),scalar.wp_real,self.device)
        wp.copy(init_mask,mask,stream=self.stream);wp.copy(init_ctrl,initial_ctrl,stream=self.stream)
        init_plan.launch(self.stream)
    owner.initialize_ctrl_from_reset=types.MethodType(initialize,owner)
    owner._cuda_osc_plans={'planning':planning,'run':owner._run,'reset':owner._reset,
        'bootstrap_begin':owner._bootstrap_begin,'bootstrap_end':owner._bootstrap_end,'seed':owner._seed,
        'initialize':init_plan}
    # The existing flag guards only nullspace_matrix[49]. The shared projector
    # still computes torques; the other diagnostics and transaction copies stay.
    # When disabled, seed/commit retains this field's last published value.
    return {'dataflow_profile':dataflow_profile,'effective_dataflow_profile':('fast' if fast else 'compatible'),'alias_fallback':aliases,'diagnostic_profile':diagnostic_profile,
        'emit_diagnostics':emit_diagnostics,'workspace_diagnostics':int(bool(c.diagnostics)),
        'effective_diagnostic_profile':('compatibility' if emit_diagnostics else 'fast'),
        'held_diagnostics':([] if emit_diagnostics else ['nullspace_matrix']),
        'gather':gather_report,'groups':{
        'run':{str(k):p.manifest for k,p in owner._run.plans.items()},
        'reset':{str(k):p.manifest for k,p in owner._reset.plans.items()},
        'seed':owner._seed.manifest}}


def install_dataflow(bundle,root,*,dataflow_profile="fast",diagnostic_profile="compatibility"):
    """Exact M/J-derived cache and live-input shared loading; retained outputs."""
    return install(bundle,root,dataflow_profile=dataflow_profile,diagnostic_profile=diagnostic_profile)

