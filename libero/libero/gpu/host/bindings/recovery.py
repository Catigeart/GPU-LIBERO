"""Native check/reset/protocol; real conditional forward stays scheduler-owned."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
import warp as wp
from libero.gpu.host.build.native import NativePlan
HEADER=(GPU_ROOT / 'cuda/runtime/recovery.cuh')

def _stage_barriers(stages):
    return [i+1<len(stages) and (s[2]=='block' or stages[i+1][2]=='block')
            for i,s in enumerate(stages)]

def install(bundle,root):
    r=getattr(bundle,'runtime',None)
    if r is None and hasattr(bundle,'recovery'):r=bundle
    if r is None or not hasattr(r,'recovery'):return {'recovery':'not present'}
    d=r.recovery
    if getattr(r,'_conditional_reset_forward',None) is None and d.N!=1:
        raise NotImplementedError('multi-world recovery requires a device any-reset conditional helper')
    plans={};field_stages={}
    for field,warning in [('qpos',4),('qvel',5),('qacc',6)]:
        stages=[('cf_recovery::check',[d.views[field],d.active,d.reset_mask,d.reason,d.bad_index,d.source_error,d.catalog[field]['width'],warning],'block')]
        for name,view in d.views.items():
            kind={wp.float64:'f64',wp.int32:'i32',wp.int64:'i64'}[view.dtype]
            stages.append(('cf_recovery::reset_'+kind,[view,d.defaults[name],d.reset_mask,d.catalog[name]['width']],'block'))
        stages.append(('cf_recovery::warnings',[d.warning_number,d.warning_lastinfo,d.reset_mask,d.reason,d.bad_index],'lane0'))
        if getattr(d,'_fusion_protocol',None) is not None:
            stages.append(('cf_recovery::protocol',[d.reset_mask,*d._fusion_protocol],'lane0'))
        field_stages[field]=stages
        plans[field]=NativePlan('reset_'+field,[HEADER],stages,d.N,root,stage_barriers=_stage_barriers(stages))
    def launch(self,field,warning,stream):
        self.validate_bindings()
        if getattr(self,'_fusion_protocol',None) is not None:self.validate_protocol(self._fusion_protocol_owner)
        plans[field].launch(stream or self.stream);return self.reset_mask
    d._launch=types.MethodType(launch,d)
    n=r.forward.n;event=getattr(n,'final_event',None)
    stages=[('cf_recovery::acceleration',[r.intent,r.first_error,r.forward_valid,r.forward_written,r.state_epoch,n.valid,n.n_written,r.forward.expected_newton,d.active],'lane0')]
    if event is not None and hasattr(r,'event_slot'):
        if r.event_slot.shape!=(d.N,) or r.event_count.shape!=(d.N,) or r.event_overflow.shape!=(d.N,):raise ValueError('event recovery needs per-world slots/counters')
        stages.append(('cf_recovery::select',[d.active,r.euler.qacc,event.candidate,event.candidate_attempt,event.candidate_completed,event.candidate_origin,event.source_stage,n.n_attempt,n.n_written,r.forward.expected_newton,r.state_epoch,r.forward.d.forward_epoch,r.event_count,r.event_slot,r.event_overflow,r.event_metadata,r.event_capacity,r.runtime_shape.nv],'block'))
    before_copy=None;sources=[HEADER]
    if hasattr(r,'event_before'):
        from .copies import CopyPlan
        before_copy=CopyPlan([(r.event_sources[name],target) for name,target in r.event_before.items()],root,r.event_slot,compile_plan=False)
        if not before_copy.fallback:
            if before_copy.nworld!=d.N:raise ValueError('recovery event world count differs')
            sources.append(before_copy.source);stages+=before_copy.stages
    admit=NativePlan('acceleration_admission',[*sources],stages,d.N,root,stage_barriers=_stage_barriers(stages))
    checks=field_stages['qpos']+field_stages['qvel']
    prechecks=NativePlan('position_velocity_checks',[HEADER],checks,d.N,root,stage_barriers=_stage_barriers(checks))
    checks=stages+field_stages['qacc']
    acceleration_check=NativePlan('acceleration_admit_event_check',[*sources],checks,d.N,root,stage_barriers=_stage_barriers(checks))
    def validate_event(self):
        if getattr(self,'capture_events',False) and (self._event_owner_tokens()!=self._event_tokens or self.forward.n.final_event is not self._event_solver):raise ValueError('event retained owner drift')
        if before_copy is not None:before_copy.validate_bindings()
    def acceleration(self):
        validate_event(self)
        admit.launch(self.stream)
        if before_copy is not None and before_copy.fallback:self._copy_event(self.event_before)
    r._admit_current_acceleration=types.MethodType(acceleration,r)
    warning_admit=NativePlan('warning_admission',[HEADER],[('cf_recovery::admit',[r.intent,r.first_error,d.active],'lane0')],d.N,root)
    r._cuda_recovery_plans=tuple(plans.values())+(admit,warning_admit,prechecks,acceleration_check)
    def paired_checks(self):
        self.recovery.validate_bindings()
        if getattr(self.recovery,'_fusion_protocol',None) is None:
            self._protocol(self.recovery.check_position(stream=self.stream))
            self._protocol(self.recovery.check_velocity(stream=self.stream))
            return
        self.recovery.validate_protocol(self)
        prechecks.launch(self.stream)
        # The CUDA stages already ran protocol after each actual reset. This
        # retained callback validates the identity of that protocol mask only.
        self._protocol(self.recovery.reset_mask)
    def admit_check(self):
        validate_event(self)
        self.recovery.validate_bindings()
        if before_copy is not None and before_copy.fallback:
            acceleration(self)
            self._protocol(self.recovery.check_acceleration(stream=self.stream))
        else:
            acceleration_check.launch(self.stream)
            self._protocol(self.recovery.reset_mask)
    r._cuda_pre_step_checks=types.MethodType(paired_checks,r)
    r._cuda_admit_check_acceleration=types.MethodType(admit_check,r)
    r._cuda_recovery_event_copy=before_copy
    def enqueue(self):
        self._recovery_validate();warning_admit.launch(self.stream)
        self._cuda_pre_step_checks()
        self.forward.enqueue(self.intent,self.stream)
        self._cuda_admit_check_acceleration()
        conditional=getattr(self,'_conditional_reset_forward',None)
        if conditional is not None:
            conditional()
        else:
            # Legacy recovery_runtime.py admits exactly one private world.
            # Never interpret a multi-world mask's first element as any-reset.
            if self.recovery.N!=1:
                raise NotImplementedError('multi-world recovery requires a device any-reset conditional helper')
            if self.stream.is_capturing:
                wp.capture_if(self.recovery.reset_mask,on_true=self._reset_forward,
                              stream=self.stream)
            else:
                wp.capture_launch(self._reset_forward_graph,stream=self.stream)
        self._integrate()
    r.enqueue_physics_step=types.MethodType(enqueue,r)
    return {'recovery':list(plans),'event_world_scope':d.N,
            'ordered_position_reset_velocity_checks':True,
            'acceleration_event_before_reset_in_kernel':before_copy is not None and not before_copy.fallback,
            'manifests':[p.manifest for p in r._cuda_recovery_plans]}
