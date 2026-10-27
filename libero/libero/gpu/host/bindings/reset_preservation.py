"""Preserve untouched worlds around conditional masked recovery forward.

Recovery is an extra physical forward for reset worlds only. Some baseline
stages deliberately clear inactive derived outputs; those clears must not erase
another world's already valid ordinary forward or its same-step Euler inputs.
Only bounded Euler, observation and publication/count roots are copied here;
large J/contact/history scratch stays excluded and is not used by same-step
Euler. This opt-in installer leaves ordinary forward semantics unchanged.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
import warp as wp
from libero.gpu.host.build.native import NativePlan


def _token(a):
    return (id(a),int(a.ptr),a.shape,a.strides,a.dtype,a.device)


def collect_roots(bundle):
    r=bundle.runtime;f=r.forward;d=f.d;n=f.n;post=bundle.post;W=r.euler.capacity
    roots={};excluded={}
    def add(name,a):
        if not isinstance(a,wp.array):return
        if not a.size:return
        if a.dtype not in (scalar.wp_real,wp.int32,wp.int64):return
        if not a.is_contiguous or a.device!=r.stream.device or a.size%W:
            raise ValueError('preservation needs exact contiguous world domain: '+name)
        # Public efc_force is committed only for admitted worlds. Its large
        # canonical buffer is held by the original N commit on mask0.
        if name in ('N.public.efc_force','N.public.efc_state'):
            excluded[name]={'shape':a.shape,'reason':'canonical N force/state commit holds mask0, not read by same-step Euler'}
            return
        roots[name]=a
    def attrs(prefix,owner,names):
        for name in names:add(prefix+'.'+name,getattr(owner,name,None))
    def mapping(prefix,m):
        for name,a in m.items():add(prefix+'.'+name,a)
    attrs('Euler',r.euler,('qpos','qvel','time','qM','qacc','qfrc_smooth','qfrc_constraint',
        'qH','qHDiagInv','ctrl','qacc_warmstart','sensordata'))
    attrs('runtime',r,('first_error','state_epoch','expected_state_epoch','forward_valid',
        'forward_written','forward_error','valid','written'))
    attrs('forward',f,('mask','valid','written','error','expected_forward','expected_newton',
        'c_mask','n_admitted','n_upstream','projection_status','projection_item','_counts'))
    attrs('D',d,('active','state_epoch','source_state_epoch','forward_epoch','first_error',
        'cvel','cdof_dot','bias_admitted','force_admitted','smooth_repaired'))
    mapping('D.valid',d.valid);mapping('D.written',d.written);mapping('D.public',d.public)
    st=d.stages
    attrs('D.stages',st,('active','gate_position','gate_velocity','gate_bias','gate_control',
        'gate_actuation','gate_acceleration','status_position','status_velocity','status_bias',
        'status_control','status_actuation','status_acceleration','status_comvel','status_rne',
        'cacc','frc','qfrc_bias','qvel','qfrc_applied','xfrc_applied'))
    pos=st.position
    attrs('F.body',pos.body,('active','qpos','mocap_pos','mocap_quat','xpos','xquat','xmat',
        'xanchor','xaxis','world_xipos','world_ximat','status'))
    attrs('F.frames',pos.frames,('active','body_xpos','body_xquat','body_xmat',
        'xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat','status'))
    attrs('F.mass',pos.mass,('active','xipos','ximat','subtree_com','cinert','cdof','crb','status'))
    attrs('F.mass.factor',pos.mass.mass,('active','qm','qld','qldiag','qLDiagInv','qL','status',
        'warning_count','warning_last'))
    factor=getattr(pos,'factor',None)
    if factor is not None:
        attrs('F.packed_factor',factor,('active','ld','inv','sqrt_inv','pivot','status','warning_count','warning_last'))
    attrs('F.position',pos,('frame_active','mass_active','first_stage','first_code','result_kind'))
    foundation=f.f
    attrs('Foundation',foundation,('admitted','errors','first_stage','first_local','first_item','first_event'))
    for i,g in enumerate(foundation.gates):
        attrs('Foundation.gate'+str(i),g,('active','valid','valid_epoch','forward_epoch','errors',
            'first_stage','first_local','first_item','first_event'))
    attrs('C',f.c,('active','valid','written','error','ncon'))
    cw=f.c.workspace
    attrs('C.workspace',cw,('active','ncon','count','status','fe_code','fe_count','final_count',
        'contact_count','written','valid','attempt'))
    attrs('C.finalize',cw.finalize,('active','ncon','warnings','status','counts'))
    attrs('C.workspace',cw,('effective_active','fe_stage','fe_item'))
    attrs('transport',f.transport,('active','ncon'))
    sw=f.s.workspace
    attrs('S',sw,('status','committed','reference_ready','ref_epoch','position_epoch','counts',
        'nefc','ne','nf','ncon','nrow','nnz','active','attempt','written'))
    attrs('N',n,('valid','written','n_written','n_attempt','n_expected','publication_error'))
    mapping('N.public',n.public_arrays)
    attrs('N.workspace',n.ws,('nefc','ne','nf','ncon','count','empty','newton_world','active',
        'err_code','solver_exit','iters','iteration','qacc','stop_reason','solver_iter',
        'solver_iter_base','solver_nnz','solver_stats_f','solver_stats_i','out_improvement',
        'out_gradient','out_stop','out_reason','choice','s07_status','s08a_status','s08b_status',
        's08c_status','s08d_status','status','route','error_item'))
    event=getattr(n,'final_event',None)
    if event is not None:
        attrs('N.final_event',event,('candidate','candidate_attempt','candidate_completed',
            'candidate_origin','source_stage','rejected_final_entry'))
    mapping('post.public',post.public)
    attrs('post',post,('valid','written','expected_consumed','state_consumed','forward_consumed'))
    attrs('post.work',post.post,('active','status')) if hasattr(post,'post') else None
    for required in ('Euler.qM','Euler.qacc','Euler.qfrc_smooth','Euler.qfrc_constraint',
                     'forward.valid','forward.written','forward.error','forward.expected_forward',
                     'forward.expected_newton','D.forward_epoch','D.source_state_epoch',
                     'N.valid','N.n_written','N.n_attempt','post.valid','post.written',
                     'S.counts','transport.ncon','C.ncon'):
        if required not in roots:raise ValueError('missing required preservation root: '+required)
    # Alias views are intentionally collapsed by allocation span, choosing the
    # widest view at each address so no overlapping copies fight during restore.
    unique={}
    for name,a in roots.items():
        old=unique.get(int(a.ptr))
        if old is None or a.size>old[1].size:unique[int(a.ptr)]=(name,a)
    selected=tuple(unique.values())
    ordered=sorted(selected,key=lambda item:int(item[1].ptr))
    for (_,left),(_,right) in zip(ordered,ordered[1:]):
        end=int(left.ptr)+left.size*wp.types.type_size_in_bytes(left.dtype)
        if end>int(right.ptr):raise ValueError('partially overlapping preservation roots')
    return selected,excluded


class ResetWorldPreserver:
    def __init__(self,bundle,root):
        self.bundle=bundle;self.runtime=r=bundle.runtime;self.mask=r.recovery.reset_mask
        self.roots,self.excluded=collect_roots(bundle)
        self.tokens=tuple(_token(a) for _,a in self.roots);self.mask_token=_token(self.mask)
        self.snapshots=[];save=[];restore=[];position=[];self.inventory=[]
        W=r.euler.capacity
        with wp.ScopedStream(r.stream):
            for name,a in self.roots:
                if not a.size:continue
                b=wp.zeros_like(a);self.snapshots.append(b)
                kind={scalar.wp_real:'f64',wp.int32:'i32',wp.int64:'i64'}[a.dtype]
                width=a.size//W
                save.append(('reset_preservation::save_'+kind,[self.mask,a,b,width],'block'))
                stage=('reset_preservation::restore_'+kind,[self.mask,a,b,width],'block')
                restore.append(stage)
                if name.startswith(('F.body.','F.frames.','F.mass.','F.packed_factor.')) or name in ('Euler.qpos','Euler.qM'):
                    if not name.endswith(('.active','.status','.warning_count','.warning_last')):
                        position.append(stage)
                self.inventory.append(dict(name=name,shape=a.shape,dtype=str(a.dtype),width=width,
                    bytes=a.size*wp.types.type_size_in_bytes(a.dtype)))
        header=(GPU_ROOT / 'cuda/runtime/reset_preservation.cuh')
        self.save_plan=NativePlan('save_nonreset_worlds',[header],save,W,root,
            stage_barriers=[False]*len(save))
        self.restore_plan=NativePlan('restore_nonreset_worlds',[header],restore,W,root,
            stage_barriers=[False]*len(restore))
        self.position_plan=NativePlan("restore_nonreset_position_for_tape",[header],position,W,root,
            stage_barriers=[False]*len(position))
        self.saved_tokens=tuple(_token(a) for a in self.snapshots)
    def validate(self):
        roots,_=collect_roots(self.bundle)
        if tuple(_token(a) for _,a in roots)!=self.tokens or _token(self.mask)!=self.mask_token:
            raise ValueError('prepared nonreset world preservation owner changed')
        if tuple(_token(a) for a in self.snapshots)!=self.saved_tokens:
            raise ValueError('prepared nonreset world snapshots changed')
    def save(self,stream):self.validate();self.save_plan.launch(stream)
    def restore(self,stream):self.validate();self.restore_plan.launch(stream)
    def restore_position(self,stream):self.validate();self.position_plan.launch(stream)


def install(bundle,root):
    r=bundle.runtime
    if not hasattr(r,'recovery'):return {'installed':False,'reason':'no recovery'}
    if r.euler.capacity==1:return {'installed':False,'reason':'singleton has no untouched world'}
    if hasattr(r,'_reset_preserver'):raise ValueError('reset preservation already installed')
    holder=ResetWorldPreserver(bundle,Path(root));r._reset_preserver=holder
    tape=getattr(r.forward,"position_tape",None)
    if tape is not None:
        original_capture=tape.capture
        def capture(owner):
            if getattr(r.forward,"_reset_preservation_scope",False):holder.restore_position(r.stream)
            return original_capture()
        tape.capture=MethodType(capture,tape)
    return dict(installed=True,worlds=r.euler.capacity,roots=holder.inventory,
        excluded=holder.excluded,snapshot_bytes=sum(x['bytes'] for x in holder.inventory),
        timing='inside conditional reset branch; save -> masked forward -> restore before Euler',
        position_tape_supported=True,
        manifests=[holder.save_plan.manifest,holder.restore_plan.manifest,holder.position_plan.manifest])