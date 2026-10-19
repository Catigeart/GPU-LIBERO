"""Exact-bit, resident prefix reuse; force/Newton/post are never cached.

A device predicate skips the prefix only when every admitted world hits.
Mixed hit/miss batches recompute, preserving all existing phase masks. Graph
callbacks are inserted inline into an outer capture; ordinary enqueue launches
one pre-captured graph, with no predicate readback or host synchronization.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
import warp as wp
from libero.gpu.host.build.native import NativePlan


def _position_prefix(f):
    from libero.gpu.host.runtime.step.physics import _c_mask
    f.f.launch_position(stream=f.stream)
    f._launch(_c_mask,[f.d.active,f.d.valid['position'],f.d.written['position'],
        f.expected_forward,f.d.first_error,f.c_mask])
    f.c.enqueue(f.c_mask,f.stream)
    f.transport.enqueue()
    if f.position_tape is not None:f.position_tape.capture()
    f.s.launch_position()
    # Scene callback is attached after environment assembly; invoke outside cache.


def _velocity_prefix(f):
    from libero.gpu.host.runtime.step.physics import _reference_mask
    f.f.launch_velocity(stream=f.stream)
    f._launch(_reference_mask,[f.d.valid['velocity'],f.d.written['velocity'],
        f.expected_forward,f.d.first_error,f.transport.active])
    sw=f.s.workspace
    f.s.launch_reference()
    f.d.launch_bias(reference_done=sw.reference_ready,reference_error=sw.status,
        reference_written=sw.ref_epoch,reference_expected=f.expected_forward,stream=f.stream)


def _prefix(f):
    _position_prefix(f)
    _velocity_prefix(f)


def _begin(f,mask,stream,step1):
    from libero.gpu.host.runtime.step.shape_physics import _invalidate_full
    if stream is not f.stream:raise ValueError('foreign forward stream')
    f.validate()
    if step1:f._launch(_invalidate_full,[f.valid])
    if mask is not f.mask:wp.copy(f.mask,mask,stream=f.stream)
    f.f.begin_forward_with_plan(f.mask,stream=f.stream,plan=f.expected_forward)
    f._dataflow_reuse.enqueue()
    if f.scene_transform is not None:f.scene_transform(f.c_mask,f.stream)
    if getattr(f._dataflow_reuse,'position_only',False):_velocity_prefix(f)


def _enqueue(f,mask,stream):
    from libero.gpu.host.runtime.step.physics import _publish_forward
    _begin(f,mask,stream,False)
    f.d.launch_force(ctrl=f.ctrl,ctrl_valid=f.ctrl_valid,stream=f.stream)
    f._fusion_project_newton()
    if f.before_constraint is not None:f.before_constraint(f.stream)
    if f._fusion_entry_copy is not None:
        f._fusion_entry_copy.launch(f.stream)
    f.n.enqueue(f.n_upstream,active=f.mask,expected_epoch=f.expected_newton,
                publication_status=f.projection_status)
    if f.acceleration_sensor is not None:f.acceleration_sensor(f.n_admitted,f.stream)
    f._launch(_publish_forward,[f.d.active,f.final_valid,f.n.ws.err_code,
        f.d.first_error,f.projection_status,f.d.source_state_epoch,f.d.state_epoch,
        f.valid,f.written,f.error])


def _step1(f,mask,stream):_begin(f,mask,stream,True)


class ResidentPrefixReuse:
    def __init__(self,bundle,root):
        self.forward=f=bundle.runtime.forward;d=f.d;sw=f.s.workspace;W=f.capacity
        self.stream=f.stream;self.plans={};self.keys=[]
        with wp.ScopedStream(self.stream):
            self.saved=wp.zeros(W,dtype=wp.int32,device=d.device)
            self.hit=wp.zeros(W,dtype=wp.int32,device=d.device)
            self.miss=wp.zeros(1,dtype=wp.int32,device=d.device)
            self.math=wp.zeros(W,dtype=wp.uint64,device=d.device)
            self.consumed=wp.zeros(W,dtype=wp.uint64,device=d.device)
        # Model/options are immutable prepared holders validated by f.validate.
        # Both state epoch and exact bytes are checked: direct resident mutation
        # without advancing the producer epoch still invalidates this cache.
        fields={'state_epoch':d.state_epoch,'qpos':d.qpos,'qvel':d.qvel}
        body=d.stages.position.body
        for name in ('mocap_pos','mocap_quat'):
            a=getattr(body,name,None)
            if isinstance(a,wp.array):fields[name]=a
        for name in ('expected_model_epoch','device_model_epoch'):
            a=getattr(bundle,name,None)
            if isinstance(a,wp.array):fields[name]=a
        for name in ('rhs','count'):
            fields['factor.'+name]=getattr(d.stages.position.factor,name)
        fields.update({'C.fe_code':f.c.workspace.fe_code,'S.status':sw.status,'S.committed':sw.committed,'S.reference_ready':sw.reference_ready})
        shared={}
        def arrays(prefix,owner,names=None):
            values=owner if isinstance(owner,dict) else vars(owner)
            for name,a in values.items():
                if isinstance(a,wp.array) and (names is None or name in names):
                    shared[prefix+'.'+name]=a
        pos=d.stages.position
        # Explicit mathematical input holders, not arbitrary workspace traversal.
        arrays('body.static',pos.body.static)
        arrays('frames.static',pos.frames.static)
        arrays('compos',pos.compos)
        arrays('crb_model',pos.crb_model)
        arrays('factor.topology',pos.factor,('dof_Madr','dof_parentid','dof_simplenum'))
        arrays('transmission',d.stages.transmission,('qadr','dadr','gear'))
        arrays('comvel_model',d.stages.comvel_model)
        arrays('rne_model',d.stages.rne_model)
        arrays('passive',d.stages.passive,('jtype','qadr','dadr','stiffness','spring','damping',
            'rootid','mass','box','ancestor','wind','opt'))
        cw=f.c.workspace
        arrays('collision',cw,('body_geomadr','body_geomnum','body_weldid','body_parentid',
            'geom_bodyid','geom_type','geom_contype','geom_conaffinity','geom_rbound','geom_margin',
            'filterparent_disabled','midphase_disabled','can','plane','pair_signature','exclude_signature'))
        arrays('collision.tree',cw.tree,('body_adr','body_num','geom_adr','geom_num','child','geomid',
            'gtype','pair1','pair2','bvh_aa','geom_aa','rbound'))
        arrays('collision.dispatch',cw.dispatch.static)
        arrays('collision.detect',cw.detect,('geom_size','geom_dataid','geom_rbound','mesh_vert',
            'mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','route_id'))
        arrays('collision.params',cw.finalize.params.static)
        arrays('S.prediag',sw,('geom_body','body_inv','dof_inv','jnt_dofadr'))
        arrays('S.contact',sw.s01,('geom_body','roots','ancestor'))
        arrays('S.scalar',sw.s02.static)
        arrays('S.impedance',sw.s03.static)
        # Capacity/incoming are prepared operands, not outputs. A dirty mutation
        # must cause the original device gate to run rather than publish a hit.
        for name in ('zero_incoming','row_cap_flat'):
            fields['S.'+name]=getattr(sw,name)
        seen=set()
        for global_key,mapping in ((False,fields),(True,shared)):
            for name,a in mapping.items():
                if not a.is_contiguous or (not global_key and a.size%W):
                    raise ValueError('cache input layout unsupported: '+name)
                span=(int(a.ptr),a.size,a.dtype,global_key)
                if span in seen:continue
                seen.add(span)
                saved=wp.zeros(W*a.size,dtype=a.dtype,device=a.device) if global_key else wp.zeros_like(a)
                width=a.size*wp.types.type_size_in_bytes(a.dtype)//(1 if global_key else W)
                self.keys.append((name,a,saved,width,(int(a.ptr),a.shape,a.strides,a.dtype),global_key))
        source=(GPU_ROOT / 'cuda/runtime/reuse.cuh')
        def plan(name,stages,worlds=W):
            p=NativePlan('prefix_reuse_'+name,[source],stages,worlds,Path(root)/name)
            self.plans[name]=p;return p
        check=[('forward_reuse::start',[d.active,d.first_error,self.saved,self.hit],'lane0')]
        for _,a,saved,width,_,shared in self.keys:
            check.append(('forward_reuse::'+('compare_shared' if shared else 'compare'),[a,saved,width,self.hit],'block',
                          ['const unsigned char*','const unsigned char*','int','int*']))
        self.check=plan('check',check)
        self.predicate=plan('predicate',[('forward_reuse::predicate',
            [d.active,self.hit,W,self.miss],'lane0')],worlds=1)
        store=[]
        for _,a,saved,width,_,shared in self.keys:
            store.append(('forward_reuse::'+('save_shared' if shared else 'save_bytes'),[d.active,d.first_error,d.valid['bias'],
                sw.reference_ready,sw.status,a,saved,width],'block',
                ['const int*','const int*','const int*','const int*','const int*',
                 'const unsigned char*','unsigned char*','int']))
        self.snapshot=plan('snapshot',store)
        store=[('forward_reuse::save_valid',[d.active,d.first_error,d.valid['bias'],
                sw.reference_ready,sw.status,self.saved,self.math],'lane0')]
        self.store=plan('store',store)
        self.publish=plan('publish',[('forward_reuse::publish',[d.active,self.hit,d.first_error,
            d.forward_epoch,*[d.valid[k] for k in ('position','velocity','bias')],
            *[d.written[k] for k in ('position','velocity','bias')],sw.ref_epoch,self.consumed,f.c_mask,f.transport.active],'lane0')])
        self._graph=None

    def miss_body(self):
        self.snapshot.launch(self.stream)
        _prefix(self.forward)
        self.store.launch(self.stream)

    def branch(self):
        wp.capture_if(self.miss,on_true=self.miss_body,on_false=self.hit_body,stream=self.stream)

    def hit_body(self):
        self.publish.launch(self.stream)
        if self.forward.position_tape is not None:self.forward.position_tape.capture()

    def prepare_graph(self):
        # Cold only. Conditional bodies retain callbacks into the final installed
        # C/S/D implementations. No arithmetic executes during capture.
        with wp.ScopedCapture(stream=self.stream,force_module_load=True) as cap:self.branch()
        self._graph=cap.graph

    def enqueue(self):
        for name,a,_,_,token,_ in self.keys:
            if (int(a.ptr),a.shape,a.strides,a.dtype)!=token:
                raise ValueError('prepared reuse input changed: '+name)
        self.check.launch(self.stream);self.predicate.launch(self.stream)
        if self.stream.is_capturing:self.branch()
        else:wp.capture_launch(self._graph,stream=self.stream)


def install(bundle,build_root,*,controller_partial_refresh=False):
    f=bundle.runtime.forward
    if not hasattr(f,'_fusion_project_newton') or not hasattr(f,'_fusion_entry_copy'):
        return dict(installed=False,blocked='requires fused projection/entry-copy hooks')
    cache=ResidentPrefixReuse(bundle,build_root);f._dataflow_reuse=cache
    f.enqueue=MethodType(_enqueue,f);f.enqueue_step1=MethodType(_step1,f)
    controller=bundle.controller
    if controller is not None:
        # Preserve full N/warmstart invocation by default. Partial refresh is an
        # explicit semantic option, since original full refresh modifies solver history.
        controller.forward_refresh=f.enqueue_step1 if controller_partial_refresh else f.enqueue
        controller._callbacks=(controller.forward_refresh,controller.gather,controller.live_bias)
    cache.prepare_graph()
    bundle.cuda_reuse_plans=cache.plans
    return dict(installed=True,candidates=['01'],key_fields=[k[0] for k in cache.keys],
        snapshot_bytes=sum(k[2].size*wp.types.type_size_in_bytes(k[2].dtype) for k in cache.keys),
        broadcast_key_fields=[k[0] for k in cache.keys if k[5]],
        decision='all admitted worlds exact-bit hit; mixed hits recompute batch; broadcast snapshots independent per world',
        reused='position/collision/S-position/velocity/reference/bias',
        uncached='force/Newton/warmstart/post',controller_partial_refresh=controller_partial_refresh,
        lineage='cache hit republishes current logical consumption; math counters increment only on miss',
        plans={k:p.manifest for k,p in cache.plans.items()})


def install_dataflow(bundle,build_root,*,diagnostic_profile='compatible',controller_partial_refresh=False):
    if diagnostic_profile=='compatibility':diagnostic_profile='compatible'
    if diagnostic_profile not in ('compatible','fast'):
        raise ValueError('diagnostic_profile must be compatible or fast')
    result=install(bundle,build_root,controller_partial_refresh=controller_partial_refresh)
    result['diagnostic_profile']=diagnostic_profile
    result['fast_semantics']='full controller Newton/warmstart refresh retained by default; partial refresh is explicit experimental opt-in'
    return result
