"""Explicit qualified rigid CCD, retained global scratch and pooled contacts."""
import ctypes as ct
from .mjwarp_plane import PlaneContacts,ContactInputs,ContactParameter,qualify as qualify_plane
from ..dynamics.mjwarp_motion import bind_array,reject_output_overlap
FAMILIES=((3,5),(3,7),(5,6),(5,7),(6,6),(6,7),(7,7))


def select_families(counts,table,*,primitive_boxbox=False,general_multiccd=False,ntypes=11):
    """Cold immutable type census: unqualified pairs retain the whole boundary."""
    if len(counts)!=ntypes*(ntypes+1)//2:raise ValueError('Convex pair-count table span differs')
    active=[];indices=[]
    for pair in table:
        pair=tuple(map(int,pair));a,b=pair
        if pair==(6,6) and primitive_boxbox:continue
        if min(a,b)<0 or max(a,b)>=ntypes:raise ValueError('Convex type table exceeds pinned domain')
        index=a*(2*ntypes-a-1)//2+b
        if counts[index]:active.append(pair);indices.append(index)
    reason=None
    if any(pair not in FAMILIES for pair in active):reason='additional convex pair families retained upstream: '+str([p for p in active if p not in FAMILIES])
    elif general_multiccd and any(pair!=(6,6) for pair in active):reason='general mesh/cylinder MULTICCD retained upstream'
    return active,indices,reason


def qualify(model,data,*,implementation='upstream'):
    from libero.gpu.host.runtime.cuda_api import type_api, collision_table
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('convex_collision must be upstream or cuda')
    admission=qualify_plane(model,data,implementation=implementation)
    if admission['fallback']:admission['fallback']=admission['fallback'].replace('plane collision','convex collision')
    if implementation=='upstream':
        admission.update(pairs=[],pair_indices=[]);return admission
    table=[p for p,kind in collision_table(model).items() if kind==types.CollisionType.CONVEX]
    pairs,indices,reason=select_families(model.geom_pair_type_count,table,primitive_boxbox=bool(model.has_boxbox_margin or model.opt.disableflags & types.DisableBit.NATIVECCD),general_multiccd=not bool(model.opt.disableflags & types.DisableBit.MULTICCD),ntypes=len(types.GeomType))
    if implementation=='cuda' and reason:admission.update(selected='upstream',fallback=reason)
    admission.update(pairs=pairs,pair_indices=indices);return admission


class ConvexCollisionInputs(ct.Structure):
    _fields_=[('contact',ContactInputs),('tolerance',ContactParameter)]+[(k,ct.c_int) for k in ('iterations','epa_iterations','naccd','vertex_capacity','face_capacity','horizon_capacity','warn','active_types')]+[('type_pair',ct.c_int*14),('type_index',ct.c_int*7)]+[(k,ct.c_void_p) for k in ('nccd','overflow','vertices','projection','norm2','vertex_index','face','horizon')]


def validate_workspace(slots,iterations,horizon=24):
    if slots<0 or iterations<0 or horizon<1:raise ValueError('Convex workspace dimensions differ')
    nv,nf=10+2*iterations,6+5*iterations
    if max(slots*nv*3,slots*nf*3,slots*horizon)>2**31-256:raise ValueError('Convex workspace exceeds index storage')
    return nv,nf


class ConvexContacts(PlaneContacts):
    component='convex'
    argument_type=ConvexCollisionInputs
    entry='collision'
    families=['capsule-cylinder','capsule-mesh','cylinder-box','cylinder-mesh','convex-box-box','box-mesh','mesh-mesh']
    def __init__(self,model,data,context,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        super().__init__(model,data,context,library,stream,admission=admission)
        if len(types.GeomType)!=11 or types.MJ_MAX_EPAFACES!=5 or types.MJ_MAX_EPAHORIZON!=24 or int(types.OverflowType.CCD)!=16 or int(types.OverflowType.EPA_HORIZON)!=256:raise ValueError('Pinned convex scratch/overflow constants differ')
        pairs=admission['pairs'];indices=admission['pair_indices'];iterations=int(model.opt.ccd_iterations)
        if len(pairs)>7 or any(p not in FAMILIES for p in pairs) or iterations<0:raise ValueError('Convex admitted family/iteration span differs')
        total=sum(model.geom_pair_type_count[i] for i in indices);box_count=sum(model.geom_pair_type_count[i] for p,i in zip(pairs,indices) if p==(6,6))
        epa_iterations=16 if total and total==box_count else iterations;nv,nf=validate_workspace(data.naccdmax,epa_iterations)
        self.enabled=bool(pairs);self.pairs=tuple(pairs);self.scratch={};a=self.arrays
        tolerance=model.opt.ccd_tolerance
        if tolerance.ndim!=1 or tolerance.shape[0]<1:raise ValueError('Convex tolerance broadcast differs')
        bind_array(a,'ccd_tolerance',tolerance,wp.float32,tolerance.shape,stream)
        bind_array(a,'overflow',data.overflow,wp.int32,(data.nworld,),stream)
        nccd=wp.zeros(66,dtype=wp.int32,device=stream.device);bind_array(a,'nccd',nccd,wp.int32,(66,),stream)
        for key,kind,width in (('vertices',wp.vec3,nv),('projection',wp.vec3,nf),('norm2',wp.float32,nf),('vertex_index',wp.int32,nv),('face',wp.int32,nf),('horizon',wp.int32,24)):
            value=wp.empty((data.naccdmax,width),dtype=kind,device=stream.device);self.scratch[key]=value;bind_array(a,'epa_'+key,value,kind,value.shape,stream)
        reject_output_overlap(a,('nccd','overflow')+tuple('epa_'+k for k in self.scratch)+tuple(k for k in a if k.startswith('contact_')))
        self.contact_arguments=self.arguments
        self.arguments=ConvexCollisionInputs(contact=self.contact_arguments,tolerance=ContactParameter(tolerance.ptr,tolerance.shape[0]),iterations=iterations,epa_iterations=epa_iterations,naccd=data.naccdmax,vertex_capacity=nv,face_capacity=nf,horizon_capacity=24,warn=int(model.opt.warn_overflow),active_types=len(pairs),type_pair=(ct.c_int*14)(*(v for pair in pairs for v in pair)),type_index=(ct.c_int*7)(*indices),**{k:a[k if k in ('nccd','overflow') else 'epa_'+k].ptr for k,_ in ConvexCollisionInputs._fields_[-8:]})
        self.launch.argtypes=[ct.POINTER(ConvexCollisionInputs),ct.c_void_p]
    @property
    def identity(self):
        return dict(super().identity,qualified_pairs=[list(v) for v in self.pairs],has_candidate_families=self.enabled,numerical_workspace_bytes=sum(v.capacity for v in self.scratch.values())+self.arrays['nccd'].capacity,workspace='one retained global EPA pool; serialized typed kernels and Graph substeps',ccd_capacity=self.arguments.naccd,epa_iterations=self.arguments.epa_iterations,private_type_counters=66,general_multiccd=False)
    def enqueue(self):
        if not self.enabled:return
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex collision launch failed: CUDA error '+str(error))
