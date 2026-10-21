"""Load-only pinned NXN broadphase; current bounds and borrowed pair buffers."""
import ctypes as ct
import hashlib
from ..dynamics.mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def qualify(model,data,*,implementation='upstream'):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('broadphase must be upstream or cuda')
    reason=None
    if implementation=='cuda':
        if model.opt.integrator!=types.IntegratorType.EULER:reason='collision caller requires Euler'
        elif model.opt.enableflags & types.EnableBit.SLEEP:reason='sleep/incremental collision retained upstream'
        elif model.opt.broadphase!=types.BroadphaseType.NXN:reason='SAP collision retained upstream'
        elif model.nflex:reason='flex collision caller retained upstream'
        elif model.ngeom<1 or data.nworld<1:reason='empty broadphase domain'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,model=id(model),data=id(data))


class BroadphaseInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','geoms','pairs','explicit_pairs','capacity','filter','dataid_rows','aabb_rows','rbound_rows','margin_rows','gap_rows','pair_margin_rows','pair_gap_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('geom_type','geom_dataid','pairs_in','pair_ids_in','aabb','rbound','margin','gap','pair_margin','pair_gap','xpos','xmat','count','pairs_out','pair_ids_out','world_out')]


def validate_pairs(geom_type,pairs,ids,npair,nsensor):
    import numpy as np
    ng=len(geom_type)
    if np.any(geom_type<0) or np.any(geom_type>8):raise ValueError('Broadphase geometry type differs')
    if np.any(pairs<0) or np.any(pairs>=ng) or np.any(pairs[:,0]>=pairs[:,1]) or len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Broadphase filtered geometry pairs differ')
    if ids.shape!=pairs.shape or np.any(ids[:,0]<-2) or np.any(ids[:,0]>=npair) or np.any(ids[:,1]<-1) or np.any(ids[:,1]>=nsensor):raise ValueError('Broadphase pair metadata exceeds storage')
    if np.any((ids[:,0]<-1)&(ids[:,1]<0)):raise ValueError('Broadphase prefiltered pair is excluded')


class BroadphaseNumerics:
    def __init__(self,model,data,context,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Broadphase admission owner differs')
        if admission['requested']!='cuda' or admission['selected']!='cuda':raise ValueError('Broadphase CUDA domain not admitted')
        if tuple(int(x) for x in (types.GeomType.MESH,types.BroadphaseType.NXN,types.BroadphaseFilter.PLANE,types.BroadphaseFilter.SPHERE,types.BroadphaseFilter.AABB,types.BroadphaseFilter.OBB))!=(7,0,1,2,4,8) or types.MJ_MAXVAL!=1.e10:raise ValueError('Pinned broadphase enums/constants differ')
        n,ng,nc,npair=data.nworld,model.ngeom,data.naconmax,model.npair
        pairs=model.nxn_geom_pair_filtered
        if pairs.ndim!=1:raise ValueError('Broadphase pair layout differs')
        count=pairs.shape[0]
        if nc<0 or npair<0 or max(n*ng*9,n*count,nc*2,count*2,npair,model.nsensorcollision)>2**31-256:raise ValueError('Broadphase dimensions exceed index storage')
        self.model,self.data,self.context,self.stream=model,data,context,stream
        self.path,self.library=load_leaf('broadphase',BroadphaseInputs,library,stream)
        self.arrays=a={}
        bind_array(a,'geom_type',model.geom_type,wp.int32,(ng,),stream)
        bind_array(a,'pairs_in',pairs,wp.vec2i,(count,),stream)
        bind_array(a,'pair_ids_in',model.nxn_pairid_filtered,wp.vec2i,(count,),stream)
        validate_pairs(*(a[k].numpy() for k in ('geom_type','pairs_in','pair_ids_in')),npair,model.nsensorcollision)
        rows={}
        for key,dtype,width,shape_tail,components in (('geom_dataid',wp.int32,ng,(ng,),1),('aabb',wp.vec3,ng,(ng,2),6),
                ('rbound',wp.float32,ng,(ng,),1),('margin',wp.float32,ng,(ng,),1),('gap',wp.float32,ng,(ng,),1),
                ('pair_margin',wp.float32,npair,(npair,),1),('pair_gap',wp.float32,npair,(npair,),1)):
            value=getattr(model,key if key.startswith(('geom_','pair_')) else 'geom_'+key)
            if value.ndim!=1+len(shape_tail) or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:raise ValueError('Broadphase broadcast differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],)+shape_tail,stream);rows[('dataid' if key=='geom_dataid' else key)+'_rows']=value.shape[0]
        for key,field,dtype,shape in (('xpos','geom_xpos',wp.vec3,(n,ng)),('xmat','geom_xmat',wp.mat33,(n,ng)),('count','ncollision',wp.int32,(1,))):bind_array(a,key,getattr(data,field),dtype,shape,stream)
        for key,field,dtype in (('pairs_out','collision_pair',wp.vec2i),('pair_ids_out','collision_pairid',wp.vec2i),('world_out','collision_worldid',wp.int32)):
            bind_array(a,key,getattr(context,field),dtype,(nc,),stream)
        reject_output_overlap(a,('count','pairs_out','pair_ids_out','world_out'))
        self.arguments=BroadphaseInputs(worlds=n,geoms=ng,pairs=count,explicit_pairs=npair,capacity=nc,filter=int(model.opt.broadphase_filter),**rows,
            **{k:a[k].ptr for k,_ in BroadphaseInputs._fields_[13:]})
        self.launch=self.library.flash_mjwarp_broadphase_launch
        self.launch.argtypes=[ct.POINTER(BroadphaseInputs),ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('broadphase'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),numerical_workspace_bytes=0,shared_bytes=0,
            values='borrow-current-input-no-cache',algorithm='pinned prefiltered NXN pairs, plane/sphere/AABB/six-axis OBB',allocation='pooled FP32-path integer atomics; attempted count retained beyond capacity')

    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Broadphase launch failed: CUDA error '+str(error))
