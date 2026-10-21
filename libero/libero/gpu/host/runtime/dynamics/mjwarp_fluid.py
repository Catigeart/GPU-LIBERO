"""Load-only inertia-box fluid, current inputs and one retained wrench workspace."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def caller_unsupported_reason(model,data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator!=types.IntegratorType.EULER:return 'fluid caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if min(data.nworld,model.nv,model.nq,model.njnt)<1 or model.nbody<2:return 'empty fluid domain'
    if model.ntendon:return 'tendon passive caller retained upstream'
    if any(getattr(model,k) for k in ('nflex','nflexedge','nflexelem','nflexbend_interp','nflexintcell')):return 'flex passive caller retained upstream'
    if not model.has_fluid:return 'fluid disabled at model import'
    return None


def qualify(model,data,*,implementation='upstream'):
    import numpy as np
    if implementation not in ('upstream','cuda'):raise ValueError('fluid must be upstream or cuda')
    reason=caller_unsupported_reason(model,data) if implementation=='cuda' else None
    if implementation=='cuda' and reason is None:
        flags=model.body_fluid_ellipsoid.numpy()
        if flags.shape!=(model.nbody,):raise ValueError('Fluid ellipsoid flag dimensions differ')
        if np.any(flags):reason='ellipsoid fluid retained upstream'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,model=id(model),data=id(data))


def validate_topology(parent,root,subtree,dof_body):
    import numpy as np
    nb=len(parent)
    if nb<2 or len(root)!=nb or len(subtree)!=nb or parent[0]!=0:raise ValueError('Fluid topology dimensions differ')
    if np.any(parent[1:]<0) or np.any(parent[1:]>=np.arange(1,nb)):raise ValueError('Fluid parent order differs')
    counts=np.ones(nb,int);roots=np.zeros(nb,int)
    for b in range(nb-1,0,-1):counts[parent[b]]+=counts[b]
    for b in range(1,nb):roots[b]=b if parent[b]==0 else roots[parent[b]]
    if not np.array_equal(root,roots) or not np.array_equal(subtree,counts):raise ValueError('Fluid root/subtree metadata differs')
    for b in range(1,nb):
        if b+counts[b]>parent[b]+counts[parent[b]]:raise ValueError('Fluid body order is not contiguous DFS')
    if np.any(dof_body<1) or np.any(dof_body>=nb):raise ValueError('Fluid DOF body exceeds storage')


class FluidInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','nv','mass_rows','inertia_rows','wind_rows','density_rows','viscosity_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('root','dof_body','subtree_size','mass','inertia','wind','density','viscosity',
        'xipos','ximat','subtree_com','cvel','cdof','body_force','qfrc')]


class FluidNumerics:
    def __init__(self,model,data,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Fluid admission owner differs')
        if admission['requested']!='cuda' or admission['selected']!='cuda':raise ValueError('Fluid CUDA domain not admitted')
        n,nb,nv=data.nworld,model.nbody,model.nv
        if max(n*nb*9,n*nv*6)>2**31-256:raise ValueError('Fluid dimensions exceed index storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('fluid',FluidInputs,library,stream)
        a=self.arrays={};static={}
        for key,field,width in (('parent','body_parentid',nb),('root','body_rootid',nb),('subtree_size','body_subtreenum',nb),('dof_body','dof_bodyid',nv)):
            static[key]=bind_array(a,key,getattr(model,field),wp.int32,(width,),stream).numpy()
        validate_topology(*(static[k] for k in ('parent','root','subtree_size','dof_body')))
        bind_array(a,'ellipsoid',model.body_fluid_ellipsoid,wp.bool,(nb,),stream)
        rows={}
        for key,dtype,width,components in (('mass',wp.float32,nb,1),('inertia',wp.vec3,nb,3)):
            value=getattr(model,'body_'+key)
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:raise ValueError('Fluid model broadcast differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],width),stream);rows[key+'_rows']=value.shape[0]
        for key,dtype,components in (('wind',wp.vec3,3),('density',wp.float32,1),('viscosity',wp.float32,1)):
            value=getattr(model.opt,key)
            if value.ndim!=1 or value.shape[0]<1 or value.shape[0]*components>2**31-256:raise ValueError('Fluid option broadcast differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],),stream);rows[key+'_rows']=value.shape[0]
        for key,field,width,dtype in (('xipos','xipos',nb,wp.vec3),('ximat','ximat',nb,wp.mat33),('subtree_com','subtree_com',nb,wp.vec3),
                ('cvel','cvel',nb,wp.spatial_vector),('cdof','cdof',nv,wp.spatial_vector),('qfrc','qfrc_fluid',nv,wp.float32)):
            bind_array(a,key,getattr(data,field),dtype,(n,width),stream)
        a['body_force']=wp.empty((n,nb),dtype=wp.spatial_vector,device=stream.device)
        reject_output_overlap(a,('body_force','qfrc'))
        self.arguments=FluidInputs(worlds=n,bodies=nb,nv=nv,**rows,**{k:a[k].ptr for k,_ in FluidInputs._fields_[8:]})
        self.launch=self.library.flash_mjwarp_fluid_launch
        self.launch.argtypes=[ct.POINTER(FluidInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int
        self.fused_launch=self.library.flash_mjwarp_fluid_fused
        self.fused_launch.argtypes=[ct.POINTER(FluidInputs),ct.c_void_p];self.fused_launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('fluid'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            numerical_workspace_bytes=self.arrays['body_force'].capacity,topology_bytes=0,
            workspace='one retained (world,body,force-then-torque) array; reused across Graph substeps',values='borrow-current-input-no-cache')

    def enqueue(self,phase=None):
        if phase is None:self.enqueue(0);self.enqueue(1);return
        if phase not in (0,1):raise ValueError('Fluid phase must be 0 or 1')
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Fluid launch failed: CUDA error '+str(error))

    def enqueue_fused(self):
        error=self.fused_launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused fluid failed: CUDA error '+str(error))
