"""Load-only applied/contact wrench assembly; borrows all Model/Data storage."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def qualify(model,data,*,implementation='upstream'):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('external_wrench must be upstream or cuda')
    reason=None
    if implementation=='cuda':
        if model.opt.integrator!=types.IntegratorType.EULER:reason='external wrench caller requires Euler'
        elif model.opt.enableflags & types.EnableBit.SLEEP:reason='active-body compaction'
        elif data.nworld<1 or model.nbody<1:reason='empty external wrench domain'
        elif model.neq:reason='equality external wrenches retained upstream'
        elif model.ntendon:reason='tendon external wrenches retained upstream'
        elif any(getattr(model,k) for k in ('nflex','nflexedge','nflexelem','nflexbend_interp','nflexintcell')):reason='flex external wrenches retained upstream'
        elif model.opt.cone not in (types.ConeType.PYRAMIDAL,types.ConeType.ELLIPTIC):reason='unsupported contact cone'
        elif model.nmaxcondim not in (1,3,4,6):reason='unsupported maximum contact dimension'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,model=id(model),data=id(data))


def validate_topology(parent,root,geom_body):
    import numpy as np
    nb=len(parent)
    if nb<1 or len(root)!=nb or parent[0]!=0:raise ValueError('External wrench topology dimensions differ')
    if np.any(parent[1:]<0) or np.any(parent[1:]>=np.arange(1,nb)):raise ValueError('External wrench parent order differs')
    expected=np.zeros(nb,np.int32)
    for b in range(1,nb):expected[b]=b if parent[b]==0 else expected[parent[b]]
    if not np.array_equal(root,expected):raise ValueError('External wrench root references differ')
    if np.any(geom_body<0) or np.any(geom_body>=nb):raise ValueError('External wrench geom body exceeds storage')


class ExternalWrenchInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','geoms','contact_capacity','rows','address_columns','cone')]
    _fields_ += [(k,ct.c_void_p) for k in ('root','geom_body','contact_dim','contact_geom','contact_address','contact_world','contact_count',
        'applied','xipos','subtree_com','contact_pos','contact_frame','contact_friction','contact_adhesion','constraint_force','external')]


class ExternalWrenchNumerics:
    def __init__(self,model,data,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('External wrench admission owner differs')
        if admission['requested']!='cuda' or admission['selected']!='cuda':raise ValueError('External wrench CUDA domain not admitted')
        if (int(types.ConeType.PYRAMIDAL),int(types.ConeType.ELLIPTIC))!=(0,1):raise ValueError('Pinned contact cone representation differs')
        n,nb,ng,nc,nj=data.nworld,model.nbody,model.ngeom,data.naconmax,data.njmax
        address=data.contact.efc_address
        if address.ndim!=2 or address.shape[1]<max(1,model.nmaxcondim):raise ValueError('External wrench address columns differ')
        columns=address.shape[1]
        if min(ng,nc,nj)<0 or max(n*nb*6,n*nj,nc*max(9,columns,5),ng)>2**31-256:raise ValueError('External wrench dimensions exceed index storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('external_wrench',ExternalWrenchInputs,library,stream)
        self.arrays=a={}
        parent=bind_array(a,'parent',model.body_parentid,wp.int32,(nb,),stream)
        root=bind_array(a,'root',model.body_rootid,wp.int32,(nb,),stream)
        geom=bind_array(a,'geom_body',model.geom_bodyid,wp.int32,(ng,),stream)
        validate_topology(parent.numpy(),root.numpy(),geom.numpy())
        for key,field,dtype,shape in (('applied','xfrc_applied',wp.spatial_vector,(n,nb)),('xipos','xipos',wp.vec3,(n,nb)),
                ('subtree_com','subtree_com',wp.vec3,(n,nb)),('external','cfrc_ext',wp.spatial_vector,(n,nb)),('contact_count','nacon',wp.int32,(1,))):
            bind_array(a,key,getattr(data,field),dtype,shape,stream)
        for key,field,dtype,shape in (('contact_dim','dim',wp.int32,(nc,)),('contact_geom','geom',wp.vec2i,(nc,)),
                ('contact_address','efc_address',wp.int32,(nc,columns)),('contact_world','worldid',wp.int32,(nc,)),
                ('contact_pos','pos',wp.vec3,(nc,)),('contact_frame','frame',wp.mat33,(nc,)),
                ('contact_friction','friction',types.vec5,(nc,)),('contact_adhesion','adhesion',wp.float32,(nc,))):
            bind_array(a,key,getattr(data.contact,field),dtype,shape,stream)
        bind_array(a,'constraint_force',data.efc.force,wp.float32,(n,nj),stream)
        reject_output_overlap(a,('external',))
        self.arguments=ExternalWrenchInputs(worlds=n,bodies=nb,geoms=ng,contact_capacity=nc,rows=nj,address_columns=columns,cone=int(model.opt.cone),
            **{k:a[k].ptr for k,_ in ExternalWrenchInputs._fields_[7:]})
        self.launch=self.library.flash_mjwarp_external_wrench_launch
        self.launch.argtypes=[ct.POINTER(ExternalWrenchInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('external_wrench'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            numerical_workspace_bytes=0,topology_bytes=0,values='borrow-current-input-no-cache',
            accumulation='pinned FP32 componentwise contact atomics',cone=self.arguments.cone,address_columns=self.arguments.address_columns)

    def enqueue(self,phase=None):
        if phase is None:self.enqueue(0);self.enqueue(1);return
        if phase not in (0,1):raise ValueError('External wrench phase must be 0 or 1')
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('External wrench launch failed: CUDA error '+str(error))
