"""Load-only RNE suffix; borrows current solved acceleration/external wrenches."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id, load_leaf, bind_array, reject_output_overlap


class PostconstraintInputs(ct.Structure):
    _fields_ = [(k,ct.c_int) for k in ('worlds','nv','bodies','branches','gravity_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('body_dofnum','body_dofadr','branch_bodies','branch_start',
        'body_owner','body_parent','gravity','qvel','qacc','cdof','cinert','cvel','cdof_dot','cfrc_ext','cacc','cfrc_int')]


class PostconstraintRNE:
    def __init__(self,model,data,library,stream,topology):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if topology.model is not model or topology.data is not data or topology.stream is not stream:
            raise ValueError('Postconstraint topology owner differs')
        self.model,self.data,self.stream,self.topology=model,data,stream,topology
        self.path,self.library=load_leaf('postconstraint',PostconstraintInputs,library,stream)
        self.gravity_disabled=bool(model.opt.disableflags & types.DisableBit.GRAVITY)
        self.arrays=a=dict(topology.arrays);n,nv,nb=data.nworld,model.nv,model.nbody
        gravity=model.opt.gravity
        if gravity.ndim!=1 or gravity.shape[0]<1 or gravity.shape[0]*3>2**31-1:
            raise ValueError('Postconstraint gravity broadcast shape differs')
        bind_array(a,'gravity',gravity,wp.vec3,tuple(gravity.shape),stream)
        for key,dtype,shape in (('qvel',wp.float32,(n,nv)),('qacc',wp.float32,(n,nv)),
                ('cdof',wp.spatial_vector,(n,nv)),('cinert',types.vec10,(n,nb)),
                ('cvel',wp.spatial_vector,(n,nb)),('cdof_dot',wp.spatial_vector,(n,nv)),
                ('cfrc_ext',wp.spatial_vector,(n,nb)),('cacc',wp.spatial_vector,(n,nb)),
                ('cfrc_int',wp.spatial_vector,(n,nb))):
            bind_array(a,key,getattr(data,key),dtype,shape,stream)
        reject_output_overlap(a,('cacc','cfrc_int'))
        self.arguments=PostconstraintInputs(worlds=n,nv=nv,bodies=nb,branches=model.nbranch,gravity_rows=gravity.shape[0],
            **{k:a[k].ptr for k,_ in PostconstraintInputs._fields_[5:]})
        self.launch=self.library.flash_mjwarp_postconstraint_launch
        self.launch.argtypes=[ct.POINTER(PostconstraintInputs),ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    def prepare_fusion(self):
        if hasattr(self,'fused_launch'):return self
        self.tree=self.topology.prepare_layer_tree()
        self.interval=self.tree.prepare_intervals()
        self.fusion_eligible=self.interval is not None and self.model.nbranch<=32
        self.fused_launch=self.library.flash_mjwarp_postconstraint_fused
        self.fused_launch.argtypes=[ct.POINTER(PostconstraintInputs),ct.c_void_p,ct.c_int,ct.c_void_p]
        self.fused_launch.restype=ct.c_int
        return self

    def enqueue_fused(self):
        if not self.fusion_eligible:self.enqueue();return
        error=self.fused_launch(ct.byref(self.arguments),self.interval.ptr,
                                int(self.gravity_disabled),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused postconstraint failed: CUDA error '+str(error))

    @property
    def identity(self):
        return dict(source_id=source_id('postconstraint'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            gravity_disabled=self.gravity_disabled,values='borrow-current-input-no-cache',**self.topology.identity)

    def enqueue(self):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(self.stream)
        with wp.ScopedStream(self.stream):
            if self.gravity_disabled:self.arrays['cacc'].zero_()
            commands=[(p,None,0) for p in ([1,2] if self.gravity_disabled else [0,1,2])]
            commands.extend((3,a.ptr,a.size) for a in reversed(self.topology.layers) if a.size)
            for phase,layer,count in commands:
                error=self.launch(ct.byref(self.arguments),phase,layer,count,self.stream.cuda_stream)
                if error:raise RuntimeError('Postconstraint RNE launch failed: CUDA error '+str(error))
