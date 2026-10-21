"""FIXED camera/light admission and load-only binding of borrowed world frames."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def caller_unsupported_reason(model,data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator!=types.IntegratorType.EULER:return 'camlight caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if data.nworld<1 or model.nbody<1:return 'empty camera/light world domain'
    if model.ncam<0 or model.nlight<0:raise ValueError('CamLight counts must be nonnegative')
    if model.ncam+model.nlight==0:return 'empty camera/light domain'
    return None


def qualify(model,data,*,implementation='upstream'):
    """Read immutable modes once before optional product admission/loading."""
    import numpy as np
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('camlight must be upstream or cuda')
    reason=caller_unsupported_reason(model,data) if implementation=='cuda' else None
    if implementation=='cuda' and reason is None:
        if int(types.CamLightType.FIXED)!=0:raise ValueError('Pinned FIXED mode representation differs')
        for field,width in (('cam_mode',model.ncam),('light_mode',model.nlight)):
            modes=getattr(model,field).numpy()
            if modes.shape!=(width,):raise ValueError('CamLight mode dimensions differ: '+field)
            if np.any(modes!=0):reason='non-FIXED camera/light modes retained upstream'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,
                model=id(model),data=id(data))


def validate_indices(nb,cam_body,light_body):
    import numpy as np
    for bodies in (cam_body,light_body):
        if np.any(bodies<0) or np.any(bodies>=nb):raise ValueError('CamLight attachment body exceeds storage')


class CamLightInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','cameras','lights','cam_pos_rows','cam_quat_rows','light_pos_rows','light_dir_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('cam_body','light_body','cam_pos','cam_quat','light_pos','light_dir','xpos','xquat',
                                          'cam_xpos','cam_xmat','light_xpos','light_xdir')]


class CamLightFrames:
    def __init__(self,model,data,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('CamLight admission owner differs')
        if admission['requested']!='cuda' or admission['selected']!='cuda':raise ValueError('CamLight CUDA domain not admitted')
        n,nb,nc,nl=data.nworld,model.nbody,model.ncam,model.nlight
        if max(n*nb*4,n*nc*9,n*nl*3)>2**31-256:raise ValueError('CamLight dimensions exceed index storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('camlight',CamLightInputs,library,stream)
        a=self.arrays={}
        for key,field,width in (('cam_body','cam_bodyid',nc),('light_body','light_bodyid',nl),('cam_mode','cam_mode',nc),('light_mode','light_mode',nl)):
            bind_array(a,key,getattr(model,field),wp.int32,(width,),stream)
        validate_indices(nb,a['cam_body'].numpy(),a['light_body'].numpy())
        rows={}
        for key,width,dtype,components in (('cam_pos',nc,wp.vec3,3),('cam_quat',nc,wp.quat,4),('light_pos',nl,wp.vec3,3),('light_dir',nl,wp.vec3,3)):
            value=getattr(model,key)
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:
                raise ValueError('CamLight broadcast storage differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],width),stream);rows[key+'_rows']=value.shape[0]
        for key,width,dtype in (('xpos',nb,wp.vec3),('xquat',nb,wp.quat),('cam_xpos',nc,wp.vec3),('cam_xmat',nc,wp.mat33),
                                ('light_xpos',nl,wp.vec3),('light_xdir',nl,wp.vec3)):
            bind_array(a,key,getattr(data,key),dtype,(n,width),stream)
        reject_output_overlap(a,('cam_xpos','cam_xmat','light_xpos','light_xdir'))
        self.arguments=CamLightInputs(worlds=n,bodies=nb,cameras=nc,lights=nl,**rows,
            **{k:a[k].ptr for k,_ in CamLightInputs._fields_[8:]})
        self.launch=self.library.flash_mjwarp_camlight_launch
        self.launch.argtypes=[ct.POINTER(CamLightInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('camlight'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            cameras=self.model.ncam,lights=self.model.nlight,modes='FIXED only; structural edits require rebind/recapture',
            numerical_workspace_bytes=0,additional_topology_bytes=0,values='borrow-current-input-no-cache')

    def enqueue(self,phase=None):
        if phase is None:
            self.enqueue(0);self.enqueue(1);return
        if phase not in (0,1):raise ValueError('CamLight phase must be 0 or 1')
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('CamLight launch failed: CUDA error '+str(error))
