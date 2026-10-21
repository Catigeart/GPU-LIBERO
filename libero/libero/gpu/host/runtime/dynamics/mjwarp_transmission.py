"""Load-only scalar transmission and generic CSR velocity with borrowed outputs."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def caller_unsupported_reason(model,data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator!=types.IntegratorType.EULER:return 'transmission caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if min(data.nworld,model.nv,model.nq,model.njnt,model.nactuator)<1:return 'empty transmission domain'
    if data.actuator_moment.shape[1]<1:return 'empty actuator moment storage'
    return None


def qualify(model,data,*,transmission='upstream',actuator_velocity='upstream'):
    """One cold static-domain read; no physical coefficient or CSR value cache."""
    import numpy as np
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    requested=dict(transmission=transmission,actuator_velocity=actuator_velocity)
    if any(v not in ('upstream','cuda') for v in requested.values()):raise ValueError('transmission/actuator_velocity must be upstream or cuda')
    base=caller_unsupported_reason(model,data) if 'cuda' in requested.values() else None
    reasons={k:base if v=='cuda' else None for k,v in requested.items()};values={}
    if transmission=='cuda' and base is None:
        if (int(types.TrnType.JOINT),int(types.TrnType.JOINTINPARENT),int(types.JointType.SLIDE),int(types.JointType.HINGE))!=(0,1,2,3):
            raise ValueError('Transmission pinned enums differ')
        for key,shape in (('actuator_trntype',(model.nactuator,)),('actuator_trnid',(model.nactuator,2)),
                ('jnt_type',(model.njnt,)),('jnt_qposadr',(model.njnt,)),('jnt_dofadr',(model.njnt,))):
            values[key]=getattr(model,key).numpy()
            if values[key].shape!=shape:raise ValueError('Transmission metadata shape differs: '+key)
        if np.any(~np.isin(values['actuator_trntype'],(0,1))):reasons['transmission']='non-joint transmission retained upstream'
        else:
            ids=values['actuator_trnid'][:,0]
            if np.any(ids<0) or np.any(ids>=model.njnt):raise ValueError('Transmission joint exceeds storage')
            if np.any(~np.isin(values['jnt_type'][ids],(2,3))):reasons['transmission']='FREE/BALL transmission retained upstream'
            else:
                q=values['jnt_qposadr'][ids];d=values['jnt_dofadr'][ids]
                if np.any(q<0) or np.any(q>=model.nq) or np.any(d<0) or np.any(d>=model.nv):
                    raise ValueError('Transmission joint span exceeds storage')
                if data.actuator_moment.shape[1]<model.nactuator:raise ValueError('Transmission moment capacity below scalar row count')
    selected={k:'upstream' if reasons[k] else v for k,v in requested.items()}
    return dict(requested=requested,selected=selected,reasons=reasons,values=values,model=id(model),data=id(data))


class TransmissionInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','nq','actuators','moments','gear_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('joint_qposadr','joint_dofadr','actuator_joint','gear','qpos','qvel',
        'total','rownnz','rowadr','column','moment','length','velocity')]


class TransmissionNumerics:
    def __init__(self,model,data,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Transmission admission owner differs')
        selected=admission['selected']
        if 'cuda' not in selected.values():raise ValueError('No qualified CUDA transmission leaf')
        n,nv,nq,na,nj,nm=data.nworld,model.nv,model.nq,model.nactuator,model.njnt,data.actuator_moment.shape[1]
        if min(n,nv,nq,na,nj,nm)<1 or max(n*nv,n*nq,n*na,n*nm)>2**31-256:
            raise ValueError('Transmission dimensions exceed index storage')
        self.model,self.data,self.stream,self.selected=model,data,stream,selected.copy()
        self.path,self.library=load_leaf('transmission',TransmissionInputs,library,stream)
        self.arrays=a={};gear_rows=1
        for key,field,width,dtype in (('rownnz','moment_rownnz',na,wp.int32),('rowadr','moment_rowadr',na,wp.int32),
                ('column','moment_colind',nm,wp.int32),('moment','actuator_moment',nm,wp.float32),
                ('qvel','qvel',nv,wp.float32),('length','actuator_length',na,wp.float32),('velocity','actuator_velocity',na,wp.float32)):
            bind_array(a,key,getattr(data,field),dtype,(n,width),stream)
        outputs=[]
        if selected['transmission']=='cuda':
            for key,field,width,dtype in (('joint_qposadr','jnt_qposadr',nj,wp.int32),('joint_dofadr','jnt_dofadr',nj,wp.int32),
                    ('actuator_joint','actuator_trnid',na,wp.vec2i)):
                bind_array(a,key,getattr(model,field),dtype,(width,),stream)
            gear=model.actuator_gear
            if gear.ndim!=2 or gear.shape[0]<1 or gear.shape[0]*na*6>2**31-256:raise ValueError('Transmission gear broadcast storage differs')
            gear_rows=gear.shape[0];bind_array(a,'gear',gear,wp.spatial_vector,(gear_rows,na),stream)
            bind_array(a,'qpos',data.qpos,wp.float32,(n,nq),stream)
            # Pinned per-call zero counter, retained once for Graph lifetime.
            a['total']=wp.empty(n,dtype=int,device=stream.device)
            outputs+=['rownnz','rowadr','column','moment','length','total']
        if selected['actuator_velocity']=='cuda':outputs+=['velocity']
        reject_output_overlap(a,outputs)
        self.arguments=TransmissionInputs(worlds=n,nv=nv,nq=nq,actuators=na,moments=nm,gear_rows=gear_rows,
            **{k:a[k].ptr if k in a else 0 for k,_ in TransmissionInputs._fields_[6:]})
        self.launch=self.library.flash_mjwarp_transmission_launch
        self.launch.argtypes=[ct.POINTER(TransmissionInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('transmission'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            row_allocation='pinned per-world atomic append',counter_workspace_bytes=self.arrays['total'].capacity if 'total' in self.arrays else 0,
            numerical_workspace_bytes=0,values='borrow-current-input-no-cache',velocity='generic current CSR; ordered row dot')

    def enqueue(self,phase):
        if phase not in (0,1) or self.selected[('transmission','actuator_velocity')[phase]]!='cuda':raise ValueError('CUDA transmission phase not selected')
        if phase==0:self.arrays['total'].zero_()
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Transmission launch failed: CUDA error '+str(error))
