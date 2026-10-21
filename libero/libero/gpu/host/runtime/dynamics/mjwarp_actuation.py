"""Load-only stateless actuation leaves; current sparse moments stay on device."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


def qualify(model,data):
    """Cold enum/index admission only; never cache numerical parameter values."""
    import numpy as np
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    reason=None;values={}
    if model.opt.integrator!=types.IntegratorType.EULER:reason='actuation caller requires Euler'
    elif model.opt.enableflags & types.EnableBit.SLEEP:reason='active-body compaction'
    elif model.na:reason='stateful actuator dynamics retained upstream'
    elif model.ntendon:reason='tendon force limits retained upstream'
    elif min(model.nactuator,model.nu,model.nv,model.njnt)<1:reason='empty actuation domain'
    elif data.actuator_moment.shape[1]<1:reason='empty actuator moment storage'
    if reason is None:
        if (int(types.GainType.FIXED),int(types.GainType.AFFINE),int(types.BiasType.NONE),
            int(types.BiasType.AFFINE),int(types.DynType.NONE))!=(0,1,0,1,0):
            raise ValueError('Actuation pinned enum representation differs')
        for field in ('actuator_dyntype','actuator_gaintype','actuator_biastype','actuator_trntype',
                'actuator_trnid','actuator_ctrladr','actuator_ctrlnum','actuator_actadr','jnt_type'):
            values[field]=getattr(model,field).numpy()
        if np.any(values['actuator_dyntype']!=0) or np.any(values['actuator_actadr']>=0):
            reason='stateful or custom actuator dynamics retained upstream'
        elif np.any(~np.isin(values['actuator_gaintype'],(0,1))) or np.any(~np.isin(values['actuator_biastype'],(0,1))):
            reason='non-fixed/affine gain or bias retained upstream'
        elif np.any(~np.isin(values['actuator_trntype'],(int(types.TrnType.JOINT),int(types.TrnType.JOINTINPARENT)))):
            reason='non-joint transmission retained upstream'
        else:
            ids=values['actuator_trnid'][:,0]
            if np.any(ids<0) or np.any(ids>=model.njnt):raise ValueError('Actuation transmission joint exceeds storage')
            if np.any(~np.isin(values['jnt_type'][ids],(int(types.JointType.SLIDE),int(types.JointType.HINGE)))):
                reason='rotational setpoint wrapping retained upstream'
            counts=values['actuator_ctrlnum'];adr=values['actuator_ctrladr']
            if np.any(~np.isin(counts,(0,1))):reason='multi-control actuator retained upstream'
            elif np.any((counts>0)&((adr<0)|(adr>=model.nu))):raise ValueError('Actuation control address exceeds storage')
    return dict(reason=reason,values=values,model=id(model),data=id(data))


class ActuationInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','controls','actuators','joints','moments',
        'gain_rows','bias_rows','ctrlrange_rows','forcerange_rows','jointrange_rows','disable_clampctrl','gravity_enabled')]
    _fields_ += [(k,ct.c_void_p) for k in ('gain_type','bias_type','ctrladr','ctrlnum','dof_joint','actgravcomp',
        'ctrllimited','forcelimited','jointlimited','gainprm','biasprm','ctrlrange','forcerange','jointrange',
        'moment_rownnz','moment_rowadr','moment_colind','moment','ctrl','length','velocity','gravcomp','force','qfrc')]


class ActuatorForces:
    def __init__(self,model,data,library,stream,*,admission=None,control=None):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        admission=qualify(model,data) if admission is None else admission
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Actuation admission owner differs')
        if admission['reason']:raise ValueError('Unsupported CUDA actuation: '+admission['reason'])
        n,nv,nu,na,nj,nm=data.nworld,model.nv,model.nu,model.nactuator,model.njnt,data.actuator_moment.shape[1]
        if min(n,nv,nu,na,nj,nm)<1 or max(n*nv,n*nu,n*na,n*nm)>2**31-256:
            raise ValueError('Actuation dimensions exceed index storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('actuation',ActuationInputs,library,stream)
        self.arrays=a={}
        for key,field,width,dtype in (('gain_type','actuator_gaintype',na,wp.int32),('bias_type','actuator_biastype',na,wp.int32),
                ('ctrladr','actuator_ctrladr',na,wp.int32),('ctrlnum','actuator_ctrlnum',na,wp.int32),
                ('dof_joint','dof_jntid',nv,wp.int32),('actgravcomp','jnt_actgravcomp',nj,wp.int32),
                ('ctrllimited','actuator_ctrllimited',nu,wp.bool),('forcelimited','actuator_forcelimited',na,wp.bool),
                ('jointlimited','jnt_actfrclimited',nj,wp.bool)):
            bind_array(a,key,getattr(model,field),dtype,(width,),stream)
        dof_joint=a['dof_joint'].numpy()
        if np.any(dof_joint<0) or np.any(dof_joint>=nj):raise ValueError('Actuation DOF joint exceeds storage')
        rows={}
        for key,field,width,dtype,rowkey,components in (('gainprm','actuator_gainprm',na,types.vec10,'gain_rows',10),
                ('biasprm','actuator_biasprm',na,types.vec10,'bias_rows',10),
                ('ctrlrange','actuator_ctrlrange',nu,wp.vec2,'ctrlrange_rows',2),
                ('forcerange','actuator_forcerange',na,wp.vec2,'forcerange_rows',2),
                ('jointrange','jnt_actfrcrange',nj,wp.vec2,'jointrange_rows',2)):
            value=getattr(model,field)
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:
                raise ValueError('Actuation broadcast storage differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],width),stream);rows[rowkey]=value.shape[0]
        for key,field,width,dtype in (('moment_rownnz','moment_rownnz',na,wp.int32),('moment_rowadr','moment_rowadr',na,wp.int32),
                ('moment_colind','moment_colind',nm,wp.int32),('moment','actuator_moment',nm,wp.float32),
                ('length','actuator_length',na,wp.float32),('velocity','actuator_velocity',na,wp.float32),
                ('gravcomp','qfrc_gravcomp',nv,wp.float32),('force','actuator_force',na,wp.float32),('qfrc','qfrc_actuator',nv,wp.float32)):
            bind_array(a,key,getattr(data,field),dtype,(n,width),stream)
        bind_array(a,'ctrl',data.ctrl if control is None else control,wp.float32,(n,nu),stream)
        reject_output_overlap(a,('force','qfrc'))
        self.flags=dict(disable_clampctrl=int(bool(model.opt.disableflags & types.DisableBit.CLAMPCTRL)),
            gravity_enabled=int(not (model.opt.disableflags & types.DisableBit.GRAVITY)))
        self.arguments=ActuationInputs(worlds=n,nv=nv,controls=nu,actuators=na,joints=nj,moments=nm,**rows,**self.flags,
            **{k:a[k].ptr for k,_ in ActuationInputs._fields_[13:]})
        self.launch=self.library.flash_mjwarp_actuation_launch
        self.launch.argtypes=[ct.POINTER(ActuationInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('actuation'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            flags=self.flags,values='borrow-current-input-no-cache',numerical_workspace_bytes=0,
            domain='stateless fixed/affine scalar-joint actuators',moments='current upstream CSR; no numerical/index cache')

    def enqueue(self,phase):
        if phase==1:self.arrays['qfrc'].zero_()
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Actuation launch failed: CUDA error '+str(error))
