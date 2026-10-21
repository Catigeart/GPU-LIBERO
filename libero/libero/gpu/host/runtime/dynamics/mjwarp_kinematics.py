"""Borrowed rigid kinematics arrays; shared cold branch owner, no physical cache."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap,MotionTopology


def caller_unsupported_reason(model,data):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if model.opt.integrator!=types.IntegratorType.EULER:return 'kinematics caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:return 'active-body compaction'
    if min(data.nworld,model.nv,model.nq,model.njnt,model.nbranch)<1 or model.nbody<2:return 'empty rigid kinematics domain'
    return None


def validate_indices(nq,nm,qadr,kinds,jnum,jadr,mocap,root,weld,geom_body,site_body):
    import numpy as np
    nb,nj=len(jnum),len(kinds)
    if any(len(a)!=nb for a in (jadr,mocap,root,weld)) or len(qadr)!=nj:raise ValueError('Kinematics index dimensions differ')
    if np.any(kinds<0) or np.any(kinds>3):raise ValueError('Kinematics joint type differs')
    if np.any(mocap< -1) or np.any(mocap>=nm):raise ValueError('Kinematics mocap index exceeds storage')
    for values in (root,weld,geom_body,site_body):
        if np.any(values<0) or np.any(values>=nb):raise ValueError('Kinematics body index exceeds storage')
    cover=np.zeros(nq,int)
    for j,k in enumerate(kinds):
        at=int(qadr[j]);width=(7,4,1,1)[k]
        if at<0 or at+width>nq:raise ValueError('Kinematics qpos span exceeds storage')
        cover[at:at+width]+=1
    if not np.all(cover==1):raise ValueError('Kinematics qpos spans overlap or omit coordinates')
    for b,count in enumerate(jnum):
        first=int(jadr[b])
        if count and (first<0 or first+count>nj):raise ValueError('Kinematics body joint span exceeds storage')
        if count and np.any(kinds[first:first+count]==0) and (count!=1 or kinds[first]!=0):
            raise ValueError('Kinematics FREE joint must be sole body joint')


class KinematicsInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','joints','nq','branches','geoms','sites','mocaps',
        'qpos0_rows','body_pos_rows','body_quat_rows','jnt_pos_rows','jnt_axis_rows','ipos_rows','iquat_rows','gpos_rows','gquat_rows','spos_rows','squat_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('parent','mocap','jnum','jadr','kind','qadr','branch_body','branch_start','owner','root','weld','geom_body','site_body',
        'qpos0','body_pos','body_quat','jnt_pos','jnt_axis','ipos','iquat','gpos','gquat','spos','squat','qpos','mocap_pos','mocap_quat',
        'xpos','xquat','anchor','axis','xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat')]


class AttachedTopology(ct.Structure):
    _fields_=[(k,ct.c_void_p) for k in ('geom_start','geom_id','site_start','site_id')]


class FrameAttachments(ct.Structure):
    _fields_=[(k,ct.c_void_p) for k in ('start','job')]


class KinematicFrames:
    def __init__(self,model,data,library,stream,*,topology=None):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        reason=caller_unsupported_reason(model,data)
        if reason:raise ValueError('Unsupported CUDA kinematics: '+reason)
        n,nb,nj,nq,ng,ns,nm=data.nworld,model.nbody,model.njnt,model.nq,model.ngeom,model.nsite,model.nmocap
        if min(ng,ns,nm)<0 or max(n*nb*9,n*nj*3,n*nq,n*ng*9,n*ns*9,n*nm*4,n*model.nbranch)>2**31-256:
            raise ValueError('Kinematics dimensions exceed index storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('kinematics',KinematicsInputs,library,stream)
        self.owns_topology=topology is None
        self.topology=MotionTopology(model,data,stream) if topology is None else topology
        if self.topology.model is not model or self.topology.data is not data or self.topology.stream is not stream:
            raise ValueError('Kinematics topology owner differs')
        a=self.arrays={};t=self.topology.arrays
        for key,source in (('parent','body_parent'),('jnum','body_jntnum'),('jadr','body_jntadr'),('kind','jnt_type'),
                ('branch_body','branch_bodies'),('branch_start','branch_start'),('owner','body_owner')):a[key]=t[source]
        values={}
        for key,field,width in (('mocap','body_mocapid',nb),('qadr','jnt_qposadr',nj),('root','body_rootid',nb),
                ('weld','body_weldid',nb),('geom_body','geom_bodyid',ng),('site_body','site_bodyid',ns)):
            values[key]=bind_array(a,key,getattr(model,field),wp.int32,(width,),stream).numpy()
        for key in ('kind','jnum','jadr'):values[key]=a[key].numpy()
        validate_indices(nq,nm,*(values[k] for k in ('qadr','kind','jnum','jadr','mocap','root','weld','geom_body','site_body')))
        if np.any((values['mocap']>=0)&((a['parent'].numpy()!=0)|(values['jnum']!=0))):raise ValueError('Kinematics mocap body must be fixed under world')
        rows={}
        for key,field,width,dtype,components in (('qpos0','qpos0',nq,wp.float32,1),('body_pos','body_pos',nb,wp.vec3,3),
                ('body_quat','body_quat',nb,wp.quat,4),('jnt_pos','jnt_pos',nj,wp.vec3,3),('jnt_axis','jnt_axis',nj,wp.vec3,3),
                ('ipos','body_ipos',nb,wp.vec3,3),('iquat','body_iquat',nb,wp.quat,4),('gpos','geom_pos',ng,wp.vec3,3),
                ('gquat','geom_quat',ng,wp.quat,4),('spos','site_pos',ns,wp.vec3,3),('squat','site_quat',ns,wp.quat,4)):
            value=getattr(model,field)
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:raise ValueError('Kinematics broadcast storage differs: '+key)
            bind_array(a,key,value,dtype,(value.shape[0],width),stream);rows[key+'_rows']=value.shape[0]
        for key,field,width,dtype in (('qpos','qpos',nq,wp.float32),('mocap_pos','mocap_pos',nm,wp.vec3),('mocap_quat','mocap_quat',nm,wp.quat),
                ('xpos','xpos',nb,wp.vec3),('xquat','xquat',nb,wp.quat),('anchor','xanchor',nj,wp.vec3),('axis','xaxis',nj,wp.vec3),
                ('xmat','xmat',nb,wp.mat33),('xipos','xipos',nb,wp.vec3),('ximat','ximat',nb,wp.mat33),
                ('geom_xpos','geom_xpos',ng,wp.vec3),('geom_xmat','geom_xmat',ng,wp.mat33),('site_xpos','site_xpos',ns,wp.vec3),('site_xmat','site_xmat',ns,wp.mat33)):
            bind_array(a,key,getattr(data,field),dtype,(n,width),stream)
        reject_output_overlap(a,('xpos','xquat','anchor','axis','xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat'))
        self.arguments=KinematicsInputs(worlds=n,bodies=nb,joints=nj,nq=nq,branches=model.nbranch,geoms=ng,sites=ns,mocaps=nm,**rows,
            **{k:a[k].ptr for k,_ in KinematicsInputs._fields_[19:]})
        self.launch=self.library.flash_mjwarp_kinematics_launch
        self.launch.argtypes=[ct.POINTER(KinematicsInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('kinematics'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            numerical_workspace_bytes=0,additional_topology_bytes=self.topology.identity['topology_bytes'] if self.owns_topology else 0,
            attachment_topology_bytes=sum(a.capacity for a in getattr(self,'attached_arrays',{}).values()),
            frame_topology_bytes=sum(a.capacity for a in getattr(self,'frame_arrays',{}).values()),
            frames=('canonical-branch' if self.combine_frames else 'separate-world-tiles') if hasattr(self,'frames') else 'unbound',
            topology='owned' if self.owns_topology else 'shared',owner_policy='first-full-leaf-path',prefix='private registers',values='borrow-current-input-no-cache')

    def enqueue(self,phase):
        error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
        if error:raise RuntimeError('Kinematics launch failed: CUDA error '+str(error))

    def prepare_fusion(self):
        """Group immutable attachment indices by their canonical branch owner."""
        if hasattr(self,'attached'):return
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp=array_api(self.stream)
        size=self.library.flash_mjwarp_kinematics_attached_bytes;size.restype=ct.c_size_t
        if size()!=ct.sizeof(AttachedTopology):raise ValueError('Attached topology ABI differs')
        values={};owner=self.topology.arrays['body_owner'].numpy()
        for name in ('geom','site'):
            ids=owner[self.arrays[name+'_body'].numpy()]
            ids=np.where(ids<0,0,ids)
            values[name+'_start']=np.r_[0,np.cumsum(np.bincount(ids,minlength=self.model.nbranch))].astype(np.int32)
            values[name+'_id']=np.argsort(ids,kind='stable').astype(np.int32)
        self.attached_arrays={k:wp.array(v,dtype=wp.int32,device=self.stream.device) for k,v in values.items()}
        self.attached=AttachedTopology(**{k:a.ptr for k,a in self.attached_arrays.items()})
        self.fused=self.library.flash_mjwarp_kinematics_fused
        self.fused.argtypes=[ct.POINTER(KinematicsInputs),ct.POINTER(AttachedTopology),ct.c_void_p];self.fused.restype=ct.c_int

    def enqueue_fused(self):
        error=self.fused(ct.byref(self.arguments),ct.byref(self.attached),self.stream.cuda_stream)
        if error:raise RuntimeError('Fused kinematics launch failed: CUDA error '+str(error))

    def prepare_frames(self,frames):
        """Bind admitted FIXED frames to their canonical body-publishing warp."""
        import numpy as np
        from .mjwarp_camlight import CamLightInputs
        from libero.gpu.host.runtime.cuda_api import array_api
        if frames.model is not self.model or frames.data is not self.data or frames.stream is not self.stream:
            raise ValueError('Kinematic camera/light owners differ')
        if hasattr(self,'frames'):
            if self.frames is not frames:raise ValueError('Kinematic camera/light binding differs')
            return
        self.prepare_fusion()
        for key in ('xpos','xquat'):
            if self.arrays[key].ptr!=frames.arrays[key].ptr:raise ValueError('Kinematic frame producer differs: '+key)
        arrays={'kin_'+k:v for k,v in self.arrays.items()}
        arrays.update({'frame_'+k:v for k,v in frames.arrays.items() if k not in ('xpos','xquat')})
        reject_output_overlap(arrays,tuple('kin_'+k for k in ('xpos','xquat','anchor','axis','xmat','xipos','ximat',
            'geom_xpos','geom_xmat','site_xpos','site_xmat'))+tuple('frame_'+k for k in ('cam_xpos','cam_xmat','light_xpos','light_xdir')))
        for name,kind in (('frames',CamLightInputs),('frame_ids',FrameAttachments)):
            size=getattr(self.library,'flash_mjwarp_kinematics_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Kinematic frames ABI differs: '+name)
        owner=self.topology.arrays['body_owner'].numpy();wp=array_api(self.stream)
        body=np.concatenate([self.arrays[k+'_body'].numpy() for k in ('geom','site')]+
                            [frames.arrays[k+'_body'].numpy() for k in ('cam','light')])
        ids=owner[body];ids=np.where(ids<0,0,ids)
        count=np.bincount(ids,minlength=self.model.nbranch)
        frame_count=np.bincount(ids[self.model.ngeom+self.model.nsite:],minlength=self.model.nbranch)
        self.combine_frames=bool(frame_count.max(initial=0)<=32)
        values=dict(start=np.r_[0,np.cumsum(count)].astype(np.int32),job=np.argsort(ids,kind='stable').astype(np.int32))
        self.frame_arrays={k:wp.array(v,dtype=wp.int32,device=self.stream.device) for k,v in values.items()}
        self.frame_ids=FrameAttachments(**{k:a.ptr for k,a in self.frame_arrays.items()});self.frames=frames
        self.frames_launch=self.library.flash_mjwarp_kinematics_frames
        self.frames_launch.argtypes=[ct.POINTER(KinematicsInputs),ct.POINTER(AttachedTopology),
            ct.POINTER(CamLightInputs),ct.POINTER(FrameAttachments),ct.c_void_p];self.frames_launch.restype=ct.c_int

    def enqueue_frames(self):
        if not self.combine_frames:
            self.enqueue_fused();self.frames.enqueue_fused();return
        error=self.frames_launch(ct.byref(self.arguments),ct.byref(self.attached),ct.byref(self.frames.arguments),
                                 ct.byref(self.frame_ids),self.stream.cuda_stream)
        if error:raise RuntimeError('Kinematic camera/light launch failed: CUDA error '+str(error))
