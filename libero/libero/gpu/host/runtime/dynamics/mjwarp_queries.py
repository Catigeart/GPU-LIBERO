"""Borrowed rigid kinematic queries with independent load-only entry ABIs."""
import ctypes as ct
import hashlib
from .mjwarp_motion import bind_array,load_leaf,reject_output_overlap,source_id


class QueryGeometry(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','dofs','ancestor_stride')]+[(k,ct.c_void_p) for k in ('root','ancestor','com','cdof')]
class JacobianInputs(ct.Structure):
    _fields_=[('geometry',QueryGeometry)]+[(k,ct.c_void_p) for k in ('point','body','position','rotation')]
class ForceSensorInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','bodies','sites','sensors','channels','disabled')]+[(k,ct.c_void_p) for k in ('root','site_body','type','object','address','datatype','cutoff','site_position','site_rotation','com','wrench','output')]
class GravityInputs(ct.Structure):
    _fields_=[('geometry',QueryGeometry)]+[(k,ct.c_int) for k in ('gravity_rows','mass_rows','coefficient_rows')]+[(k,ct.c_void_p) for k in ('gravity','mass','coefficient','position','output')]


def validate_geometry(root,ancestor,bodies,dofs):
    import numpy as np
    if bodies<1 or dofs<0 or root.shape!=(bodies,) or ancestor.ndim!=2 or ancestor.shape[0]!=bodies or ancestor.shape[1]<dofs or np.any(root<0) or np.any(root>=bodies) or np.any((ancestor!=0)&(ancestor!=1)):raise ValueError('Query geometry topology/layout differs')


def bind_geometry(model,data,arrays,stream):
    from libero.gpu.host.runtime.cuda_api import array_api
    wp = array_api(stream)
    n,nb,nv=data.nworld,model.nbody,model.nv
    stride=model.body_isdofancestor.shape[1]
    if n<1 or nb<1 or nv<0 or max(n*nb*3,n*nv*6,nb*stride)>2**31-256:raise ValueError('Query dimensions exceed index storage')
    for key,value,dtype,shape in (('root',model.body_rootid,wp.int32,(nb,)),('ancestor',model.body_isdofancestor,wp.int32,(nb,stride)),('com',data.subtree_com,wp.vec3,(n,nb)),('cdof',data.cdof,wp.spatial_vector,(n,nv))):bind_array(arrays,key,value,dtype,shape,stream)
    validate_geometry(arrays['root'].numpy(),arrays['ancestor'].numpy(),nb,nv)
    return QueryGeometry(worlds=n,bodies=nb,dofs=nv,ancestor_stride=stride,**{k:arrays[k].ptr for k in ('root','ancestor','com','cdof')})


def validate_force_sensors(root,site_body,types,objects,addresses,datatypes,dimensions,object_types,stages,bodies,channels):
    import numpy as np
    ns=len(types)
    if bodies<1 or channels<0 or root.shape!=(bodies,) or np.any(root<0) or np.any(root>=bodies) or np.any(site_body<0) or np.any(site_body>=bodies):raise ValueError('Force sensor root/site body differs')
    if any(v.shape!=(ns,) for v in (objects,addresses,datatypes,dimensions,object_types,stages)) or np.any((types!=4)&(types!=5)) or np.any(objects<0) or np.any(objects>=len(site_body)) or np.any(dimensions!=3) or np.any(object_types!=6) or not np.array_equal(np.sort(stages),np.arange(ns)) or np.any(datatypes<0) or np.any(datatypes>3):raise ValueError('Force sensor declared type/stage/domain differs')
    if np.any(addresses<0) or np.any(addresses.astype(np.int64)+3>channels) or np.any(np.diff(np.sort(addresses))<3):raise ValueError('Force sensor output spans overlap/exceed storage')


class _Query:
    def _bind_entry(self,kind,args_type,library,stream,arguments,outputs):
        self.stream=stream;self.path,self.library=load_leaf('queries',args_type,library,stream,entry=kind);self.arguments=arguments;self.kind=kind
        reject_output_overlap(self.arrays,outputs)
        self.launch=getattr(self.library,'flash_mjwarp_queries_'+kind+'_launch');self.launch.argtypes=[ct.POINTER(args_type),ct.c_void_p];self.launch.restype=ct.c_int
    @property
    def identity(self):return dict(source_id=source_id('queries'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),entry=self.kind,numerical_workspace_bytes=0,shared_bytes=0,values='borrow-current-input-no-cache')
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Query '+self.kind+' launch failed: CUDA error '+str(error))


class Jacobian(_Query):
    def __init__(self,model,data,point,body,jacp,jacr,library,stream):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        self.arrays=a={};geometry=bind_geometry(model,data,a,stream);n,nv=data.nworld,model.nv
        for key,value,dtype in (('point',point,wp.vec3),('body',body,wp.int32)):bind_array(a,key,value,dtype,(n,),stream)
        ids=body.numpy()
        if np.any(ids<0) or np.any(ids>=model.nbody):raise ValueError('Query body id exceeds storage')
        outputs=[]
        for key,value in (('position',jacp),('rotation',jacr)):
            if value is not None:bind_array(a,key,value,wp.float32,(n,3,nv),stream);outputs.append(key)
        arguments=JacobianInputs(geometry=geometry,point=point.ptr,body=body.ptr,position=jacp.ptr if jacp is not None else None,rotation=jacr.ptr if jacr is not None else None)
        self._bind_entry('jacobian',JacobianInputs,library,stream,arguments,tuple(outputs))


class ForceSensors(_Query):
    def __init__(self,model,data,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if tuple(map(int,(types.SensorType.FORCE,types.SensorType.TORQUE,types.ObjType.SITE,types.DataType.REAL)))!=(4,5,6,0):raise ValueError('Pinned force sensor constants differ')
        n,nb,ns,nt,nd=data.nworld,model.nbody,model.nsite,model.nsensor,model.nsensordata
        if n<1 or nb<1 or min(ns,nt,nd)<0 or max(n*nb*6,n*ns*9,n*nd,n*nt)>2**31-256:raise ValueError('Force sensor dimensions exceed index storage')
        self.arrays=a={}
        for key,field,width in (('root','body_rootid',nb),('site_body','site_bodyid',ns),('type','sensor_type',nt),('object','sensor_objid',nt),('address','sensor_adr',nt),('datatype','sensor_datatype',nt),('dimension','sensor_dim',nt),('object_type','sensor_objtype',nt),('stage','sensor_stage_acc_adr',nt)):bind_array(a,key,getattr(model,field),wp.int32,(width,),stream)
        validate_force_sensors(*(a[k].numpy() for k in ('root','site_body','type','object','address','datatype','dimension','object_type','stage')),nb,nd)
        bind_array(a,'cutoff',model.sensor_cutoff,wp.float32,(nt,),stream)
        for key,value,dtype,shape in (('site_position',data.site_xpos,wp.vec3,(n,ns)),('site_rotation',data.site_xmat,wp.mat33,(n,ns)),('com',data.subtree_com,wp.vec3,(n,nb)),('wrench',data.cfrc_int,wp.spatial_vector,(n,nb)),('output',data.sensordata,wp.float32,(n,nd))):bind_array(a,key,value,dtype,shape,stream)
        arguments=ForceSensorInputs(worlds=n,bodies=nb,sites=ns,sensors=nt,channels=nd,disabled=int(bool(model.opt.disableflags & types.DisableBit.SENSOR)),**{k:a[k].ptr for k,_ in ForceSensorInputs._fields_[6:]})
        self._bind_entry('sensor',ForceSensorInputs,library,stream,arguments,('output',))


class GravityProjection(_Query):
    def __init__(self,model,data,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        self.arrays=a={};geometry=bind_geometry(model,data,a,stream);n,nb,nv=data.nworld,model.nbody,model.nv;rows={}
        if n*(nb-1)*nv>2**31-256:raise ValueError('Gravity thread span exceeds index storage')
        for key,value,dtype,width in (('gravity',model.opt.gravity,wp.vec3,1),('mass',model.body_mass,wp.float32,nb),('coefficient',model.body_gravcomp,wp.float32,nb)):
            shape=value.shape
            if value.ndim!=(1 if key=='gravity' else 2) or shape[0]<1 or shape[0]*width*3>2**31-256:raise ValueError('Gravity parameter broadcast differs')
            bind_array(a,key,value,dtype,(shape[0],) if key=='gravity' else (shape[0],nb),stream);rows[{'gravity':'gravity_rows','mass':'mass_rows','coefficient':'coefficient_rows'}[key]]=shape[0]
        bind_array(a,'position',data.xipos,wp.vec3,(n,nb),stream);bind_array(a,'output',data.qfrc_gravcomp,wp.float32,(n,nv),stream)
        arguments=GravityInputs(geometry=geometry,**rows,**{k:a[k].ptr for k,_ in GravityInputs._fields_[4:]})
        self._bind_entry('gravity',GravityInputs,library,stream,arguments,('output',))
