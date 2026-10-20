"""Frozen two-model workspace and caller-owned state for the N16 prefix."""
from __future__ import annotations
import libero.gpu.precision as scalar
import hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from .kernel import Model, Data, Diagnostic, _compute, _commit

STATIC={'qpos0':('float64',lambda n:(n['nq'],)),
 'body_parentid':('int32',lambda n:(n['nbody'],)), 'body_jntnum':('int32',lambda n:(n['nbody'],)),
 'body_jntadr':('int32',lambda n:(n['nbody'],)), 'body_mocapid':('int32',lambda n:(n['nbody'],)),
 'body_pos':('float64',lambda n:(n['nbody'],3)), 'body_quat':('float64',lambda n:(n['nbody'],4)),
 'jnt_type':('int32',lambda n:(n['njnt'],)), 'jnt_qposadr':('int32',lambda n:(n['njnt'],)),
 'jnt_axis':('float64',lambda n:(n['njnt'],3)), 'jnt_pos':('float64',lambda n:(n['njnt'],3))}
FIXED={
 'real':{'file_sha256':'3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42',
         'field_sha256':'cb64df28b2a92bf5e2771cae5f8077118b6a8ddc687c0b7bf699f326abbaef10',
         'spec_sha256':'21c54ef50cf43e6c7e986e5b13163bd6b46fc7038f5eaf54771f625083329a9c',
         'read_set_sha256':'875b705684cc92976c92b3ab49810fd7048e6ecbb70d78ca2f384248966cb483'},
 'synthetic':{'file_sha256':'8025007f27143c14a048993c6faac2f1224b5190ab548848c8627f54cc3ae632',
         'field_sha256':'56597d0674c44ede081da4eb8326a7e1e1c02e4af3e18313f0642e2d2f24c73b',
         'spec_sha256':'55d58163ad3e6da65a0dc0bfbc7e9b351ea43c49b6c0f3cc2b290ef5117866ad',
         'read_set_sha256':'aaacbfb75332056e7b82e3016e19b6727f76433fd6f768ee97eadad4320200c2'}}

def _read_set_hash(arrays,dims):
    h=hashlib.sha256()
    values={**arrays,**{k:np.array(v,np.int32) for k,v in dims.items()}}
    for name in sorted(values):
        a=np.asarray(values[name]);a=a.astype('<i8',copy=False) if a.dtype.kind in 'iu' else a.astype('<f8',copy=False)
        a=np.ascontiguousarray(a)
        h.update(json.dumps([name,a.shape,a.dtype.str],separators=(',',':')).encode());h.update(b'\0');h.update(a.tobytes())
    return h.hexdigest()

class FrozenModel:
    def __init__(self, spec, manifest, kind):
        spec=Path(spec); digest=hashlib.sha256(spec.read_bytes()).hexdigest()
        if kind not in FIXED:raise ValueError('unsupported model kind')
        item=manifest[kind]
        for field in ('file_sha256','field_sha256','spec_sha256'):
            if item[field]!=FIXED[kind][field]:raise ValueError('frozen model manifest '+field)
        if digest!=item['spec_sha256']:raise ValueError('model spec SHA mismatch')
        with np.load(spec,allow_pickle=False) as z:
            self.dims={k:int(z[k]) for k in ('nq','nbody','njnt','nmocap')}
            if self.dims!=item['dims']:raise ValueError('model dimension mismatch')
            self.arrays={k:z[k].copy() for k in STATIC}
        self.kind=kind;self.field_sha256=FIXED[kind]['field_sha256'];self.spec_sha256=digest
        self.validate()
    def validate(self):
        n=self.dims;a=self.arrays
        if self.kind not in ('real','synthetic') or n not in ({'nq':48,'nbody':38,'njnt':18,'nmocap':0},{'nq':13,'nbody':6,'njnt':4,'nmocap':1}):raise ValueError('unsupported frozen topology')
        for k,(dt,shape) in STATIC.items():
            if a[k].dtype!=np.dtype(dt) or a[k].shape!=shape(n) or (dt=='float64' and not np.isfinite(a[k]).all()):raise ValueError('model schema '+k)
        if _read_set_hash(a,n)!=FIXED[self.kind]['read_set_sha256']:raise ValueError('frozen read-set SHA mismatch')
        if int(a['body_parentid'][0])!=0 or np.any(a['body_jntnum']<0) or np.any(a['body_mocapid']<-1) or np.any(a['body_mocapid']>=n['nmocap']):raise ValueError('model topology')
        if np.any(~np.isin(a['jnt_type'],[0,1,2,3])):raise ValueError('joint type')
        for b in range(1,n['nbody']):
            if not 0<=a['body_parentid'][b]<b:raise ValueError('parent ordering')
            jnum=int(a['body_jntnum'][b]); jadr=int(a['body_jntadr'][b])
            if jnum and not 0<=jadr<jadr+jnum<=n['njnt']:raise ValueError('joint range')
        for j,t in enumerate(a['jnt_type']):
            q=int(a['jnt_qposadr'][j]);size={0:7,1:4,2:1,3:1}[int(t)]
            if not 0<=q<q+size<=n['nq']:raise ValueError('qpos range')

class BodyKinematicsWorkspace:
    """Static prepare uploads once; launch only invokes two resident kernels.

    active=0 or invalid active leaves the caller's qpos/mocap_quat bits unchanged.
    Geometry and diagnostic outputs are cleared for inactive/error worlds.
    status: 0 success/inactive, 1 invalid active, 2 trig domain, 3 type,
    4 nonfinite input, 5 internal nonfinite arithmetic.
    """
    def __init__(self, model:FrozenModel, capacity:int, device='cuda:0'):
        if not isinstance(model,FrozenModel):raise TypeError('requires verified FrozenModel')
        if not isinstance(capacity,int) or capacity<=0:raise ValueError('capacity')
        model.validate();self.model=model;self.capacity=capacity;self.device=device
        n=model.dims;nb=n['nbody'];nj=n['njnt'];nm=n['nmocap'];nq=n['nq']
        m=Model()
        for target,source,dtype in (('qpos0','qpos0',scalar.wp_real),('parent','body_parentid',wp.int32),('jnum','body_jntnum',wp.int32),('jadr','body_jntadr',wp.int32),('mocapid','body_mocapid',wp.int32),('bpos','body_pos',scalar.wp_real),('bquat','body_quat',scalar.wp_real),('jtype','jnt_type',wp.int32),('qadr','jnt_qposadr',wp.int32),('jaxis','jnt_axis',scalar.wp_real),('jpos','jnt_pos',scalar.wp_real)):
            setattr(m,target,wp.array(np.ascontiguousarray(model.arrays[source].reshape(-1)),dtype=dtype,device=device))
        m.nq=nq;m.nb=nb;m.nj=nj;m.nm=nm;self.static=m
        d=Data();g=Diagnostic()
        def f(shape):return wp.zeros(shape,scalar.wp_real,device=device)
        def i(shape):return wp.zeros(shape,wp.int32,device=device)
        d.active=i(capacity);d.qpos=f((capacity,nq));d.mocap_pos=f((capacity,max(1,nm*3)));d.mocap_quat=f((capacity,max(1,nm*4)))
        d.qtmp=f((capacity,nq));d.mqtmp=f((capacity,max(1,nm*4)))
        d.xpos=f((capacity,nb*3));d.xquat=f((capacity,nb*4));d.xmat=f((capacity,nb*9));d.xanchor=f((capacity,nj*3));d.xaxis=f((capacity,nj*3))
        d.world_xipos=f((capacity,3));d.world_ximat=f((capacity,9));d.status=i(capacity)
        for name,size in (('norm_before',nj),('mocap_before',nm),('pre_pos',nb*3),('pre_quat',nb*4),('joint_pos',nj*3),('joint_quat',nj*4),('body_norm_before',nb),('trig_arg',nj),('trig_sin',nj),('trig_cos',nj)):
            setattr(g,name,f((capacity,max(1,size))))
        for name,size in (('body_route',nb),('joint_route',nj),('body_mat_identity',nb),('joint_rot_identity',nj),('hinge_zero',nj),('offcenter_rot_identity',nj),('norm_branch',nj),('mocap_branch',nm),('body_norm_branch',nb)):
            setattr(g,name,i((capacity,max(1,size))))
        self.data=d;self.diag=g
        for name in ('active','qpos','mocap_pos','mocap_quat','xpos','xquat','xmat','xanchor','xaxis','world_xipos','world_ximat','status'):
            setattr(self,name,getattr(d,name))
    def launch(self,stream=None):
        wp.launch(_compute,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)
        wp.launch(_commit,dim=self.capacity,inputs=[self.static,self.data,self.diag],device=self.device,stream=stream)
    def upload(self,active,qpos,mocap_pos,mocap_quat):
        """Convenience host prepare only; never called by launch or Graph replay."""
        n=self.model.dims;N=self.capacity
        shapes={'active':(N,),'qpos':(N,n['nq']),'mocap_pos':(N,n['nmocap'],3),'mocap_quat':(N,n['nmocap'],4)}
        for name,a,dt in (('active',active,np.int32),('qpos',qpos,np.float64),('mocap_pos',mocap_pos,np.float64),('mocap_quat',mocap_quat,np.float64)):
            if not isinstance(a,np.ndarray) or a.shape!=shapes[name] or a.dtype!=dt:raise ValueError('input schema '+name)
            payload=np.ascontiguousarray(a.reshape((N,-1)) if name.startswith('mocap') else a)
            if name.startswith('mocap') and payload.shape[1]==0:continue
            wp.copy(getattr(self,name),wp.array(payload,dtype=wp.int32 if name=='active' else scalar.wp_real,device=self.device))
    def download(self):
        """Diagnostic download, explicitly outside the production launch interval."""
        n=self.model.dims;N=self.capacity;nb=n['nbody'];nj=n['njnt'];nm=n['nmocap']
        shapes={'xpos':(N,nb,3),'xquat':(N,nb,4),'xmat':(N,nb,9),'xanchor':(N,nj,3),'xaxis':(N,nj,3),'world_xipos':(N,3),'world_ximat':(N,9),'qpos_after':(N,n['nq']),'mocap_after':(N,nm,4),'qpos_after_stage':(N,n['nq']),'mocap_after_stage':(N,nm,4),'norm_before':(N,nj),'mocap_before':(N,nm),'pre_pos':(N,nb,3),'pre_quat':(N,nb,4),'joint_pos':(N,nj,3),'joint_quat':(N,nj,4),'body_norm_before':(N,nb)}
        result={}
        for name,shape in shapes.items():
            if name=='qpos_after':a=self.data.qpos
            elif name=='qpos_after_stage':a=self.data.qtmp
            elif name in ('mocap_after','mocap_after_stage'):a=self.data.mocap_quat if name=='mocap_after' else self.data.mqtmp
            else:a=getattr(self.data,name) if hasattr(self.data,name) else getattr(self.diag,name)
            result[name]=a.numpy()[:,:int(np.prod(shape[1:]))].reshape(shape).copy()
        for name,extent in (('body_route',nb),('joint_route',nj),('body_mat_identity',nb),('joint_rot_identity',nj),('hinge_zero',nj),('offcenter_rot_identity',nj),('norm_branch',nj),('mocap_branch',nm),('body_norm_branch',nb),('trig_arg',nj),('trig_sin',nj),('trig_cos',nj)):
            result[name]=getattr(self.diag,name).numpy()[:,:extent].copy()
        result['status']=self.status.numpy().copy()
        return result
