"""Native-count elliptic post-RNE and two force/torque site sensors.

Preparation creates work arrays only. Caller then binds all true D/C/S/N input
owners through RuntimePostAdapter and calls freeze_bindings exactly once before
any launch. Durable public values remain separate in that publication adapter.
"""
import libero.gpu.precision as scalar
from types import SimpleNamespace
import numpy as np
import warp as wp
from .specialization import native_fields
from .runtime_adapter import _token
from .generic_atomic_postfix_r01 import post_count, force_sensor_count
from libero.gpu.reference.postprocess.rne_postconstraint.kernel import _sensor_finish


class GenericPostWorkspace:
    FIELDS=('body_parentid','body_dofadr','body_dofnum','body_rootid','geom_bodyid',
        'site_bodyid','sensor_type','sensor_objtype','sensor_objid','sensor_adr',
        'sensor_dim','sensor_needstage','sensor_cutoff','sensor_noise')
    def __init__(self,arrays,metadata,spec,capacity,contact_capacity,efc_capacity,*,
                 registry,candidate_id,source_records,metadata_record,mode='production',device='cuda:0'):
        s,self.admission=native_fields(arrays,metadata,spec,self.FIELDS,registry=registry,
            candidate_id=candidate_id,source_records=source_records,metadata_record=metadata_record,mode=mode)
        self.static=s;self.spec,self.registry,self.candidate_id,self.mode=spec,registry,candidate_id,mode
        d=metadata['dimensions'];o=metadata['options']
        self.nbody,self.nv,self.ngeom,self.nsite=(int(d[k]) for k in ('nbody','nv','ngeom','nsite'))
        if (d['nsensor'],d['nsensordata'],d['neq'],d['nplugin'],o['cone'],o['enableflags'],o['disableflags'])!=(2,6,0,0,1,0,0):
            raise ValueError('only elliptic/noequality two-force/torque sensor family implemented')
        if (not np.array_equal(s['sensor_type'],[4,5]) or not np.array_equal(s['sensor_adr'],[0,3]) or
            np.any(s['sensor_dim']!=3) or np.any(s['sensor_needstage']!=3) or
            np.any(s['sensor_objtype']!=6) or np.any(s['sensor_cutoff']!=0) or np.any(s['sensor_noise']!=0)):
            raise ValueError('actual force/torque sensor source family unsupported')
        ids=s['sensor_objid']
        if np.any(ids<0) or np.any(ids>=self.nsite):raise ValueError('sensor actual site selector')
        if any(type(v)is not int or v<1 for v in (capacity,contact_capacity,efc_capacity)):
            raise ValueError('positive actual capacities')
        self.capacity,self.contact_capacity,self.efc_capacity=capacity,contact_capacity,efc_capacity
        self.device=wp.get_device(device);n=capacity;b=self.nbody;v=self.nv;c=contact_capacity;e=efc_capacity
        for name,key in (('parent','body_parentid'),('dofadr','body_dofadr'),('dofnum','body_dofnum'),
                         ('root','body_rootid'),('geom_body','geom_bodyid')):
            setattr(self,name,wp.array(s[key],dtype=wp.int32,device=self.device))
        self.gravity=wp.array(np.asarray(o['gravity'],np.float64),dtype=scalar.wp_real,device=self.device)
        schema={'active':((n,),wp.int32),'qvel':((n,v),scalar.wp_real),'qacc':((n,v),scalar.wp_real),
            'cinert':((n,10*b),scalar.wp_real),'cvel':((n,6*b),scalar.wp_real),'cdof':((n,6*v),scalar.wp_real),
            'cdof_dot':((n,6*v),scalar.wp_real),'xipos':((n,b,3),scalar.wp_real),'subtree_com':((n,b,3),scalar.wp_real),
            'xfrc_applied':((n,6*b),scalar.wp_real),'efc_force':((n,e),scalar.wp_real),
            'ncon':((n,),wp.int32),'nefc':((n,),wp.int32),'ne':((n,),wp.int32),
            'contact_frame':((n,c*9),scalar.wp_real),'contact_pos':((n,c*3),scalar.wp_real),
            'site_xpos':((n,self.nsite,3),scalar.wp_real),'site_xmat':((n,self.nsite,9),scalar.wp_real)}
        for key in ('dim','efc_address','geom1','geom2'):schema['contact_'+key]=((n,c),wp.int32)
        for name,(shape,dtype) in schema.items():setattr(self,name,wp.zeros(shape,dtype=dtype,device=self.device))
        self.cacc=wp.zeros((n,6*b),dtype=scalar.wp_real,device=self.device)
        self.cfrc_ext=wp.zeros((n,6*b),dtype=scalar.wp_real,device=self.device)
        self.cfrc_int=wp.zeros((n,b,6),dtype=scalar.wp_real,device=self.device)
        self.contact_force=wp.zeros((n,c,6),dtype=scalar.wp_real,device=self.device)
        self.status=wp.zeros(n,dtype=wp.int32,device=self.device)
        self.sensor=SimpleNamespace(cfrc=self.cfrc_int,com=self.subtree_com,site_pos=self.site_xpos,
            site_mat=self.site_xmat,active=self.active,result=wp.zeros((n,6),dtype=scalar.wp_real,device=self.device),
            status=wp.zeros(n,dtype=wp.int32,device=self.device))
        for name,key in (('sensor_type','sensor_type'),('sensor_site','sensor_objid'),('sensor_adr','sensor_adr'),
                         ('site_body','site_bodyid'),('body_root','body_rootid')):
            setattr(self.sensor,name,wp.array(s[key],dtype=wp.int32,device=self.device))
        self._frozen=None
    def _tokens(self):
        return {k:_token(v) for k,v in vars(self).items() if isinstance(v,wp.array)}|{
            'sensor.'+k:_token(v) for k,v in vars(self.sensor).items() if isinstance(v,wp.array)}
    def freeze_bindings(self):
        if self._frozen is not None:raise ValueError('post preparation is already frozen')
        if self.sensor.cfrc is not self.cfrc_int:raise ValueError('sensor producer alias')
        self._frozen=self._tokens()
    def validate_bindings(self):
        if self._frozen is None or self._tokens()!=self._frozen:raise ValueError('actual post source/owner drift')
        if not self.registry.admit(self.candidate_id,self.spec,mode=self.mode).execution_allowed:
            raise ValueError('actual post model/source admission drift')
    def launch(self,stream=None):
        self.validate_bindings()
        names=('active','qvel','qacc','cinert','cvel','cdof','cdof_dot','xipos','subtree_com','site_xpos','site_xmat',
            'xfrc_applied','efc_force','ncon','nefc','ne','contact_dim','contact_efc_address','contact_geom1',
            'contact_geom2','contact_frame','contact_pos','parent','dofadr','dofnum','root','geom_body','gravity')
        args=[getattr(self,k) for k in names]+[self.contact_capacity,self.efc_capacity,self.cacc,self.cfrc_ext,
            self.cfrc_int,self.contact_force,self.status,self.nbody,self.nv,self.nsite,self.ngeom]
        wp.launch(post_count,dim=self.capacity,inputs=args,device=self.device,stream=stream)
        s=self.sensor
        wp.launch(force_sensor_count,dim=self.capacity,inputs=[s.cfrc,s.com,s.site_pos,s.site_mat,s.active,
            s.sensor_type,s.sensor_site,s.sensor_adr,s.site_body,s.body_root,s.result,s.status,self.nbody,self.nsite],
            device=self.device,stream=stream)
        wp.launch(_sensor_finish,dim=self.capacity,inputs=[self.active,self.status,s.status,s.result],device=self.device,stream=stream)
