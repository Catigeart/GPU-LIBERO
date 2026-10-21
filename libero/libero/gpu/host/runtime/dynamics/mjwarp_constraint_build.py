"""Load-only dense constraint construction; all numerical arrays are borrowed."""
import ctypes as ct
import hashlib
from .mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap

PARAMETERS=(('timestep','opt.timestep','float32',1,1),('impratio_invsqrt','opt.impratio_invsqrt','float32',1,1),
    ('body_weight','body_invweight0','vec2','nbody',2),('dof_weight','dof_invweight0','float32','nv',1),
    ('dof_friction','dof_frictionloss','float32','nv',1),('dof_ref','dof_solref','vec2','nv',2),('dof_imp','dof_solimp','vec5','nv',5),
    ('joint_ref','jnt_solref','vec2','njnt',2),('joint_imp','jnt_solimp','vec5','njnt',5),
    ('joint_range','jnt_range','vec2','njnt',2),('joint_margin','jnt_margin','float32','njnt',1))
CONTACTS=(('contact_dim','dim','int32'),('contact_world','worldid','int32'),('contact_geom','geom','vec2i'),('contact_type','type','int32'),
    ('contact_dist','dist','float32'),('contact_margin','includemargin','float32'),('contact_pos','pos','vec3'),('contact_frame','frame','mat33'),
    ('contact_friction','friction','vec5'),('contact_ref','solref','vec2'),('contact_ref_friction','solreffriction','vec2'),
    ('contact_imp','solimp','vec5'),('contact_adhesion','adhesion','float32'))
ROWS=(('row_type','type','int32'),('row_id','id','int32'),('row_pos','pos','float32'),('row_margin','margin','float32'),
    ('row_D','D','float32'),('row_vel','vel','float32'),('row_aref','aref','float32'),('row_friction','frictionloss','float32'))


def qualify(model,data,*,implementation='upstream'):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('constraint_build must be upstream or cuda')
    reason=None
    if implementation=='cuda':
        if model.opt.integrator!=types.IntegratorType.EULER:reason='constraint construction caller requires Euler'
        elif model.opt.enableflags & types.EnableBit.SLEEP:reason='active-body compaction'
        elif model.is_sparse:reason='sparse constraint construction retained upstream'
        elif data.nworld<1 or model.nv<1 or model.njnt<1 or model.nbody<2:reason='empty constraint construction domain'
        elif model.neq:reason='equality row construction retained upstream'
        elif model.ntendon:reason='tendon row construction retained upstream'
        elif any(getattr(model,k) for k in ('nflex','nflexedge','nflexelem','nflexbend_interp','nflexintcell')):reason='flex row construction retained upstream'
        elif model.jnt_limited_ball_adr.size:reason='limited BALL row construction retained upstream'
        elif model.flg_surfacevel:reason='moving contact surface retained upstream'
        elif model.opt.cone not in (types.ConeType.PYRAMIDAL,types.ConeType.ELLIPTIC):reason='unsupported contact cone'
        elif model.nmaxcondim not in (1,3,4,6):reason='unsupported maximum contact dimension'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,model=id(model),data=id(data))


def parameter(model,path):
    for key in path.split('.'):model=getattr(model,key)
    return model


def validate_metadata(values,nq,nv,padded):
    import numpy as np
    from .mjwarp_external_wrench import validate_topology
    validate_topology(values['parent'],values['root'],values['geom_body'])
    joint=values['joint_type'];width_q=np.array([7,4,1,1]);width_v=np.array([6,3,1,1])
    if np.any(joint<0) or np.any(joint>3):raise ValueError('Constraint construction joint type differs')
    q,v=values['joint_qpos'],values['joint_dof']
    if np.any(q<0) or np.any(q+width_q[joint]>nq) or np.any(v<0) or np.any(v+width_v[joint]>nv):raise ValueError('Constraint construction joint addresses differ')
    limited=values['limited_joint']
    if np.any(limited<0) or np.any(limited>=len(joint)) or len(np.unique(limited))!=len(limited):raise ValueError('Constraint construction limited ids differ')
    if np.any((joint[limited]!=2)&(joint[limited]!=3)):raise ValueError('Constraint construction limited joint type differs')
    body=values['dof_body'];nb=len(values['parent'])
    if np.any(body<1) or np.any(body>=nb):raise ValueError('Constraint construction DOF body differs')
    expected=np.zeros((nb,padded),np.int32)
    for b in range(1,nb):
        chain=set();p=b
        while p:chain.add(p);p=int(values['parent'][p])
        expected[b,:nv]=np.isin(body,list(chain))
    if not np.array_equal(values['ancestor'],expected):raise ValueError('Constraint construction ancestor mask differs')


class Broadcast(ct.Structure):_fields_=[('data',ct.c_void_p),('rows',ct.c_int)]


class ConstraintBuildInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','nq','bodies','joints','geoms','contacts','rows','padded_rows','padded_nv','address_columns','limited',
        'cone','constraints_enabled','friction_enabled','limits_enabled','contacts_enabled','refsafe','adhesion_enabled')]
    _fields_ += [(k,Broadcast) for k,*_ in PARAMETERS]
    _fields_ += [(k,ct.c_void_p) for k in ('parent','root','geom_body','ancestor','joint_qpos','joint_dof','limited_joint',
        'contact_count','contact_dim','contact_world','contact_geom','contact_type','qpos','qvel','cdof','subtree_com',
        'contact_dist','contact_margin','contact_pos','contact_frame','contact_friction','contact_ref','contact_ref_friction','contact_imp','contact_adhesion',
        'ne','nf','nl','nefc','jtdaj_count','contact_address','row_type','row_id','J','Jqvel','row_pos','row_margin','row_D','row_vel','row_aref','row_friction')]


class ConstraintConstruction:
    def __init__(self,model,data,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if admission['model']!=id(model) or admission['data']!=id(data):raise ValueError('Constraint construction admission owner differs')
        if admission['requested']!='cuda' or admission['selected']!='cuda':raise ValueError('Constraint construction domain not admitted')
        if tuple(int(x) for x in (types.ConstraintType.FRICTION_DOF,types.ConstraintType.LIMIT_JOINT,types.ConstraintType.CONTACT_FRICTIONLESS,
                types.ConstraintType.CONTACT_PYRAMIDAL,types.ConstraintType.CONTACT_ELLIPTIC,types.ConeType.PYRAMIDAL,types.ConeType.ELLIPTIC,types.ContactType.CONSTRAINT))!=(1,3,5,6,7,0,1,1):raise ValueError('Pinned constraint enums differ')
        n,nv,nq,nb,nj,ng,nc,nr,pad=data.nworld,model.nv,model.nq,model.nbody,model.njnt,model.ngeom,data.naconmax,data.njmax,model.nv_pad
        if data.efc.J.ndim!=3 or data.efc.J.shape[1]<nr or data.efc.J.shape[2]!=pad:raise ValueError('Constraint construction dense matrix shape differs')
        rp=data.efc.J.shape[1];address=data.contact.efc_address
        if address.ndim!=2 or address.shape[1]<max(model.nmaxcondim,2*(model.nmaxcondim-1)):raise ValueError('Constraint construction contact address stride differs')
        columns=address.shape[1];limited=model.jnt_limited_slide_hinge_adr.size
        if min(nr,nc,ng)<0 or pad<nv or max(n*max(1,rp)*pad,n*nb*6,n*nv*6,nc*max(9,columns),n*nq,nj*5,ng,columns)>2**31-256:raise ValueError('Constraint construction index dimensions exceed storage')
        self.model,self.data,self.stream=model,data,stream
        self.path,self.library=load_leaf('constraint_build',ConstraintBuildInputs,library,stream)
        self.arrays=a={};static={};dtype=lambda name:types.vec5 if name=='vec5' else getattr(wp,name)
        fields=(('parent','body_parentid',(nb,)),('root','body_rootid',(nb,)),('geom_body','geom_bodyid',(ng,)),('ancestor','body_isdofancestor',(nb,pad)),
            ('joint_type','jnt_type',(nj,)),('joint_qpos','jnt_qposadr',(nj,)),('joint_dof','jnt_dofadr',(nj,)),('dof_body','dof_bodyid',(nv,)),
            ('limited_joint','jnt_limited_slide_hinge_adr',(limited,)))
        for key,field,shape in fields:static[key]=bind_array(a,key,getattr(model,field),wp.int32,shape,stream).numpy()
        validate_metadata(static,nq,nv,pad)
        params={}
        for key,field,kind,width,components in PARAMETERS:
            width=getattr(model,width) if isinstance(width,str) else width;value=parameter(model,field)
            shape=(value.shape[0],) if field.startswith('opt.') else (value.shape[0],width)
            if value.ndim!=len(shape) or value.shape[0]<1 or value.shape[0]*width*components>2**31-256:raise ValueError('Constraint construction broadcast differs: '+key)
            bind_array(a,key,value,dtype(kind),shape,stream);params[key]=Broadcast(value.ptr,value.shape[0])
        for key,field,kind,shape in (('qpos','qpos','float32',(n,nq)),('qvel','qvel','float32',(n,nv)),('cdof','cdof','spatial_vector',(n,nv)),
                ('subtree_com','subtree_com','vec3',(n,nb)),('contact_count','nacon','int32',(1,))):bind_array(a,key,getattr(data,field),dtype(kind),shape,stream)
        for key,field,kind in CONTACTS:bind_array(a,key,getattr(data.contact,field),dtype(kind),(nc,),stream)
        for key in ('ne','nf','nl','nefc'):bind_array(a,key,getattr(data,key),wp.int32,(n,),stream)
        bind_array(a,'jtdaj_count',data.efc.jtdaj_nblock,wp.int32,(n,),stream)
        bind_array(a,'contact_address',address,wp.int32,(nc,columns),stream)
        for key,field,kind in ROWS:bind_array(a,key,getattr(data.efc,field),dtype(kind),(n,nr),stream)
        bind_array(a,'J',data.efc.J,wp.float32,(n,rp,pad),stream)
        bind_array(a,'Jqvel',data.efc.Jqvel,wp.float32,(n,nr),stream)
        outputs=('ne','nf','nl','nefc','jtdaj_count','contact_address','J','Jqvel')+tuple(k for k,*_ in ROWS)
        reject_output_overlap(a,outputs)
        flags=model.opt.disableflags
        enabled=int(not bool(flags & types.DisableBit.CONSTRAINT))
        choices=dict(constraints_enabled=enabled,friction_enabled=int(not bool(flags & types.DisableBit.FRICTIONLOSS)),
            limits_enabled=int(not bool(flags & types.DisableBit.LIMIT)),contacts_enabled=int(not bool(flags & types.DisableBit.CONTACT)),
            refsafe=int(not bool(flags & types.DisableBit.REFSAFE)),adhesion_enabled=int(model.flg_adhesion))
        self.arguments=ConstraintBuildInputs(worlds=n,nv=nv,nq=nq,bodies=nb,joints=nj,geoms=ng,contacts=nc,rows=nr,padded_rows=rp,padded_nv=pad,
            address_columns=columns,limited=limited,cone=int(model.opt.cone),**choices,**params,
            **{k:a[k].ptr for k,_ in ConstraintBuildInputs._fields_[19+len(PARAMETERS):]})
        phases=[0]
        if enabled:
            if choices['friction_enabled']:phases.append(1)
            if choices['limits_enabled'] and limited:phases.append(2)
            if choices['contacts_enabled']:phases.extend((3,4,5))
        self.phases=tuple(phases)
        self.launch=self.library.flash_mjwarp_constraint_build_launch
        self.launch.argtypes=[ct.POINTER(ConstraintBuildInputs),ct.c_int,ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(source_id=source_id('constraint_build'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            numerical_workspace_bytes=0,topology_bytes=0,values='borrow-current-input-no-cache',phases=list(self.phases),
            allocation='pinned atomic per-world counts and contiguous contact row blocks',shared_bytes_per_jacobian_block=32)

    def enqueue(self):
        for phase in self.phases:
            error=self.launch(ct.byref(self.arguments),phase,self.stream.cuda_stream)
            if error:raise RuntimeError('Constraint construction launch failed: CUDA error '+str(error))
