"""Native-count passive and split-actuation candidates.

Static preparation is explicit and source-pinned. Original fixed kernels and
their licences remain untouched. Hot methods allocate nothing and read no host
results. New models require separate independent numerical acceptance.
"""
import libero.gpu.precision as scalar
import math
import numpy as np
import warp as wp
from .specialization import native_fields
from .runtime_adapter import _token, _array
from .generic_atomic import joint_passive_count, fluid_count, actuation_count


class _Prepared:
    def _finish_prepare(self):
        self._tokens = {k: _token(v) for k, v in vars(self).items() if isinstance(v, wp.array)}
        self._scalars = (self.capacity, str(self.device), self.nq, self.nv, self.nu, self.nbody)
    def validate_bindings(self):
        if self._scalars != (self.capacity, str(self.device), self.nq, self.nv, self.nu, self.nbody):
            raise ValueError('generic prepared dimension/device drift')
        if not self.registry.admit(self.candidate_id, self.spec, mode=self.mode,
                required_scope='component-runtime' if self.mode == 'production' else None).execution_allowed:
            raise ValueError('generic stage model/source admission drift')
        for key, token in self._tokens.items():
            if _token(getattr(self, key)) != token:
                raise ValueError('generic stage actual owner drift: '+key)
    def _input(self, name, shape, dtype, inputs):
        value = inputs.get(name)
        if value is None:
            value = wp.zeros(shape, dtype=dtype, device=self.device)
        _array(value, shape, dtype, self.device)
        setattr(self, name, value)
    def _static(self, name, value, dtype=None):
        a = np.ascontiguousarray(value)
        if dtype is None:
            dtype = wp.int32 if a.dtype == np.int32 else scalar.wp_real
        setattr(self, name, wp.array(a, dtype=dtype, device=self.device))
    def _init(self, arrays, metadata, spec, capacity, fields, registry, candidate_id,
              source_records, metadata_record, mode, device):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive actual world capacity required')
        self.static, self.admission = native_fields(arrays, metadata, spec, fields,
            registry=registry, candidate_id=candidate_id, source_records=source_records,
            metadata_record=metadata_record, mode=mode)
        self.spec, self.registry, self.candidate_id, self.mode = spec, registry, candidate_id, mode
        self.capacity, self.device = capacity, wp.get_device(device)
        self.nq, self.nv, self.nu, self.nbody = (int(metadata['dimensions'][k]) for k in ('nq','nv','nu','nbody'))


class GenericPassiveWorkspace(_Prepared):
    FIELDS = ('jnt_type','jnt_qposadr','jnt_dofadr','jnt_stiffness','qpos_spring',
        'dof_damping','body_mass','body_inertia','body_parentid','body_rootid',
        'body_dofnum','body_dofadr','dof_parentid','body_gravcomp','geom_fluid')
    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', inputs=None):
        self._init(arrays,metadata,spec,capacity,self.FIELDS,registry,candidate_id,
                   source_records,metadata_record,mode,device)
        s = self.static; self.njnt = int(metadata['dimensions']['njnt'])
        for key in self.FIELDS:
            a = s[key]
            if a.dtype.kind == 'f' and not np.isfinite(a).all():
                raise ValueError('passive finite native field: '+key)
        if np.any(s['body_gravcomp'] != 0) or np.any(s['geom_fluid'] != 0):
            raise ValueError('gravity compensation/ellipsoid fluid family not implemented')
        if np.any(~np.isin(s['jnt_type'], [0,1,2,3])):
            raise ValueError('joint passive family')
        # Original separate RN box operations, generalized only in body count.
        box = np.zeros((self.nbody,3),np.float64)
        for body in range(1,self.nbody):
            m = float(s['body_mass'][body])
            if m < 1e-15: continue
            i0,i1,i2 = map(float,s['body_inertia'][body])
            for k,v in enumerate((i1+i2-i0,i0+i2-i1,i0+i1-i2)):
                box[body,k] = math.sqrt(max(1e-15,v)/m*6.0)
        ancestor = np.zeros((self.nbody,self.nv),np.int32)
        for body in range(1,self.nbody):
            owner = body
            while owner and s['body_dofnum'][owner] == 0:
                owner = int(s['body_parentid'][owner])
            if not owner: continue
            dof = int(s['body_dofadr'][owner]+s['body_dofnum'][owner]-1)
            seen=set()
            while dof >= 0:
                if dof >= self.nv or dof in seen: raise ValueError('fluid ancestor source domain')
                seen.add(dof); ancestor[body,dof]=1; dof=int(s['dof_parentid'][dof])
        for name,key in (('jtype','jnt_type'),('qadr','jnt_qposadr'),('dadr','jnt_dofadr'),
                         ('stiffness','jnt_stiffness'),('spring','qpos_spring'),('damping','dof_damping'),
                         ('rootid','body_rootid'),('mass','body_mass')):
            self._static(name,s[key])
        self._static('box',box); self._static('ancestor',ancestor)
        opt=metadata['options']
        self._static('wind',np.asarray(opt['wind'],np.float64))
        self._static('opt',np.asarray([opt['viscosity'],opt['density']],np.float64))
        self.passive_disabled = int(bool(int(opt['disableflags']) & 32))
        if self.passive_disabled:
            raise ValueError('disabled passive is a separate candidate family, not this joint/fluid chain')
        inputs={} if inputs is None else inputs
        for name,shape in (('qpos',(capacity,self.nq)),('qvel',(capacity,self.nv)),
            ('cvel',(capacity,self.nbody,6)),('subtree_com',(capacity,self.nbody,3)),
            ('xipos',(capacity,self.nbody,3)),('ximat',(capacity,self.nbody,9)),
            ('cdof',(capacity,self.nv,6))):
            self._input(name,shape,scalar.wp_real,inputs)
        self._input('active',(capacity,),wp.int32,inputs)
        self.qfrc=wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device)
        self.status=wp.zeros(capacity,dtype=wp.int32,device=self.device)
        self.stages=wp.zeros((capacity,self.nbody,33+7*self.nv),dtype=scalar.wp_real,device=self.device)
        self._finish_prepare()
    def launch(self, stream=None):
        self.validate_bindings()
        wp.launch(joint_passive_count,dim=self.capacity,inputs=[self.qpos,self.qvel,self.active,
            self.jtype,self.qadr,self.dadr,self.stiffness,self.spring,self.damping,self.qfrc,
            self.status,self.nv,self.nq,self.njnt],device=self.device,stream=stream)
        wp.launch(fluid_count,dim=self.capacity,inputs=[self.cvel,self.subtree_com,self.xipos,
            self.ximat,self.cdof,self.active,self.rootid,self.mass,self.box,self.ancestor,
            self.wind,self.opt,self.passive_disabled,0,self.stages,self.qfrc,self.status,
            self.nbody,self.nv],device=self.device,stream=stream)


class GenericActuationWorkspace(_Prepared):
    FIELDS=('actuator_gaintype','actuator_gainprm','actuator_biastype','actuator_biasprm',
        'actuator_dyntype','actuator_ctrllimited','actuator_ctrlrange',
        'actuator_forcelimited','actuator_forcerange')
    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', inputs=None):
        self._init(arrays,metadata,spec,capacity,self.FIELDS,registry,candidate_id,
                   source_records,metadata_record,mode,device)
        s=self.static; d=metadata['dimensions']
        if int(d.get('na',0)) or np.any(s['actuator_gaintype']!=0) or np.any(s['actuator_dyntype']!=0):
            raise ValueError('only zero-activation constant-gain actuation implemented')
        if np.any(~np.isin(s['actuator_biastype'],[0,1])) or np.any(s['actuator_ctrllimited']!=1):
            raise ValueError('bias/control-limit actuation family unsupported')
        for key in self.FIELDS:
            if s[key].dtype.kind=='f' and not np.isfinite(s[key]).all():
                raise ValueError('finite actuation static source: '+key)
        for name,value in (('gain',s['actuator_gainprm'][:,0]),('bias0',s['actuator_biasprm'][:,0]),
            ('bias1',s['actuator_biasprm'][:,1]),('bias2',s['actuator_biasprm'][:,2]),
            ('bias_type',s['actuator_biastype']),('ctrl_lo',s['actuator_ctrlrange'][:,0]),
            ('ctrl_hi',s['actuator_ctrlrange'][:,1]),('force_lo',s['actuator_forcerange'][:,0]),
            ('force_hi',s['actuator_forcerange'][:,1]),('force_limited',s['actuator_forcelimited'].astype(np.int32))):
            self._static(name,value)
        if np.any(s['actuator_ctrlrange'][:,0]>s['actuator_ctrlrange'][:,1]):
            raise ValueError('native ctrl range source')
        self.disable_flags=wp.int32(metadata['options']['disableflags'])
        inputs={} if inputs is None else inputs
        for name,shape,dtype in (('length',(capacity,self.nu),scalar.wp_real),
            ('moment',(capacity,self.nu,self.nv),scalar.wp_real),('velocity',(capacity,self.nu),scalar.wp_real),
            ('ctrl',(capacity,self.nu),scalar.wp_real),('active',(capacity,),wp.int32),
            ('vel_status',(capacity,),wp.int32)):
            self._input(name,shape,dtype,inputs)
        for name in ('force','force_pre','ctrl_used'):
            setattr(self,name,wp.zeros((capacity,self.nu),dtype=scalar.wp_real,device=self.device))
        for name in ('ctrl_route','force_route'):
            setattr(self,name,wp.zeros((capacity,self.nu),dtype=wp.int32,device=self.device))
        self.qfrc_actuator=wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device)
        for name in ('status','warn_badctrl_count','warn_badctrl_lastinfo','warning_number','warning_lastinfo'):
            setattr(self,name,wp.zeros(capacity,dtype=wp.int32,device=self.device))
        self._finish_prepare()
    def launch(self,stream=None):
        self.validate_bindings()
        names=('length','moment','velocity','ctrl','active','vel_status','gain','bias0','bias1','bias2',
            'bias_type','ctrl_lo','ctrl_hi','force_lo','force_hi','force_limited','disable_flags',
            'force','force_pre','ctrl_used','ctrl_route','force_route','qfrc_actuator','status',
            'warn_badctrl_count','warn_badctrl_lastinfo','warning_number','warning_lastinfo')
        wp.launch(actuation_count,dim=self.capacity,inputs=[getattr(self,k) for k in names]+[self.nv,self.nu],
            device=self.device,stream=stream)


class GenericSmoothWorkspace(_Prepared):
    """Original three RN -> ordered body projection -> packed single RHS solve.

    This is an atomic working workspace. Durable public commit and producer
    clocks belong to the runtime publication adapter; atomic failed results
    preserve the original zero-output semantics.
    """
    FIELDS=('body_parentid','body_rootid','body_dofadr','body_dofnum',
        'dof_parentid','dof_Madr','dof_simplenum')
    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', inputs=None):
        self._init(arrays,metadata,spec,capacity,self.FIELDS,registry,candidate_id,
                   source_records,metadata_record,mode,device)
        from libero.gpu.reference.dynamics.external_force_projection.kernel import StaticTopology
        from libero.gpu.reference.integration.euler_integrator.generic_workspace import GenericMassWorkspace
        topology=StaticTopology({k:self.static[k] for k in
            ('body_parentid','body_rootid','body_dofadr','body_dofnum','dof_parentid')})
        if topology.nbody!=self.nbody or topology.nv!=self.nv:
            raise ValueError('actual smooth topology dimension mismatch')
        self._static('rootid',self.static['body_rootid'])
        self._static('ancestor',topology.ancestor_mask)
        self._static('body_ids',np.tile(np.arange(1,self.nbody,dtype=np.int32),(capacity,1)))
        self._static('body_count',np.full(capacity,self.nbody-1,np.int32))
        self.solver=GenericMassWorkspace(metadata,self.static,capacity,self.device)
        self.solver.count.fill_(1)
        inputs={} if inputs is None else inputs
        for name,shape,dtype in (('passive',(capacity,self.nv),scalar.wp_real),
            ('bias',(capacity,self.nv),scalar.wp_real),('applied',(capacity,self.nv),scalar.wp_real),
            ('actuator',(capacity,self.nv),scalar.wp_real),('xfrc',(capacity,self.nbody,6),scalar.wp_real),
            ('xipos',(capacity,self.nbody,3),scalar.wp_real),('subtree_com',(capacity,self.nbody,3),scalar.wp_real),
            ('cdof',(capacity,self.nv,6),scalar.wp_real),('qm',(capacity,self.solver.nM),scalar.wp_real),
            ('active',(capacity,),wp.int32)):
            self._input(name,shape,dtype,inputs)
        # All assignments are prepare-only and followed by the sole token freeze.
        self.solver.qm=self.qm;self.solver.active=self.active
        for name in ('smooth','seed','qfrc','scratch','qacc_smooth'):
            setattr(self,name,wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device))
        self.stages=wp.zeros((capacity,1,1,1),dtype=scalar.wp_real,device=self.device)
        for name in ('entry_status','d02_status','status'):
            setattr(self,name,wp.zeros(capacity,dtype=wp.int32,device=self.device))
        self._finish_prepare()
        self._solver_tokens={k:_token(v) for k,v in vars(self.solver).items() if isinstance(v,wp.array)}
    def validate_bindings(self):
        super().validate_bindings()
        for key,token in self._solver_tokens.items():
            if _token(getattr(self.solver,key))!=token:
                raise ValueError('actual sparse solve owner drift: '+key)
    def launch(self,stream=None):
        self.validate_bindings()
        from .acceleration import _three_rn, _seed_rhs, _route
        from libero.gpu.reference.dynamics.external_force_projection.kernel import _project
        wp.launch(_three_rn,dim=self.capacity,inputs=[self.passive,self.bias,self.applied,
            self.actuator,self.active,self.entry_status,self.smooth,self.seed,self.nv],
            device=self.device,stream=stream)
        wp.launch(_project,dim=self.capacity,inputs=[self.seed,self.xfrc,self.xipos,self.subtree_com,
            self.cdof,self.body_ids,self.body_count,self.active,self.rootid,self.ancestor,
            self.scratch,self.stages,self.qfrc,self.d02_status,self.nbody,self.nv,0],
            device=self.device,stream=stream)
        wp.launch(_seed_rhs,dim=self.capacity,inputs=[self.qfrc,self.solver.rhs,self.active,
            self.entry_status,self.d02_status,self.nv],device=self.device,stream=stream)
        self.solver.launch(stream=stream)
        wp.launch(_route,dim=self.capacity,inputs=[self.entry_status,self.d02_status,
            self.solver.status,self.active,self.solver.solution,self.qfrc,self.smooth,
            self.qacc_smooth,self.status,self.nv],device=self.device,stream=stream)
