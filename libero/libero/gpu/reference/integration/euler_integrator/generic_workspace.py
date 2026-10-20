"""Diagnostic candidate for validated MuJoCo 2.3.7 Euler topologies.

Old EulerWorkspace/MassSolveWorkspace constructors and licences are unchanged.
This new allocation path reuses their exact device arithmetic, including native
pivot repair and certified quaternion trigonometry. New model qualification is
independent: structural readiness alone is never a production acceptance.
"""
import libero.gpu.precision as scalar
import warp as wp
from .kernel import EulerWorkspace
from .specialization import validate_specialization
from libero.gpu.reference.position.mass_solve.kernel import MassSolveWorkspace

class GenericMassWorkspace(MassSolveWorkspace):
    def __init__(self, metadata, arrays, capacity, device):
        self.nv=metadata['dimensions']['nv'];self.nM=metadata['dimensions']['nM']
        self.capacity,self.device=capacity,device
        for n in ('dof_Madr','dof_parentid','dof_simplenum'):
            setattr(self,n,wp.array(arrays[n],dtype=wp.int32,device=device))
        for n,shape,dtype in (
            ('qm',(capacity,self.nM),scalar.wp_real),('rhs',(capacity,4,self.nv),scalar.wp_real),
            ('count',(capacity,),wp.int32),('active',(capacity,),wp.int32),
            ('ld',(capacity,self.nM),scalar.wp_real),('inv',(capacity,self.nv),scalar.wp_real),
            ('sqrt_inv',(capacity,self.nv),scalar.wp_real),('solution',(capacity,4,self.nv),scalar.wp_real),
            ('pivot',(capacity,self.nv),wp.int32),('status',(capacity,),wp.int32),
            ('warning_count',(capacity,),wp.int32),('warning_last',(capacity,),wp.int32)):
            setattr(self,n,wp.zeros(shape,dtype=dtype,device=device))

class GenericEulerWorkspace(EulerWorkspace):
    """New-layout diagnostic workspace; not the old frozen qualification.

    Runtime registry must supply independent acceptance before production
    dispatch. This class intentionally declares qualification pending.
    Inherits upload/launch unchanged; no host-side substep calculation.
    """
    qualification_status='pending_independent_model_acceptance'
    def __init__(self, model, capacity, device='cuda:0', *, mode='diagnostic'):
        if mode!='diagnostic':
            raise ValueError('generic Euler production requires independent registry-qualified adapter')
        if isinstance(capacity,bool) or int(capacity)!=capacity or capacity<=0:
            raise ValueError('Euler capacity must be a positive integer')
        capacity=int(capacity)
        metadata,arrays=validate_specialization(model)
        self.specialization=metadata
        self.model_identity=metadata['specialization_key']
        for n,v in metadata['dimensions'].items():setattr(self,n,v)
        self.capacity,self.device,self.dt=capacity,device,float(model.opt.timestep)
        self.mass=GenericMassWorkspace(metadata,arrays,capacity,device)
        for n in ('dof_damping','jnt_type','jnt_qposadr','jnt_dofadr'):
            setattr(self,n,wp.array(arrays[n],dtype=scalar.wp_real if n=='dof_damping' else wp.int32,device=device))
        self.nquat=sum(int(t) in (0,1) for t in arrays['jnt_type'])
        for n,shape in (
            ('qpos',(capacity,self.nq)),('qvel',(capacity,self.nv)),('time',(capacity,)),
            ('qM',(capacity,self.nM)),('qfrc_smooth',(capacity,self.nv)),
            ('qfrc_constraint',(capacity,self.nv)),('qacc',(capacity,self.nv)),
            ('qH',(capacity,self.nM)),('qHDiagInv',(capacity,self.nv)),
            ('qpos_after',(capacity,self.nq)),('qvel_after',(capacity,self.nv)),
            ('qacc_used',(capacity,self.nv)),('time_after',(capacity,)),
            ('axis',(capacity,self.nquat,3)),('angle',(capacity,self.nquat)),
            ('increment',(capacity,self.nquat,4)),('pre_norm',(capacity,self.nquat,4)),
            ('qacc_warmstart',(capacity,self.nv)),('zero_work',(capacity,self.nv))):
            setattr(self,n,wp.zeros(shape,dtype=scalar.wp_real,device=device))
        for n,shape in (('active',(capacity,)),('damped',(capacity,)),('status',(capacity,)),
                        ('axis_fallback',(capacity,self.nquat)),('zero_angle',(capacity,self.nquat)),
                        ('quat_fallback',(capacity,self.nquat)),('zero_skip',(capacity,self.nv))):
            setattr(self,n,wp.zeros(shape,dtype=wp.int32,device=device))
        self.ctrl=wp.zeros((capacity,self.nu),dtype=scalar.wp_real,device=device) if self.nu else None
        self.sensordata=wp.zeros((capacity,self.nsensordata),dtype=scalar.wp_real,device=device) if self.nsensordata else None
