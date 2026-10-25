"""New-model resident dynamics backend, including real controller operands.

No PRE whitelist or fixed-model licence is borrowed. This adapter requires
structural admission for the actual model, reuses original arithmetic, and
publishes only device-successful results. Numerical acceptance is separate.
"""
import libero.gpu.precision as scalar
from types import SimpleNamespace
import numpy as np
import warp as wp
from . import runtime_adapter as r
from .specialization import GenericPositionWorkspace, GenericTransmissionWorkspace, GenericJacobianInputsWorkspace, native_fields
from .generic_stages import GenericPassiveWorkspace, GenericActuationWorkspace, GenericSmoothWorkspace
from .publication import commit_bias, commit_smooth


@wp.kernel(enable_backward=False)
def _site(pos:wp.array(dtype=scalar.wp_real),mat:wp.array(dtype=scalar.wp_real),active:wp.array(dtype=wp.int32),
          outp:wp.array(dtype=scalar.wp_real,ndim=2),outm:wp.array(dtype=scalar.wp_real,ndim=2),ns:int,site:int):
    w=wp.tid()
    if active[w]==1:
        for k in range(3):outp[w,k]=pos[w*ns*3+site*3+k]
        for k in range(9):outm[w,k]=mat[w*ns*9+site*9+k]


class GenericRuntimeDynamicsAdapter:
    """Same assembly-facing phase API as fixed LiveDynamicsAdapter.

    ``candidate_ids`` includes kinematics, frames, mass and dynamics. The last
    candidate covers the actual model and supported parameter domain.
    ``robot`` is an actual decoder robot_metadata member.
    """
    def __init__(self,arrays,metadata,spec,capacity,*,robot,registry,candidate_ids,
                 source_records,metadata_record,state_epoch,first_error,stream,mode='production',device='cuda:0'):
        from libero.gpu.reference.dynamics.comvel.kernel import ComVelModel
        from libero.gpu.reference.dynamics.rne_bias.kernel import RNEBiasModel
        from libero.gpu.reference.control.osc.controller import PandaOSCWorkspace
        self.capacity,self.device,self.stream=capacity,wp.get_device(device),stream
        if not self.device.is_cuda:raise ValueError('real CUDA dynamics runtime required')
        r._array(state_epoch,(capacity,),wp.int64,self.device);r._array(first_error,(capacity,),wp.int32,self.device)
        self.state_epoch,self.first_error=state_epoch,first_error
        d=metadata['dimensions'];self.nv=int(d['nv']);self.nq=int(d['nq']);self.nbody=int(d['nbody']);self.nsite=int(d['nsite'])
        self.nu=int(d['nu'])
        fields=('body_parentid','body_dofadr','body_dofnum','dof_bodyid','dof_jntid','jnt_type')
        static,_=native_fields(arrays,metadata,spec,fields,source_records=source_records,
            metadata_record=metadata_record,registry=registry,candidate_id=candidate_ids['dynamics'],mode=mode)
        self.spec,self.registry,self.candidate_ids,self.mode=spec,registry,dict(candidate_ids),mode
        common=dict(registry=registry,source_records=source_records,metadata_record=metadata_record,mode=mode,device=self.device)
        self.position=GenericPositionWorkspace(arrays,metadata,spec,capacity,candidate_ids=candidate_ids,**common)
        self.qpos=self.position.qpos;self.qvel=wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device)
        for name in ('active','requested_mask','gate_velocity','gate_bias','actuation_active','gate_acceleration','gate_control'):
            setattr(self,name,wp.zeros(capacity,dtype=wp.int32,device=self.device))
        self.gate_position=self.position.active
        self.forward_epoch=wp.zeros(capacity,dtype=wp.int64,device=self.device)
        self.source_state_epoch=wp.zeros(capacity,dtype=wp.int64,device=self.device)
        self.valid={k:wp.zeros(capacity,dtype=wp.int32,device=self.device) for k in ('position','velocity','bias','smooth')}
        self.written={k:wp.zeros(capacity,dtype=wp.int64,device=self.device) for k in self.valid}
        self.public={k:wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device) for k in ('qfrc_bias','qfrc_smooth','qacc_smooth')}
        self.smooth_repaired=wp.zeros(capacity,dtype=wp.int32,device=self.device)
        self.transmission=GenericTransmissionWorkspace(arrays,metadata,spec,capacity,
            candidate_id=candidate_ids['dynamics'],inputs=dict(qpos=self.qpos,qvel=self.qvel,
            position_active=self.gate_position,velocity_active=self.gate_velocity),**common)
        for name,width in (('cvel',self.nbody*6),('cdof_dot',self.nv*6),('cacc',self.nbody*6),('frc',self.nbody*6),('qfrc_bias',self.nv)):
            setattr(self,name,wp.zeros((capacity,width),dtype=scalar.wp_real,device=self.device))
        for name in ('status_comvel','status_rne','status_position','status_velocity'):
            setattr(self,name,wp.zeros(capacity,dtype=wp.int32,device=self.device))
        model=SimpleNamespace(**d,**static,opt_gravity=np.asarray(metadata['options']['gravity'],np.float64),
            gravity_disabled=np.asarray([bool(int(metadata['options']['disableflags'])&64)],np.int32))
        self.comvel_model=ComVelModel.prepare(model,device=self.device)
        self.rne_model=RNEBiasModel.prepare(model,device=self.device)
        body=lambda a,width:a.reshape((capacity,self.nbody,width))
        self.passive=GenericPassiveWorkspace(arrays,metadata,spec,capacity,candidate_id=candidate_ids['dynamics'],
            inputs=dict(qpos=self.qpos,qvel=self.qvel,active=self.gate_velocity,cvel=body(self.cvel,6),
                cdof=self.position.cdof.reshape((capacity,self.nv,6)),subtree_com=body(self.position.subtree_com,3),
                xipos=body(self.position.frames.xipos,3),ximat=body(self.position.frames.ximat,9)),**common)
        self.actuation=GenericActuationWorkspace(arrays,metadata,spec,capacity,candidate_id=candidate_ids['dynamics'],
            inputs=dict(length=self.transmission.length,moment=self.transmission.moment,velocity=self.transmission.velocity,
                vel_status=self.transmission.velocity_status,active=self.actuation_active),**common)
        self.qfrc_applied=wp.zeros((capacity,self.nv),dtype=scalar.wp_real,device=self.device)
        self.xfrc_applied=wp.zeros((capacity,self.nbody,6),dtype=scalar.wp_real,device=self.device)
        self.acceleration=GenericSmoothWorkspace(arrays,metadata,spec,capacity,candidate_id=candidate_ids['dynamics'],
            inputs=dict(passive=self.passive.qfrc,bias=self.public['qfrc_bias'],applied=self.qfrc_applied,
                actuator=self.actuation.qfrc_actuator,xfrc=self.xfrc_applied,xipos=body(self.position.frames.xipos,3),
                subtree_com=body(self.position.subtree_com,3),cdof=self.position.cdof.reshape((capacity,self.nv,6)),
                qm=self.position.qM,active=self.gate_acceleration),**common)
        self.site_position=wp.zeros((capacity,3),dtype=scalar.wp_real,device=self.device)
        self.site_rotation=wp.zeros((capacity,9),dtype=scalar.wp_real,device=self.device)
        self.joint_controller=robot['controller_config']['type'] in ('JOINT_TORQUE','JOINT_POSITION','JOINT_VELOCITY')
        if self.joint_controller:
            from libero.gpu.host.runtime.control.joint_inputs import GenericJointInputsWorkspace
            from libero.gpu.host.runtime.control.joint import JointControllerWorkspace
            self.jacobian=GenericJointInputsWorkspace(arrays,metadata,spec,capacity,robot=robot,
                candidate_id=candidate_ids['dynamics'],**common)
            self.control=JointControllerWorkspace(capacity,robot=robot,nu=self.nu,
                timestep=float(robot.get('controller_model_timestep',metadata['options']['timestep'])),policy_freq=float(robot['control_freq']),device=self.device)
            n=self.control.joint_dim
            self.control.phys_mass=self.jacobian.M.reshape((capacity,n*n))
            self.control.phys_q=self.jacobian.joint_pos
            self.control.phys_qvel=self.jacobian.joint_vel
        else:
            self.jacobian=GenericJacobianInputsWorkspace(arrays,metadata,spec,capacity,robot=robot,
                candidate_id=candidate_ids['dynamics'],inputs=dict(qpos=self.qpos,qvel=self.qvel,qM=self.position.qM,
                    qfrc_bias=self.public['qfrc_bias'],cdof=self.position.cdof.reshape((capacity,self.nv,6)),
                    subtree_com=body(self.position.subtree_com,3),site_pos=self.site_position,site_mat=self.site_rotation,
                    active=self.gate_control),**common)
            self.control=PandaOSCWorkspace(capacity,device=self.device,uncouple=True,diagnostics=True)
            cfg=robot['controller_config']
            if cfg['type'] not in ('OSC_POSE','OSC_POSITION'):
                raise NotImplementedError('GPU controller family not implemented: '+str(cfg['type']))
            kp=np.broadcast_to(np.asarray(cfg['kp'],np.float64),(6,)).copy()
            damping=np.broadcast_to(np.asarray(cfg['damping_ratio'],np.float64),(6,))
            kd=2.0*np.sqrt(kp)*damping
            if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp<0):raise ValueError('controller gains source')
            self.control.gains.assign(np.tile(np.r_[kp,kd],(capacity,1)))
            self.control.name=cfg['type']
            self.control.position_only=cfg['type']=='OSC_POSITION'
            self.control.control_limits=(np.full(3 if cfg['type']=='OSC_POSITION' else 6,-1.),np.full(3 if cfg['type']=='OSC_POSITION' else 6,1.))
            from libero.gpu.host.runtime.control.robot_osc import prepare_robot_osc
            prepare_robot_osc(self.control,robot,self.nu)
            from libero.gpu.host.runtime.control.variable_osc import prepare_variable_osc
            prepare_variable_osc(self.control,cfg)
            self.control.uncouple=int(bool(cfg.get('uncouple_pos_ori',True)))
            from libero.gpu.host.runtime.control.general_osc import prepare_general_osc
            prepare_general_osc(self.control,cfg,timestep=float(robot.get('controller_model_timestep',metadata['options']['timestep'])),policy_freq=float(robot['control_freq']))
            self.control.phys_mass=self.jacobian.M.reshape((capacity,49))
            self.control.phys_jacobian=self.jacobian.J.reshape((capacity,42))
            self.control.phys_q=self.jacobian.joint_pos;self.control.phys_qvel=self.jacobian.joint_vel
            self.control.phys_ee_pos=self.jacobian.ee_pos;self.control.phys_ee_rot=self.jacobian.ee_mat
        self._arm_indices=self.jacobian.vi
        self.controller_parent_valid=wp.zeros(capacity,dtype=wp.int32,device=self.device)
        self.controller_parent_written=wp.zeros(capacity,dtype=wp.int64,device=self.device)
        # Honest compatibility views for assembly geometry; not fixed ModelPack.
        self.position.mass=SimpleNamespace(mass=self.position.factor,xipos=self.position.frames.xipos,
            ximat=self.position.frames.ximat,subtree_com=self.position.subtree_com,
            cinert=self.position.cinert,cdof=self.position.cdof)
        self.stages=self
        self.robot=robot;self.metadata=metadata
        self._objects=(self.position,self.transmission,self.passive,self.actuation,self.acceleration,self.jacobian,self.control)
        self._holds={phase:r._NoForwardHold(fields,capacity,self.device) for phase,fields in {
            'position':{**{k:getattr(self.position,k) for k in ('subtree_com','cinert','cdof','crb','result_kind','first_code','first_stage')},
                **{'body.'+k:getattr(self.position.body,k) for k in ('qpos','xpos','xquat','xmat','xanchor','xaxis','status')},
                **{'frame.'+k:getattr(self.position.frames,k) for k in ('xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat','status')},
                **{'factor.'+k:getattr(self.position.factor,k) for k in ('qm','ld','inv','sqrt_inv','status')},
                **{'tr.'+k:getattr(self.transmission,k) for k in ('length','moment','position_status')},
                'valid':self.valid['position'],'written':self.written['position']},
            'velocity':{'cvel':self.cvel,'cdof_dot':self.cdof_dot,'passive':self.passive.qfrc,'status':self.passive.status,
                'trvel':self.transmission.velocity,'valid':self.valid['velocity'],'written':self.written['velocity']},
            'bias':{'cacc':self.cacc,'frc':self.frc,'bias':self.qfrc_bias,'status':self.status_rne,
                'valid':self.valid['bias'],'written':self.written['bias']},
            'smooth':{**{'act.'+k:getattr(self.actuation,k) for k in ('force','force_pre','ctrl_used','ctrl_route','force_route','qfrc_actuator','status','warning_number','warning_lastinfo')},
                **{'acc.'+k:getattr(self.acceleration,k) for k in ('smooth','qfrc','qacc_smooth','status')},
                'valid':self.valid['smooth'],'written':self.written['smooth'],'repair':self.smooth_repaired}}.items()}
        self._frozen=self._tokens()
        self._control_tokens={k:r._token(v) for k,v in vars(self.control).items() if isinstance(v,wp.array)}
        self._stream_token=(id(stream),int(stream.cuda_stream),int(self.device.context))
    def _tokens(self):
        return {k:r._token(v) for k,v in vars(self).items() if isinstance(v,wp.array)}|{
            f'{kind}.{k}':r._token(v) for kind,mapping in (('valid',self.valid),('written',self.written),('public',self.public)) for k,v in mapping.items()}
    def validate_bindings(self):self.validate()
    def validate(self,stream=None):
        if stream is not None and stream is not self.stream:raise ValueError('foreign prepared dynamics stream')
        if self._stream_token!=(id(self.stream),int(self.stream.cuda_stream),int(self.device.context)):raise ValueError('prepared stream/context drift')
        if self._tokens()!=self._frozen:raise ValueError('actual dynamics owner drift')
        if self._objects!=(self.position,self.transmission,self.passive,self.actuation,self.acceleration,self.jacobian,self.control):
            raise ValueError('actual prepared dynamics module owner drift')
        if self._control_tokens!={k:r._token(v) for k,v in vars(self.control).items() if isinstance(v,wp.array)}:
            raise ValueError('actual controller physical/config owner drift')
        for workspace in self._objects[:-1]:workspace.validate_bindings()
    def _launch(self,kernel,inputs):wp.launch(kernel,dim=self.capacity,inputs=inputs,device=self.device,stream=self.stream)
    begin_forward=r.LiveDynamicsAdapter.begin_forward
    _gate=r.LiveDynamicsAdapter._gate
    @property
    def position_valid(self):return self.valid['position']
    @property
    def velocity_valid(self):return self.valid['velocity']
    @property
    def bias_valid(self):return self.valid['bias']
    @property
    def smooth_valid(self):return self.valid['smooth']
    def launch_position(self,stream=None):
        self.validate(stream);self._holds['position'].snapshot(self.stream)
        self._launch(r._gate,[self.active,self.first_error,self.state_epoch,self.source_state_epoch,
            self.forward_epoch,self.valid['position'],self.written['position'],self.gate_position,0])
        self.position.launch(stream=self.stream);self.transmission.launch_position(stream=self.stream)
        self._launch(r._position_status,[self.position.result_kind,self.position.first_code,self.transmission.position_status,self.status_position])
        self._launch(r._finish,[self.gate_position,self.status_position,self.forward_epoch,self.valid['position'],self.written['position'],self.first_error,1,210])
        self._holds['position'].restore_unrequested(self.requested_mask,self.stream)
    def launch_velocity(self,stream=None):
        from libero.gpu.reference.dynamics.comvel.kernel import launch
        self.validate(stream);self._holds['velocity'].snapshot(self.stream);self._gate(self.gate_velocity,'position')
        self.transmission.launch_velocity(stream=self.stream)
        launch(self.comvel_model,self.gate_velocity,self.position.cdof,self.qvel,self.cvel,self.cdof_dot,self.status_comvel,stream=self.stream)
        self.passive.launch(stream=self.stream)
        self._launch(r._velocity_status,[self.transmission.velocity_status,self.status_comvel,self.passive.status,self.status_velocity])
        self._launch(r._finish,[self.gate_velocity,self.status_velocity,self.forward_epoch,self.valid['velocity'],self.written['velocity'],self.first_error,0,212])
        self._holds['velocity'].restore_unrequested(self.requested_mask,self.stream)
    def launch_bias(self,*,reference_done,reference_error,reference_written,reference_expected,stream=None):
        from libero.gpu.reference.dynamics.rne_bias.kernel import launch
        self.validate(stream);self._holds['bias'].snapshot(self.stream);self._gate(self.gate_bias,'velocity')
        for a,dtype in ((reference_done,wp.int32),(reference_error,wp.int32),(reference_written,wp.int64),(reference_expected,wp.int64)):
            r._array(a,(self.capacity,),dtype,self.device)
        self._launch(r._reference_gate,[reference_done,reference_error,reference_written,reference_expected,self.forward_epoch,self.gate_bias,self.first_error])
        launch(self.rne_model,self.gate_bias,self.cdof_dot,self.qvel,self.position.cinert,self.cvel,self.position.cdof,self.cacc,self.frc,self.qfrc_bias,self.status_rne,stream=self.stream)
        self._launch(commit_bias,[self.active,self.gate_bias,self.status_rne,self.forward_epoch,self.first_error,
            self.qfrc_bias,self.public['qfrc_bias'],self.valid['bias'],self.written['bias'],self.nv])
        self._holds['bias'].restore_unrequested(self.requested_mask,self.stream)
    def launch_force(self,*,ctrl,ctrl_valid,stream=None):
        self.validate(stream);self._holds['smooth'].snapshot(self.stream)
        r._array(ctrl,(self.capacity,self.nu),scalar.wp_real,self.device);r._array(ctrl_valid,(self.capacity,),wp.int32,self.device)
        self._gate(self.actuation_active,'bias');self._launch(r._force_gate,[ctrl_valid,self.actuation_active])
        self._launch(r._copy_admitted,[self.actuation_active,ctrl,self.actuation.ctrl,self.nu]);self.actuation.launch(stream=self.stream)
        self._launch(r._actuation_finish,[self.actuation_active,self.actuation.status,self.first_error])
        self._gate(self.gate_acceleration,'bias');self._launch(r._force_gate,[self.actuation_active,self.gate_acceleration])
        self.acceleration.launch(stream=self.stream)
        self._launch(commit_smooth,[self.active,self.gate_acceleration,self.acceleration.status,
            self.forward_epoch,self.first_error,self.acceleration.qfrc,self.acceleration.qacc_smooth,
            self.public['qfrc_smooth'],self.public['qacc_smooth'],self.valid['smooth'],self.written['smooth'],self.smooth_repaired,self.nv])
        self._holds['smooth'].restore_unrequested(self.requested_mask,self.stream)
    def gather_controller_inputs(self,mask,stream=None):
        self.validate(stream);r._array(mask,(self.capacity,),wp.int32,self.device)
        if self.joint_controller:
            return self.jacobian.gather(mask,stream=self.stream)
        self._launch(r._gate,[mask,self.first_error,self.state_epoch,self.source_state_epoch,self.forward_epoch,
            self.valid['bias'],self.written['bias'],self.gate_control,1])
        self._launch(_site,[self.position.frames.site_xpos,self.position.frames.site_xmat,self.gate_control,
            self.site_position,self.site_rotation,self.nsite,int(self.robot['eef_site_id'])])
        self.jacobian.launch(stream=self.stream)
        self._launch(r._selected_velocity,[self.gate_control,self.jacobian.status,self.jacobian.xvelp,self.jacobian.xvelr,self.control.phys_ee_velocity,self.first_error])
    def gather_live_bias(self,mask,stream=None):
        self.validate(stream)
        if self.joint_controller:return self.jacobian.live_bias(mask,stream=self.stream)
        return r.LiveDynamicsAdapter.gather_live_bias(self,mask,stream=stream)
    controller_inputs=r.LiveDynamicsAdapter.controller_inputs
    def public_arrays(self):return dict(self.public)
    def catalog(self):return {'candidate':'generic native-count dynamics','model':self.spec.model_sha256,
        'bindings':self._frozen,'mode':self.mode,'numerics':'pending independent actual-model acceptance'}
