import libero.gpu.precision as scalar
from .copy_factory import prepare_copy_collision
"""Fixed-model cold assembly; only model data and reset state cross H2D."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import inspect
import numpy as np
import warp as wp


def prepare_fixed_copy_tape_physics(pack, *, state, device='cuda:0', stream=None, with_controller=False,
                          tape_steps=0, memory_budget=None):
    """Prepare one physical world for the first real state-feedback test.

    ``state`` is an explicit reset-boundary packet, not CPU forward truth.
    The output records missing controller/sensor hooks so this first milestone
    cannot be mistaken for a complete robot/environment implementation.
    """
    from libero.gpu.host.runtime.dynamics.runtime_adapter import RuntimeDynamicsAdapter, RuntimePostAdapter, RuntimeControllerAdapter
    from libero.gpu.host.runtime.dynamics.rules import load_rules
    from libero.gpu.host.runtime.collision.runtime_adapter import CollisionRuntimeAdapter, plan_capacity, FIXED_ACCEPTED_CAPACITIES
    from libero.gpu.host.runtime.constraints.workspace import PositionReferenceWorkspace
    from libero.gpu.host.runtime.constraints.runtime_adapter import ResidentConstraintRuntime
    from libero.gpu.host.runtime.constraints.identity import STATIC_SCHEMA, MODEL_FACTS
    from libero.gpu.host.runtime.newton.workspace import NewtonFwdWorkspace
    from libero.gpu.host.runtime.newton.parallel_tape_adapter import DiagnosticParallelTapeFixedAdapter as FixedNewtonRuntimeAdapter
    from libero.gpu.host.runtime.foundation.runtime_adapter import FoundationForwardAdapter
    from libero.gpu.reference.integration.euler_integrator import EulerWorkspace
    from libero.gpu.model.collision_static import static_from_model
    from .contact_transport import ContactTransport
    from .physics import FixedPhysicsForward
    from .runtime import ResidentStepRuntime

    model, W, R, C = pack.model, 1, 6000, 1900
    required = {'qpos':(W,48),'qvel':(W,43),'time':(W,), 'ctrl':(W,9),
                'qacc_warmstart':(W,43),'qfrc_applied':(W,43),'xfrc_applied':(W,38,6)}
    if set(state) != set(required):
        raise ValueError('reset packet must contain all fixed integration-state fields')
    for name, shape in required.items():
        if state[name].dtype != np.float64 or state[name].shape != shape:
            raise TypeError('reset boundary schema: '+name)
    stream = wp.get_stream(device) if stream is None else stream
    with wp.ScopedStream(stream):
        epoch = wp.zeros(W,dtype=wp.int64,device=device)
        expected_state = wp.zeros(W,dtype=wp.int64,device=device)
        first = wp.zeros(W,dtype=wp.int32,device=device)
        # These are explicit reset/action intent, never a manufactured forward
        # completion. All actual D/S/N completion starts invalid.
        intent = wp.full(W,1,dtype=wp.int32,device=device)
        state_valid = wp.full(W,1,dtype=wp.int32,device=device)
        fmask = wp.zeros(W,dtype=wp.int32,device=device)
        d = RuntimeDynamicsAdapter.prepare(pack,W,state_epoch=epoch,first_error=first,
                                           stream=stream,device=device)
        d.stages.upload_state(active=np.ones(W,np.int32),qpos=state['qpos'],qvel=state['qvel'],
            mocap_pos=np.zeros((W,0,3),np.float64),mocap_quat=np.zeros((W,0,4),np.float64),
            rhs=np.zeros((W,4,43),np.float64),count=np.ones(W,np.int32),
            qfrc_applied=state['qfrc_applied'],xfrc_applied=state['xfrc_applied'])
        ctrl = wp.array(state['ctrl'],dtype=scalar.wp_real,device=device)
        ctrl_valid = state_valid  # valid external ctrl input at the reset boundary
        independent_forward = wp.zeros(W,dtype=wp.int64,device=device)
        source_records = [{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in (Path(inspect.getfile(type(d))),Path(inspect.getfile(type(d.stages))),pack.mjb_path)]
        f = FoundationForwardAdapter(d,mask=fmask,expected_state_epoch=expected_state,
            state_valid=state_valid,nq=48,nv=43,source_records=source_records,stream=stream)
        pos = d.stages.position
        cactive = wp.zeros(W,dtype=wp.int32,device=device)
        # Only geometry views, not CPU contact or CPU position results.
        owner = SimpleNamespace(xpos=pos.frames.geom_xpos,xmat=pos.frames.geom_xmat,
                                xipos=pos.mass.xipos,ximat=pos.mass.ximat,active=cactive)
        cstatic = static_from_model(model)
        cplan = plan_capacity(cstatic,W,device=device,memory_budget=memory_budget,
                              pending_bytes=2048*1024**2,**FIXED_ACCEPTED_CAPACITIES)
        c = prepare_copy_collision(cstatic,cplan,source_owner=owner,
            source_paths={k:k for k in ('xpos','xmat','xipos','ximat','active')},stream=stream,device=device)
        transport = ContactTransport(c,d,nworld=W,capacity=C,stream=stream)
        sstatic = {key:np.ascontiguousarray(getattr(model,key),dtype=dtype)
                   for key,(shape,dtype) in STATIC_SCHEMA.items()}
        sw = PositionReferenceWorkspace(sstatic,W,R,C,model_facts=dict(MODEL_FACTS),device=device)
        cdof = pos.mass.cdof.flatten()
        subtree = pos.mass.subtree_com.flatten()
        s = ResidentConstraintRuntime(sw,stream,
            lambda:dict(transport.inputs,active=transport.active,ncon=transport.ncon,
                        qpos=d.qpos,cdof=cdof,subtree_com=subtree),
            lambda:dict(qvel=d.qvel,qvel_epoch=d.written['velocity'],expected_qvel_epoch=independent_forward))
        nw = NewtonFwdWorkspace(W,R,C,device=device,maxiter=int(model.opt.iterations),
                               s07_build_dir=Path('/proc/self/cwd/tmp/n-native').resolve())
        nw.stream = stream
        nw.bind_topology(np.asarray(model.dof_Madr,np.int32),np.asarray(model.dof_parentid,np.int32),
                         np.asarray(model.dof_simplenum,np.int32),
                         np.array([model.opt.tolerance,model.stat.meaninertia],np.float64))
        n = FixedNewtonRuntimeAdapter(nw,rules=load_rules(pack.mjb_path))
        reshape = lambda a,shape:a.reshape(shape)
        qm = pos.mass.mass.qm.reshape((W,153))
        warm = wp.array(state['qacc_warmstart'],dtype=scalar.wp_real,device=device)
        n.bind_inputs(dict(J=reshape(sw.efc_J,(W,R,43)),aref=reshape(sw.efc_aref,(W,R)),
            D=reshape(sw.efc_D,(W,R)),R=reshape(sw.efc_R,(W,R)),loss=reshape(sw.efc_frictionloss,(W,R)),
            efc_type=reshape(sw.efc_type,(W,R)),efc_id=reshape(sw.efc_id,(W,R)),qM=qm,
            qfrc_smooth=d.public['qfrc_smooth'],qacc_smooth=d.public['qacc_smooth'],qacc_warmstart=warm,
            contact_dim=reshape(transport.inputs['dim'],(W,C)),contact_mu=reshape(sw.contact_mu,(W,C)),
            contact_friction=reshape(transport.inputs['friction'],(W,C,5))))
        n.seal()
        physics = FixedPhysicsForward(dynamics=d,collision=c,transport=transport,constraints=s,
            newton=n,foundation=f,expected_forward=independent_forward,ctrl=ctrl,ctrl_valid=ctrl_valid,stream=stream)
        from libero.gpu.reference.postprocess.rne_postconstraint.kernel import RNEPostWorkspace
        integer_static = ('body_parentid','body_dofadr','body_dofnum','body_rootid','geom_bodyid',
            'site_bodyid','sensor_type','sensor_objtype','sensor_objid','sensor_adr','sensor_dim','sensor_needstage')
        ps = SimpleNamespace(**{key:np.ascontiguousarray(getattr(model,key),dtype=np.int32) for key in integer_static},
            **{key:np.ascontiguousarray(getattr(model,key),dtype=np.float64) for key in ('sensor_cutoff','sensor_noise')},
            nbody=38,nv=43,ngeom=270,nsite=31,nsensor=2,neq=0,cone=int(model.opt.cone),
            opt_enableflags=int(model.opt.enableflags),opt_disableflags=int(model.opt.disableflags),
            opt_gravity=np.asarray(model.opt.gravity,dtype=np.float64))
        post_work = RNEPostWorkspace(ps,W,C,R,device=device)
        public = n.public_arrays
        post_sources = dict(qvel=d.qvel,qacc=public['qacc'],cinert=pos.mass.cinert,
            cvel=d.stages.cvel,cdof=pos.mass.cdof,cdof_dot=d.stages.cdof_dot,xipos=pos.mass.xipos,
            subtree_com=pos.mass.subtree_com,site_xpos=pos.frames.site_xpos,site_xmat=pos.frames.site_xmat,
            xfrc_applied=d.stages.xfrc_applied,efc_force=public['efc_force'],ncon=transport.ncon,
            nefc=nw.nefc,ne=nw.ne,contact_dim=transport.inputs['dim'],contact_efc_address=sw.contact_address,
            contact_geom1=transport.inputs['geom1'],contact_geom2=transport.inputs['geom2'],
            contact_frame=transport.inputs['frame'],contact_pos=transport.inputs['pos'])
        post = RuntimePostAdapter(post_work,sources=post_sources,active=physics.n_admitted,first_error=first,
            d_valid=d.valid['smooth'],state_epoch=epoch,forward_epoch=d.forward_epoch,n_valid=n.valid,
            n_error=nw.err_code,n_written=n.n_written,n_attempt=n.n_attempt,n_expected=independent_forward,stream=stream)
        physics.before_constraint = post.capture_before_constraint
        physics.acceleration_sensor = lambda mask, st:post.enqueue(stream=st)
        physics.final_valid = post.valid
        physics.scope_gaps.remove('acceleration_sensor_hook')
        old_validate = physics.validate
        def validate_all():
            old_validate()
            post.validate(stream)
        physics.validate = validate_all
        controller = None
        if with_controller:
            controller = RuntimeControllerAdapter(d.stages.control,active=intent,first_error=first,
                parents=d.controller_inputs(),forward_refresh=physics.enqueue,
                gather=d.gather_controller_inputs,live_bias=d.gather_live_bias,stream=stream)
            controller.initialize_ctrl_from_reset(intent,ctrl,stream=stream)
            physics.ctrl,physics.ctrl_valid = controller.public['ctrl'],controller.valid
            physics.scope_gaps.remove('robot_controller')
            component_validate = physics.validate
            def validate_control():
                component_validate()
                controller.validate(stream)
            physics.validate = validate_control
        e = EulerWorkspace(model,W,device=device)
        # Persistent state aliases: Euler commits are the next D inputs.
        e.qpos,e.qvel = d.qpos,d.qvel
        e.time = wp.array(state['time'],dtype=scalar.wp_real,device=device)
        e.qM,e.qfrc_smooth = qm,d.public['qfrc_smooth']
        e.qacc,e.qfrc_constraint = public['qacc'],public['qfrc_constraint']
        e.ctrl,e.qacc_warmstart = physics.ctrl,public['qacc_warmstart']
        e.sensordata = post.public['sensor']
        integration_tape = position_tape = None
        if tape_steps:
            from .tape import DeviceInputTape
            integration_tape = DeviceInputTape(dict(qpos=e.qpos,qvel=e.qvel,time=e.time,
                ctrl=e.ctrl,warm=e.qacc_warmstart,qfrc_applied=d.stages.qfrc_applied,
                xfrc_applied=d.stages.xfrc_applied,active=e.active,state_epoch=epoch,
                forward_epoch=d.forward_epoch,expected_state=expected_state),
                capacity=tape_steps,stream=stream,phase='actual Euler entry')
            position_tape = DeviceInputTape(dict(transport.inputs,qpos=d.qpos,qvel=d.qvel,
                cdof=pos.mass.cdof,subtree_com=pos.mass.subtree_com,ncon=transport.ncon,
                active=transport.active,state_epoch=epoch,forward_epoch=d.forward_epoch,
                expected_forward=independent_forward),capacity=3*tape_steps+3,
                stream=stream,phase='actual constraint position entry')
            physics.position_tape = physics._position_tape = position_tape
        loop = ResidentStepRuntime(euler=e,forward=physics,controller=controller,intent=intent,
            state_epoch=epoch,forward_valid=physics.valid,forward_written=physics.written,
            forward_error=physics.error,stream=stream,observe=lambda:None,
            validate_components=physics.validate,first_error=first,expected_state_epoch=expected_state,
            integration_tape=integration_tape)
    return SimpleNamespace(runtime=loop,dynamics=d,collision=c,constraints=s,newton=n,
                           foundation=f,expected_state=expected_state,initial_state=state,
                           post=post,controller=controller,stream=stream,model_pack=pack,
                           integration_tape=integration_tape,position_tape=position_tape)
