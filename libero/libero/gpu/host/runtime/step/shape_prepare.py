"""Exact raw-model diagnostic assembly; all dynamic forward operands are GPU-produced."""
import libero.gpu.precision as scalar
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import tempfile
import gc
import traceback
import os
import numpy as np
import warp as wp



from .generic_prepare import _cold_factories


def prepare_shape_physics(built, *, state, device='cuda:0', stream=None,
                            mode='diagnostic', with_controller=True, tape_steps=0,
                            model_spec=None, constraint_package=None, nworld=1, memory_budget=None,
                            capture_entry_witness=False):
    stream = wp.get_stream(device) if stream is None else stream
    try:
        return _prepare_shape_physics(built,state=state,device=device,stream=stream,
            mode=mode,with_controller=with_controller,tape_steps=tape_steps,
            model_spec=model_spec,constraint_package=constraint_package,nworld=nworld,memory_budget=memory_budget,
            capture_entry_witness=capture_entry_witness)
    except BaseException as error:
        # Synchronize pending allocations/copies before dropping failed local
        # arenas. The original exception/trace remains available to the runner.
        try:
            wp.synchronize_stream(stream)
        except BaseException as cleanup:
            error.partial_cleanup_error = str(cleanup)
        else:
            traceback.clear_frames(error.__traceback__)
            gc.collect()
        raise


def _prepare_shape_physics(built, *, state, device='cuda:0', stream=None,
                             mode='diagnostic', with_controller=True, tape_steps=0,
                             model_spec=None, constraint_package=None, nworld=1, memory_budget=None,
                             capture_entry_witness=False):
    if mode != 'diagnostic':
        raise ValueError('raw-model full-chain production acceptance is pending')
    model = built.native_model
    from .shape_schema import RuntimeShape
    runtime_shape = RuntimeShape.from_native_metadata(built.metadata)
    if model_spec is None or constraint_package is None:
        raise ValueError("exact native ModelSpec and generated S package required")
    if int(model.na) or int(model.nmocap):
        raise ValueError('activation/mocap state requires a separate live binding')
    from libero.gpu.host.runtime.dynamics.runtime_factory import prepare_from_built, prepare_post
    from libero.gpu.host.runtime.dynamics.runtime_adapter import RuntimeControllerAdapter
    from libero.gpu.host.runtime.foundation.runtime_adapter import FoundationForwardAdapter
    from libero.gpu.host.runtime.collision.runtime_adapter import plan_model_capacity
    from libero.gpu.model.collision_static import static_from_model
    collision_factory, newton_factory, copy_choice = _cold_factories()
    from libero.gpu.host.runtime.core.generation import model_spec_for_native_metadata
    from libero.gpu.reference.integration.euler_integrator.generic_workspace import GenericEulerWorkspace
    from .contact_transport import ContactTransport
    from .shape_physics import ShapePhysicsForward
    from .shape_counter import ShapeCountedResidentStepRuntime
    shared_spec = model_spec
    # New parameter generations use their actual freshly generated S identity;
    # the component factories independently recheck the native read sets.
    package = Path(constraint_package)
    if not package.is_dir():
        raise ValueError('actual constraint specialization package is absent')
    sys.path.insert(0, str(package.parent))
    identity = importlib.import_module(package.name+'.identity')
    sm = importlib.import_module(package.name+'.workspace')
    sa = importlib.import_module(package.name+'.runtime_adapter')
    if type(nworld) is not int or nworld < 1:
        raise ValueError("positive integer nworld required")
    W = nworld
    shapes = runtime_shape.state_shapes(W)
    if set(state) != set(shapes):
        raise ValueError('complete reset boundary fields required')
    for key, shape in shapes.items():
        if state[key].dtype != np.float64 or state[key].shape != shape:
            raise TypeError('raw reset field schema: '+key)
    stream = wp.get_stream(device) if stream is None else stream
    with wp.ScopedStream(stream):
        epoch = wp.zeros(W,dtype=wp.int64,device=device)
        expected_state = wp.zeros(W,dtype=wp.int64,device=device)
        first = wp.zeros(W,dtype=wp.int32,device=device)
        intent = wp.full(W,1,dtype=wp.int32,device=device)
        state_valid = wp.full(W,1,dtype=wp.int32,device=device)
        fmask = wp.zeros(W,dtype=wp.int32,device=device)
        from libero.gpu.host.runtime.core.product_paths import build_directory
        build = build_directory('physics')
        d, intake = prepare_from_built(built,W,directory=build/'d-intake',
            decoder_record=built.metadata['decoder_record'],state_epoch=epoch,first_error=first,
            stream=stream,mode=mode,device=device,spec=shared_spec)
        spec = intake['kwargs']['spec']
        actual_model = Path(intake['model_record']['path'])
        d.qpos.assign(state['qpos']);d.qvel.assign(state['qvel'])
        d.qfrc_applied.assign(state['qfrc_applied']);d.xfrc_applied.assign(state['xfrc_applied'])
        ctrl = wp.array(state['ctrl'],dtype=scalar.wp_real,device=device)
        forward_plan = wp.zeros(W,dtype=wp.int64,device=device)
        f = FoundationForwardAdapter(d,mask=fmask,expected_state_epoch=expected_state,
            state_valid=state_valid,nq=runtime_shape.nq,nv=runtime_shape.nv,source_records=intake['kwargs']['source_records'],stream=stream)
        pos = d.position
        cactive = wp.zeros(W,dtype=wp.int32,device=device)
        owner = SimpleNamespace(xpos=pos.frames.geom_xpos,xmat=pos.frames.geom_xmat,
                                xipos=pos.mass.xipos,ximat=pos.mass.ximat,active=cactive)
        cstatic = static_from_model(model)
        cplan = plan_model_capacity(cstatic,W,device=device,memory_budget=memory_budget,
                                   metadata=built.metadata)
        C = cplan.contact_capacity
        R = C*6 + int(model.nv) + int(model.njnt)
        from libero.gpu.host.bindings.model_capacity import plan_native_model
        capacity_plan = plan_native_model(built.metadata, nworld=W, rows=R, contacts=C)
        # Stable compiler paths permit cross-process reuse while preserving
        # exact native source/argv validation and fresh runtime owner bindings.
        from libero.gpu.host.runtime.core.product_paths import current_build_project
        project = current_build_project()
        collision_build = project['root'] / 'collision' if project is not None else build
        native_root = collision_build/'c-native'
        native_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        previous_native_root = os.environ.get('P06C_NATIVE_BUILD_ROOT')
        os.environ['P06C_NATIVE_BUILD_ROOT'] = str(native_root)
        try:
            c = collision_factory(cstatic,cplan,source_owner=owner,
                source_paths={k:k for k in ('xpos','xmat','xipos','ximat','active')},
                stream=stream,device=device,capsule_source_root=collision_build/'c-capsule')
        finally:
            if previous_native_root is None:
                os.environ.pop('P06C_NATIVE_BUILD_ROOT',None)
            else:
                os.environ['P06C_NATIVE_BUILD_ROOT'] = previous_native_root
        transport = ContactTransport(c,d,nworld=W,capacity=C,stream=stream)
        sstatic = {key:np.ascontiguousarray(getattr(model,key),dtype=dtype)
                   for key,(shape,dtype) in identity.STATIC_SCHEMA.items()}
        sw = sm.PositionReferenceWorkspace(sstatic,W,R,C,model_facts=dict(identity.MODEL_FACTS),device=device)
        cdof, subtree = pos.mass.cdof.flatten(), pos.mass.subtree_com.flatten()
        s = sa.ResidentConstraintRuntime(sw,stream,
            lambda:dict(transport.inputs,active=transport.active,ncon=transport.ncon,
                        qpos=d.qpos,cdof=cdof,subtree_com=subtree),
            lambda:dict(qvel=d.qvel,qvel_epoch=d.written['velocity'],expected_qvel_epoch=forward_plan))
        qm = pos.mass.mass.qm.reshape((W,runtime_shape.nM))
        warm = wp.array(state['qacc_warmstart'],dtype=scalar.wp_real,device=device)
        n = newton_factory(model_path=actual_model,model_spec=spec,rows=R,contacts=C,nworld=W,
            build_root=build/'n-native',mode=mode,device=device,
            live_inputs=dict(J=sw.efc_J.reshape((W,R,runtime_shape.nv)),aref=sw.efc_aref.reshape((W,R)),
                D=sw.efc_D.reshape((W,R)),R=sw.efc_R.reshape((W,R)),loss=sw.efc_frictionloss.reshape((W,R)),
                efc_type=sw.efc_type.reshape((W,R)),efc_id=sw.efc_id.reshape((W,R)),qM=qm,
                qfrc_smooth=d.public['qfrc_smooth'],qacc_smooth=d.public['qacc_smooth'],qacc_warmstart=warm,
                contact_dim=transport.inputs['dim'].reshape((W,C)),contact_mu=sw.contact_mu.reshape((W,C)),
                contact_friction=transport.inputs['friction'].reshape((W,C,5))))
        physics = ShapePhysicsForward(runtime_shape=runtime_shape,dynamics=d,collision=c,transport=transport,constraints=s,
            newton=n,foundation=f,expected_forward=forward_plan,ctrl=ctrl,ctrl_valid=state_valid,stream=stream,
            capture_entry_witness=capture_entry_witness)
        from .post_contact_address import PostContactAddressBridge, capture_post_with_current_address
        address_bridge = PostContactAddressBridge(sw,active=physics.n_admitted,stream=stream)
        public, nw = n.public_arrays, n.ws
        sources = dict(qvel=d.qvel,qacc=public['qacc'],cinert=pos.mass.cinert,cvel=d.cvel,
            cdof=pos.mass.cdof,cdof_dot=d.cdof_dot,xipos=pos.mass.xipos,subtree_com=pos.mass.subtree_com,
            site_xpos=pos.frames.site_xpos,site_xmat=pos.frames.site_xmat,xfrc_applied=d.xfrc_applied,
            efc_force=public['efc_force'],ncon=transport.ncon,nefc=nw.nefc,ne=nw.ne,
            contact_dim=transport.inputs['dim'],contact_efc_address=address_bridge.current,
            contact_geom1=transport.inputs['geom1'],contact_geom2=transport.inputs['geom2'],
            contact_frame=transport.inputs['frame'],contact_pos=transport.inputs['pos'])
        post = prepare_post(intake,W,C,R,sources=sources,active=physics.n_admitted,first_error=first,
            d_valid=d.valid['smooth'],state_epoch=epoch,forward_epoch=d.forward_epoch,n_valid=n.valid,
            n_error=nw.err_code,n_written=n.n_written,n_attempt=n.n_attempt,
            n_expected=physics.expected_newton,stream=stream)
        post.contact_address_bridge = address_bridge
        physics.before_constraint = lambda st: capture_post_with_current_address(address_bridge,post,st)
        physics.acceleration_sensor = lambda mask,st:post.enqueue(stream=st)
        physics.final_valid = post.valid
        physics.scope_gaps.remove('acceleration_sensor_hook')
        original_validate = physics.validate
        def validate_all():
            original_validate();address_bridge.validate(stream);post.validate(stream)
        physics.validate = validate_all
        controller = None
        if with_controller:
            controller_type = RuntimeControllerAdapter
            controller_options = {}
            if getattr(d, 'joint_controller', False):
                from libero.gpu.host.runtime.control.joint import RuntimeJointControllerAdapter
                d.jacobian.install(d, build/'joint-inputs')
                controller_type = RuntimeJointControllerAdapter
                controller_options['build_root'] = build/'joint-controller'
            elif getattr(d.control, 'robot_projection', False):
                from libero.gpu.host.runtime.control.robot_osc import robot_osc_adapter_class
                controller_type = robot_osc_adapter_class(d.control)
                controller_options['build_root'] = build/'osc-robot-reset'
            elif getattr(d.control, 'general_goals', False):
                from libero.gpu.host.runtime.control.general_osc import RuntimeGeneralOSCControllerAdapter
                controller_type = RuntimeGeneralOSCControllerAdapter
            elif getattr(d.control, 'variable_mode', 0):
                from libero.gpu.host.runtime.control.variable_osc import RuntimeConfiguredOSCControllerAdapter
                controller_type = RuntimeConfiguredOSCControllerAdapter
            elif getattr(d.control, 'position_only', False):
                from libero.gpu.host.runtime.control.position_osc import RuntimePositionOSCControllerAdapter
                controller_type = RuntimePositionOSCControllerAdapter
            controller = controller_type(d.control,active=intent,first_error=first,
                parents=d.controller_inputs(),forward_refresh=physics.enqueue,
                gather=d.gather_controller_inputs,live_bias=d.gather_live_bias,stream=stream,
                **controller_options)
            controller.initialize_ctrl_from_reset(intent,ctrl,stream=stream)
            physics.ctrl,physics.ctrl_valid = controller.public['ctrl'],controller.valid
            physics.scope_gaps.remove('robot_controller')
        e = GenericEulerWorkspace(model,W,device=device,mode=mode)
        e.qpos,e.qvel = d.qpos,d.qvel
        e.time = wp.array(state['time'],dtype=scalar.wp_real,device=device)
        e.qM,e.qfrc_smooth = qm,d.public['qfrc_smooth']
        e.qacc,e.qfrc_constraint = public['qacc'],public['qfrc_constraint']
        e.ctrl,e.qacc_warmstart,e.sensordata = physics.ctrl,public['qacc_warmstart'],post.public['sensor']
        loop = ShapeCountedResidentStepRuntime(runtime_shape=runtime_shape,euler=e,forward=physics,controller=controller,intent=intent,
            state_epoch=epoch,forward_valid=physics.valid,forward_written=physics.written,forward_error=physics.error,
            stream=stream,observe=lambda:None,validate_components=physics.validate,
            first_error=first,expected_state_epoch=expected_state,
            control_timestep=1.0/float(built.metadata['control_freq']),
            outer_model_timestep=float(built.metadata.get('scheduler_model_timestep', model.opt.timestep)))
    bundle = SimpleNamespace(runtime=loop,dynamics=d,collision=c,constraints=s,newton=n,foundation=f,
        expected_state=expected_state,initial_state=state,post=post,controller=controller,stream=stream,
        model_pack=SimpleNamespace(model=model,mjb_path=actual_model),model_spec=spec,intake=intake,
        integration_tape=None,position_tape=None,reset_ctrl=ctrl,mode=mode,
        cold_copy_choice=copy_choice,runtime_shape=runtime_shape,
        memory_budget=cplan.memory_budget,collision_capacity_plan=cplan)
    def close():
        if getattr(bundle,"closed",False):return
        wp.synchronize_stream(stream)
        # Drop this assembly's arena roots. Facade/task owners must also drop
        # their borrowed fields and callbacks before allocator release is claimed.
        for key in tuple(vars(bundle)):
            if key not in ("closed","close"):
                setattr(bundle,key,None)
        bundle.closed = True
    bundle.close = close
    # A cold static-owner publication, after complete source/binding validation;
    # it is independent from derived forward/N numerical publication.
    physics.validate()
    from .environment import _model_commit
    with wp.ScopedStream(stream):
        bundle.expected_model_epoch=wp.zeros(W,dtype=wp.int64,device=device)
        bundle.device_model_epoch=wp.zeros(W,dtype=wp.int64,device=device)
        bundle.model_ready=wp.zeros(W,dtype=wp.int32,device=device)
    wp.launch(_model_commit,dim=W,inputs=[bundle.expected_model_epoch,bundle.device_model_epoch,
        bundle.model_ready,physics.valid,loop.valid],device=device,stream=stream)
    return bundle
