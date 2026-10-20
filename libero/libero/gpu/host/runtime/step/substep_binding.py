"""Bind the named single-substep arguments to retained device owners.

The explicit paths below refer to the existing numerical adapters. Binding
copies their prepared argument values, not generated source or launch records.
No object-graph search, schema loading, CPU physics, or device readback is used.
"""
import libero.gpu.precision as scalar
import ctypes as ct

import warp as wp

from . import substep_types as t
from .newton_binding import prepare_newton
from .forward_plan import build_forward_mass_plan
from .controller_copy_plan import access_regions, build_packets, parallel_gather_allowed
from libero.gpu.host.build.substep_catalog import select_variant


# Numerical policies are fixed; model dimensions come from the selected
# offline instance and actual producer workspaces, never from task numbers.
DYNAMICS_DEFINES = dict(D_FAST_DATAFLOW="1")
SUPPORTED_DEFINES = {
    "CUDA_D_launch_velocity_group0": dict(DYNAMICS_DEFINES,D_NO_ACTUATION_DX="1"),
    "CUDA_D_launch_bias_group0": dict(DYNAMICS_DEFINES,D_NO_ACTUATION_DX="1"),
    "CUDA_D_launch_force_group0": DYNAMICS_DEFINES,
    "constraints_reference": dict(CF_S_COMPATIBLE="0",CF_S_KBIP_PLANES="1"),
    "constraints_projection": {},"osc_gather": {},"osc_live_bias": {},
    "osc_run_policy0": dict(OSC_UNCOUPLE="1"),
    "runtime_euler_shared": dict(CF_EULER_DIAGNOSTICS="0"),
}
# The position producer shares its compile defines with the reference plan.
# These flags select how J/KBIP are constructed, not the reference operation:
# commit_position always publishes dense J and interleaved public KBIP, which
# both reference implementations consume. In particular, the retained NV43
# producer enables compact/group metadata for N1/N2. Requiring its flags to
# match NV39 rejected an otherwise identical reference ABI before launch.
REFERENCE_PRODUCER_DEFINES = (
    "CF_S_BATCH_COMPACT", "CF_S_COMMON_ANCESTOR_ZERO",
    "CF_S_GROUP_PARAMETER_REUSE", "CF_S_INTERNAL_OWNER_RECEIPT",
    "CF_S_STATIC_CANDIDATES", "CF_S_UNIFIED_METADATA",
)
DIMENSION_DEFINES = {
    **{name:('D_NM','D_NU','D_NV','D_SM') for name in SUPPORTED_DEFINES if name.startswith('CUDA_D_')},
    'constraints_reference':('CF_S_DX_SM','NF_CAP','NJ','NL_CAP','NQ','NV','S_CONTACTS','S_NBODY','S_ROWS'),
    'osc_gather':('OSC_NV',),'osc_run_policy0':('DX_SM',),
    'runtime_euler_shared':('CF_EULER_NM','CF_EULER_NQ','CF_EULER_NV','CF_EULER_QUAT_STORAGE'),
}


def _check_variant(plan, dimensions):
    name=plan.manifest['name']
    expected=dict(SUPPORTED_DEFINES.get(name,{}))
    expected.update({key:str(dimensions[key]) for key in DIMENSION_DEFINES.get(name,())})
    actual=dict(flag[2:].split('=',1) for flag in plan.manifest['compile_commands'][0]
                if flag.startswith('-D') and '=' in flag)
    if name == "constraints_reference":
        for key in REFERENCE_PRODUCER_DEFINES:
            if actual.get(key) not in ("0", "1"):
                raise ValueError(f'substep producer variant differs: {name} {key}')
    for key,value in expected.items():
        if actual.get(key)!=value:
            raise ValueError(f'substep numerical variant differs: {name} {key}')


def _value(value):
    if isinstance(value, wp.array):
        return value.ptr
    return value.value if hasattr(value, "value") else value


def _pack(kind, values):
    if len(kind._fields_) != len(values):
        raise ValueError(f"{kind.__name__}: numerical argument count changed")
    return kind(*(_value(value) for value in values))


def _parallel_channel_flags(floating, integer):
    """Qualify copies once, including cross-field and float/int aliases.

    Any channel touching another source/destination retains lane-0 ordered
    copying. Independent channels can run concurrently in either direction.
    Spans include all worlds, so shifted cross-world aliases are also excluded.
    """
    groups = [(tuple((int(a.ptr), int(a.ptr) + a.size * scalar.itemsize) for a in pair))
              for pair in floating]
    groups += [tuple((int(a.ptr), int(a.ptr) + a.size * 4) for a in pair)
               for pair in integer]
    flags = []
    for index, spans in enumerate(groups[:len(floating)]):
        independent = max(spans[0][0], spans[1][0]) >= min(spans[0][1], spans[1][1])
        for other, candidates in enumerate(groups):
            if other != index and any(max(a, c) < min(b, d)
                                      for a, b in spans for c, d in candidates):
                independent = False
        flags.append(int(independent))
    return flags


def _one_dynamics_plan(dynamics, name):
    plans = dynamics._cuda_dynamics_contexts[name].plans
    if len(plans) != 1:
        raise ValueError(f"single-substep fusion requires one {name} numerical group")
    return plans[0][1]


def _bind_plan(arguments, plan, fields, worlds, dimensions):
    _check_variant(plan, dimensions)
    if plan.nworld != worlds or plan.work_tiles != 1:
        raise ValueError("world-local numerical layout required: " + plan.manifest["name"])
    if len(plan._stage_args) != len(fields):
        raise ValueError("numerical operation sequence changed: " + plan.manifest["name"])
    for (name, kind), values in zip(fields, plan._stage_args):
        setattr(arguments, name, _pack(kind, values))


def _validate_forward_reuse(arguments, controller, core, worlds):
    """Qualify cached payloads against later force/controller/Newton/Post writes."""
    a = arguments
    nv, nb, rows = a.state.nv, a.fluid_passive.nbody, a.reference_commit.rows
    if not (a.joint_passive.out == a.fluid_passive.qfrc == a.smooth_acceleration.passive
            and a.bias_commit.pub == a.smooth_acceleration.bias):
        raise ValueError("forward reuse requires retained passive/bias owners")
    cached = [
        (a.transmission.velocity, a.transmission.nu),
        (a.body_velocity.cvel, nb * 6), (a.body_velocity.cdot, nv * 6),
        (a.body_velocity.cache_acc, nb * 6), (a.body_velocity.cache_force, nb * 6),
        (a.joint_passive.out, nv), (a.bias_commit.pub, nv),
        (a.reference_commit.pv, rows), (a.reference_commit.pa, rows),
    ]
    cached_spans = [(int(ptr), int(ptr) + worlds * width * scalar.itemsize) for ptr, width in cached]
    # The hit path republishes epochs but retains these successful stage
    # statuses/readiness flags. They must also be private to their producers.
    for ptr in (a.transmission.status, a.body_velocity.status, a.joint_passive.status,
                a.velocity_status.out, a.bias_forces.status, a.reference_commit.ready,
                a.reference_commit.status, a.reference_commit.committed):
        cached_spans.append((int(ptr), int(ptr) + worlds * 4))
    writes = [(int(v.ptr), int(v.ptr) + v.size * wp.types.type_size_in_bytes(v.dtype))
              for v in vars(controller.controller).values() if isinstance(v, wp.array) and v.size]
    act, smooth = a.actuation, a.smooth_acceleration
    force_outputs = [(a.control_copy.target, a.control_copy.width),
                     (act.force, act.nu), (act.force_pre, act.nu), (act.ctrl_used, act.nu),
                     (act.qfrc_actuator, nv), (a.smooth_commit.pf, nv), (a.smooth_commit.pa, nv)]
    force_outputs += [(getattr(smooth, name), width) for name, width in
                      (("smooth", nv), ("seed", nv), ("scratch", nv), ("qfrc", nv),
                       ("ld", smooth.nm), ("inv", nv), ("sqrt_inv", nv),
                       ("solution", 4 * nv), ("rhs", 4 * nv), ("qacc_smooth", nv))]
    writes.extend((int(ptr), int(ptr) + worlds * width * scalar.itemsize) for ptr, width in force_outputs)
    for ptr, width in [(act.status, 1), (smooth.status, 1), (smooth.d02_status, 1),
                       (smooth.solve_status, 1), (smooth.pivot, nv)]:
        writes.append((int(ptr), int(ptr) + worlds * width * 4))
    p = core.params
    for ptr, width in [(a.state.qacc, nv), (p.qacc, nv),
                       (p.canonical_qacc, nv), (p.canonical_qacc_warmstart, nv),
                       (p.canonical_qfrc_constraint, nv),
                       (p.external_r_acc, nb * 6), (p.external_r_ext, nb * 6),
                       (p.external_r_intr, nb * 6),
                       (p.external_r_decoded, a.contact_addresses.capacity * 6),
                       (p.external_s_result, p.scalar_245),
                       (p.external_c_pe, p.scalar_244), (p.external_c_ps, p.scalar_245)]:
        if ptr:
            writes.append((int(ptr), int(ptr) + worlds * width * scalar.itemsize))
    # Newton scratch keeps its prepared allocation extents, rather than a
    # duplicate list of row/cone pitches here. Read-only J/aref/M are excluded.
    scratch = {int(value) for name in
               ("efc_b", "jar_warm", "D", "force", "jar", "oldstate", "Hcone", "Jv",
                "err_code", "state", "external_r_status", "external_s_status")
               if (value := getattr(p, name))}
    for owner in core.owners:
        if int(owner.ptr) in scratch:
            writes.append((int(owner.ptr), int(owner.ptr) +
                           owner.size * wp.types.type_size_in_bytes(owner.dtype)))
    if any(max(left, start) < min(right, end)
           for left, right in cached_spans for start, end in writes):
        raise ValueError("forward cache overlaps a force/controller/Newton/Post output")


def _validate_smooth_reuse(arguments, core, worlds):
    """Cold proof for the writer-free interval between the first two forwards."""
    a=arguments;p=core.params
    nv,nb=a.state.nv,a.fluid_passive.nbody
    if a.smooth_commit.width != nv or a.smooth_acceleration.nv != nv:
        raise ValueError('compact world-major smooth payload required')
    cache=[(a.smooth_commit.pf,nv),(a.smooth_commit.pa,nv),
           (a.smooth_acceleration.applied,nv),(a.smooth_acceleration.xfrc,6*nb)]
    cache=[(int(ptr),int(ptr)+scalar.itemsize*worlds*width) for ptr,width in cache]
    smooth=a.smooth_acceleration
    first_outputs=[(a.smooth_commit.pf,nv),(a.smooth_commit.pa,nv)]
    first_outputs += [(getattr(smooth,name),width) for name,width in
        (("smooth",nv),("seed",nv),("scratch",nv),("qfrc",nv),("ld",smooth.nm),
         ("inv",nv),("sqrt_inv",nv),("solution",4*nv),("rhs",4*nv),("qacc_smooth",nv))]
    first_spans=[(int(ptr),int(ptr)+scalar.itemsize*worlds*width) for ptr,width in first_outputs if ptr]
    if any(max(l,x)<min(r,y) for l,r in cache[2:] for x,y in first_spans):
        raise ValueError('smooth output overlaps a persistent applied-force input')
    if max(cache[0][0],cache[1][0]) < min(cache[0][1],cache[1][1]):
        raise ValueError('smooth force and acceleration publications overlap')
    act=a.actuation
    outputs=[(a.control_copy.target,a.control_copy.width),(act.force,act.nu),
             (act.force_pre,act.nu),(act.ctrl_used,act.nu),(act.qfrc_actuator,nv),
             (a.state.qacc,nv),(p.qacc,nv),(p.canonical_qacc,nv),
             (p.canonical_qacc_warmstart,nv),(p.canonical_qfrc_constraint,nv),
             (p.external_r_acc,nb*6),(p.external_r_ext,nb*6),(p.external_r_intr,nb*6),
             (p.external_r_decoded,a.contact_addresses.capacity*6),
             (p.external_s_result,p.scalar_245),(p.external_c_pe,p.scalar_244),
             (p.external_c_pi,p.scalar_244),(p.external_c_ps,p.scalar_245)]
    writes=[(int(ptr),int(ptr)+scalar.itemsize*worlds*width) for ptr,width in outputs if ptr]
    scratch={int(value) for name in ('efc_b','jar_warm','D','force','jar','oldstate','Hcone','Jv',
             'err_code','state','external_r_status','external_s_status') if (value:=getattr(p,name))}
    for owner in core.owners:
        if int(owner.ptr) in scratch:
            writes.append((int(owner.ptr),int(owner.ptr)+owner.size*wp.types.type_size_in_bytes(owner.dtype)))
    if any(max(l,x)<min(r,y) for l,r in cache for x,y in writes):
        raise ValueError('cached smooth/input overlaps an intervening writer')
    return dict(passed=True,cached_spans=len(cache),writer_spans=len(writes),
                lifetime='Start:first successful forward -> same CTA second forward; no controller or external callbacks before hit; position suspension drops the local flag')


def bind(runtime, continuation, *, single_solve=False):
    """Return an owned argument aggregate and the numerical holders it borrows."""
    forward = runtime.forward
    dynamics = forward.d
    controller = runtime.controller
    recovery = runtime.recovery
    cache = forward._dataflow_reuse
    euler = runtime.euler
    worlds = euler.capacity
    workspace = forward.s.workspace
    variant = select_variant(euler.nv,euler.nM,nq=euler.nq,njnt=euler.njnt,
        nu=euler.ctrl.shape[1],rows=forward.n.ws.rows,contacts=forward.n.ws.contacts)
    sm = runtime.stream.device.arch * 10
    dimensions = dict(D_NV=variant.nv,D_NM=variant.nM,D_NU=variant.nu,D_SM=sm,
        NV=variant.nv,NQ=variant.nq,NJ=variant.njnt,S_NBODY=workspace.model_facts['nbody'],
        NF_CAP=workspace.s02.friction_pos.size//worlds,NL_CAP=workspace.s02.limit_pos.size//worlds,
        S_ROWS=variant.rows,S_CONTACTS=variant.contacts,CF_S_DX_SM=sm,OSC_NV=variant.nv,
        CF_EULER_NV=variant.nv,CF_EULER_NQ=variant.nq,CF_EULER_NM=variant.nM,
        CF_EULER_QUAT_STORAGE=variant.quaternion_joints,DX_SM=sm)
    if not getattr(cache, "position_only", False):
        raise ValueError("single-substep fusion requires epoch-based position reuse")
    if controller.forward_refresh.__self__ is not forward:
        raise ValueError("controller refresh must use this physical forward")
    if controller._world_alias_fallback:
        raise ValueError("world-local controller storage required")
    if any(getattr(controller.controller, feature, False) for feature in
           ("general_goals", "robot_projection", "position_only", "variable_mode")):
        raise ValueError("this numerical implementation requires the fixed 7+2 OSC layout")
    if (forward.position_tape is not None or forward.scene_transform is not None or
            forward._fusion_entry_copy is not None or runtime.integration_tape is not None or
            getattr(runtime, "capture_events", False)):
        raise ValueError("single-substep source does not include optional diagnostic tapes/callbacks")
    if worlds > 1 and getattr(runtime, "_reset_preserver", None) is None:
        raise ValueError("mixed-world prefix/reset continuation requires the prepared preservation owner")

    core = forward.n.cuda_entry_core
    if not core.manifest.get("external_tail_stages", 0):
        raise ValueError("complete Newton/Post numerical tail required")

    arguments = t.SubstepArguments()
    retained = [runtime, core, continuation]
    velocity = _one_dynamics_plan(dynamics, "launch_velocity")
    bias = _one_dynamics_plan(dynamics, "launch_bias")
    force = _one_dynamics_plan(dynamics, "launch_force")
    _bind_plan(arguments, velocity, [
        ("velocity_gate", t.VelocityGateData), ("transmission", t.TransmissionData),
        ("body_velocity", t.BodyVelocityData), ("joint_passive", t.JointPassiveData),
        ("fluid_passive", t.FluidPassiveData), ("velocity_status", t.VelocityStatusData),
        ("velocity_finish", t.VelocityFinishData)], worlds, dimensions)
    _bind_plan(arguments, forward.s._cuda_reference_plan, [
        ("reference_gate", t.ReferenceGateData),
        ("constraint_reference", t.ConstraintReferenceData),
        ("reference_commit", t.ReferenceCommitData)], worlds, dimensions)
    _bind_plan(arguments, bias, [
        ("bias_gate", t.BiasGateData), ("bias_reference", t.BiasReferenceData),
        ("bias_forces", t.BiasForcesData), ("bias_commit", t.BiasCommitData)], worlds, dimensions)
    _bind_plan(arguments, force, [
        ("actuation_gate", t.ActuationGateData), ("control_gate", t.ControlGateData),
        ("control_copy", t.ControlCopyData), ("actuation", t.ActuationData),
        ("actuation_finish", t.ActuationFinishData), ("smooth_gate", t.SmoothGateData),
        ("smooth_force_gate", t.SmoothForceGateData),
        ("smooth_acceleration", t.SmoothAccelerationData),
        ("smooth_commit", t.SmoothCommitData)], worlds, dimensions)
    _bind_plan(arguments, forward._cuda_s_projection, [
        ("newton_admission", t.NewtonAdmissionData),
        ("constraint_counts", t.ConstraintCountsData),
        ("cone_parameters", t.ConeParametersData), ("newton_epoch", t.NewtonEpochData)], worlds, dimensions)
    _bind_plan(arguments, dynamics._cuda_osc_gather_plan, [
        ("gather_gate", t.GatherGateData), ("end_effector_pose", t.EndEffectorPoseData),
        ("controller_inputs", t.ControllerInputsData),
        ("end_effector_velocity", t.EndEffectorVelocityData)], worlds, dimensions)
    _bind_plan(arguments, dynamics._cuda_osc_bias_plan,
               [("controller_bias", t.ControllerBiasData)], worlds, dimensions)
    _bind_plan(arguments, controller._cuda_osc_plans["planning"][0],
               [("controller_plan", t.ControllerPlanData)], worlds, dimensions)

    # Bias/control inputs are call-time arguments in the original adapters;
    # bind their actual forward owners instead of cold-construction defaults.
    workspace = forward.s.workspace
    arguments.bias_reference.done = workspace.reference_ready.ptr
    arguments.bias_reference.err = workspace.status.ptr
    arguments.bias_reference.written = workspace.ref_epoch.ptr
    arguments.bias_reference.expected = forward.expected_forward.ptr
    arguments.control_copy.source = forward.ctrl.ptr
    arguments.control_gate.valid = forward.ctrl_valid.ptr

    run = controller._run.plans[(("policy", 0),)]
    if len(run._stage_args) != 32:
        raise ValueError("fixed OSC gate/solve/commit/26-float/2-int/publication sequence required")
    for index, name, kind in [
            (0, "controller_gate", t.ControllerGateData),
            (1, "controller_solve", t.ControllerSolveData),
            (2, "controller_commit", t.ControllerCommitData),
            (31, "controller_publish", t.ControllerPublishData)]:
        setattr(arguments, name, _pack(kind, run._stage_args[index]))
    _check_variant(run, dimensions)
    _check_variant(runtime._cuda_euler_plan, dimensions)
    if len(runtime._cuda_euler_plan._stage_args) != 5:
        raise ValueError("Euler gate/evaluate/publish/expected/count sequence required")
    arguments.euler = _pack(t.EulerData, runtime._cuda_euler_plan._stage_args[1])

    foundation = forward.f
    if len(foundation.gates) != 1 or foundation.nworld != worlds:
        raise ValueError("one W-wide foundation gate required")
    gate = foundation.gates[0]
    count, actual, expected, valid = foundation._views[0]
    arguments.foundation_gate = _pack(t.FoundationGateData, [
        foundation._mask_views[0], count, actual, expected, valid, actual,
        gate.errors, gate.first_stage, gate.first_local, gate.first_item, gate.first_event,
        gate.active, gate.valid, gate.capacity, 101, foundation.admitted, foundation.errors,
        foundation.first_stage, foundation.first_local, foundation.first_item,
        foundation.first_event, 0])
    arguments.begin = _pack(t.BeginData, [
        foundation.admitted, dynamics.first_error, dynamics.state_epoch,
        dynamics.source_state_epoch, dynamics.forward_epoch, dynamics.active,
        *[dynamics.valid[key] for key in ("position", "velocity", "bias", "smooth")],
        forward.expected_forward])
    arguments.reference_admission = _pack(t.ReferenceAdmissionData, [
        dynamics.valid["velocity"], dynamics.written["velocity"], forward.expected_forward,
        dynamics.first_error, forward.transport.active])
    if single_solve:
        # step1 prepares these actual OSC dependencies without claiming that
        # actuation, Newton or the complete forward has already executed.
        arguments.controller_gate.pv = dynamics.valid["bias"].ptr
        arguments.controller_gate.written = dynamics.written["bias"].ptr
        arguments.controller_gate.expected = forward.expected_forward.ptr
    arguments.forward_publish = _pack(t.ForwardPublishData, [
        dynamics.active, forward.final_valid, forward.n.ws.err_code, dynamics.first_error,
        forward.projection_status, dynamics.source_state_epoch, dynamics.state_epoch,
        forward.valid, forward.written, forward.error])
    arguments.state = _pack(t.StateData, [
        runtime.intent, forward.mask, dynamics.requested_mask, runtime.first_error,
        runtime.state_epoch, runtime.expected_state_epoch, runtime.forward_valid,
        runtime.forward_written, runtime.forward_error, runtime.valid, runtime.written,
        runtime.integration_count, euler.qpos, euler.qvel, euler.qacc, euler.nq, euler.nv])
    arguments.position = _pack(t.PositionCacheData, [
        cache.saved, cache.cached, workspace.status, workspace.committed,
        forward.c.workspace.fe_code, dynamics.stages.position.factor.warning_count,
        forward.c.workspace.finalize.warnings, cache.work, cache.hit,
        dynamics.written["position"], cache.consumed, forward.transport.active,
        forward.c_mask])
    # The resident smooth bridge replaces loads from exactly these owners.
    # Qualify once while binding; CUDA performs no pointer/alias inventory scan.
    if (int(core.params.qacc_smooth) != int(arguments.smooth_commit.pa) or
            int(core.params.qfrc_smooth) != int(arguments.smooth_commit.pf)):
        raise ValueError("fused smooth/Newton inputs must share the published owners")
    newton_data,newton_owners=prepare_newton(core,forward.n.ws,worlds,runtime.stream,
                                           position_cache=cache)
    retained.extend(newton_owners)
    arguments.newton = _pack(t.NewtonData, [
        newton_data, forward.n_upstream, forward.mask, forward.projection_status,
        forward.expected_newton, forward.n.valid, forward.n.n_written, 0, None, None])

    fields = []
    for name, array in recovery.views.items():
        if array.dtype != scalar.wp_real:
            raise ValueError("single-substep reset inventory requires the selected scalar: " + name)
        fields.append(_pack(t.ResetField, [array, recovery.defaults[name], recovery.catalog[name]["width"]]))
    if len(fields) > 14:
        raise ValueError("native reset field inventory exceeds its declared extent")
    reset = arguments.recovery
    for name, array in dict(active=recovery.active, reset_mask=recovery.reset_mask,
            reason=recovery.reason, bad_index=recovery.bad_index, source_error=recovery.source_error,
            warning_number=recovery.warning_number, warning_lastinfo=recovery.warning_lastinfo,
            reset_events=runtime.reset_events).items():
        setattr(reset, name, array.ptr)
    reset.field_count = len(fields)
    for index, field in enumerate(fields):
        reset.fields[index] = field

    # The original fast OSC selector already requires private workspace arrays.
    # Reconfirm that proof once for this binding; CUDA skips repeated alias scans.
    c = controller.controller
    views = [(int(v.ptr), int(v.ptr)+v.size*wp.types.type_size_in_bytes(v.dtype))
             for v in vars(c).values() if isinstance(v, wp.array) and v.size]
    ordered = sorted(views)
    if any(right > next_left for (_, right), (next_left, _) in zip(ordered, ordered[1:])):
        raise ValueError("fused OSC requires the existing disjoint fast workspace")

    channels = arguments.controller_channels
    public = controller.public
    public_int = controller.public_int
    if len(public) != 26 or len(public_int) != 2:
        raise ValueError("fixed OSC publication fields required")
    floating_pairs = [(array, getattr(controller.controller, name)) for name, array in public.items()]
    integer_pairs = [(array, getattr(controller.controller, name)) for name, array in public_int.items()]
    # These outputs are written on every successful control, including cache
    # hits. Partial quaternion/nullspace diagnostics retain their old seed.
    overwritten = {"ctrl", "torques_raw", "torques_clipped", "mass_inv",
        "A_full", "A_pos", "A_ori", "lambda_full", "lambda_pos", "lambda_ori",
        "position_error", "orientation_error", "nullspace_torque"}
    parallel = _parallel_channel_flags(floating_pairs, integer_pairs)
    for index, (name, published) in enumerate(public.items()):
        channels.floating[index] = _pack(t.DoubleChannel,
            [published, getattr(controller.controller, name), published.shape[1], parallel[index] | (0 if name in overwritten else 2)])
    for index, (name, published) in enumerate(public_int.items()):
        channels.integer[index] = _pack(t.IntChannel, [published, getattr(controller.controller, name)])
    channels.floating_count = len(public)
    channels.integer_count = len(public_int)

    # Warp-zero gather and the other warps' seed copies overlap only after an
    # exact cold proof over all source, output and protocol owners.
    topology = getattr(dynamics.jacobian, "_cuda_osc_body_topology", None)
    index_owner = topology.arrays["body_dof_indices"] if topology is not None else None
    if not arguments.controller_inputs.body_dof_indices or index_owner is not None:
        index_words = index_owner.size if index_owner is not None else 0
        regions = access_regions(arguments, worlds, index_words=index_words)
        integer_regions = access_regions(arguments, worlds, index_words=index_words, skip_copy_flags=True)
        packets, seed_count, commit_count = build_packets(channels, worlds, regions, integer_regions)
        if not parallel_gather_allowed(arguments, worlds, index_words=index_words):
            packets = []
        if packets:
            packet_type = t.ControllerCopyPacket * len(packets)
            packet_bytes = bytes(packet_type(*packets))
            with wp.ScopedStream(runtime.stream):
                packet_owner = wp.array(list(packet_bytes), dtype=wp.uint8, device=runtime.stream.device)
            channels.copy_packets = packet_owner.ptr
            channels.seed_packets = seed_count
            channels.commit_packets = commit_count
            retained.append(packet_owner)

    observer = getattr(runtime.observe, "__self__", None)
    if observer is None:
        raise ValueError("prepared device observer required")
    post = observer.prepared.post
    bridge = post.contact_address_bridge
    source = bridge.sources
    arguments.contact_addresses = _pack(t.ContactAddressData, [
        bridge.active, source["status"], source["committed"], source["ncon"],
        source["contact_exclude"], source["contact_address"], bridge.current, bridge.contacts])
    copies = post._fusion_epoch_copy.pairs
    if len(copies) != 3:
        raise ValueError("three declared Post epoch copies required")
    for index, (source, target) in enumerate(copies):
        if (source.dtype != wp.int64 or target.dtype != wp.int64 or
                source.size != worlds or target.size != worlds):
            raise ValueError("one int64 Post epoch per world required")
        arguments.post_epochs[index] = _pack(t.EpochCopy, [source, target, source.size // worlds])
    arguments.continuation = continuation.ptr
    # This owner is admitted only for immutable model/mocap data and the fixed
    # native controller. A cache entry lasts at most one physical substep;
    # reset/state writes and even a same-state prefix rebuild invalidate it.
    smooth = dynamics.acceleration
    mass_values = build_forward_mass_plan(smooth.static['dof_parentid'],
                                         smooth.static['dof_Madr'], variant.nM)
    # These immutable owners are already admitted by the dynamics binding.
    # The position producer proves factor finiteness and publishes its epoch;
    # smooth must consume that exact factor, not an unrelated working array.
    factor = dynamics.position.factor
    for name in ('ld', 'inv', 'sqrt_inv', 'pivot', 'status', 'active',
                 'warning_count', 'warning_last'):
        if getattr(arguments.smooth_acceleration, 'source_' + name) != getattr(factor, name).ptr:
            raise ValueError('smooth position factor owner differs: ' + name)
    with wp.ScopedStream(runtime.stream):
        forward_metadata = wp.zeros((worlds, t.FORWARD_CACHE_WORDS),
                                    dtype=wp.uint64, device=runtime.stream.device)
        fluid_workspace = wp.empty((worlds, 9 * arguments.fluid_passive.nbody + euler.nv),
                                   dtype=scalar.wp_real, device=runtime.stream.device)
        mass_plan = wp.array(mass_values, dtype=wp.int32, device=runtime.stream.device)
    arguments.forward_workspace = _pack(t.ForwardWorkspaceData,
        [forward_metadata, cache.generation, fluid_workspace, mass_plan])
    retained.extend((forward_metadata, fluid_workspace, mass_plan))
    _validate_forward_reuse(arguments, controller, core, worlds)
    _validate_smooth_reuse(arguments, core, worlds)
    return arguments, retained, core
