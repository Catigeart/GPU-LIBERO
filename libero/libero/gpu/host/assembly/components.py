"""Cold entry for handwritten CUDA fusion, with explicit component selection."""
from libero.gpu.host.paths import GPU_ROOT
import importlib
import os
from contextlib import nullcontext
from collections import deque
from pathlib import Path
import warp as wp
from libero.gpu.host.build.startup import startup_phase

COMPONENTS=('position','dynamics','osc','collision','collision_detect',
            'constraints','post','euler','recovery','copies','forward','capacity')


def _capture_reset_forward(loop):
    """Record a complete, explicitly begun forward using installed components."""
    with wp.ScopedCapture(stream=loop.stream) as capture:
        wp.capture_if(getattr(loop,'reset_predicate',loop.recovery.reset_mask),
                      on_true=loop._reset_forward,stream=loop.stream)
    loop._reset_forward_graph=capture.graph


def _plans(bundle):
    """Collect installed NativePlans without traversing model/library objects."""
    from libero.gpu.host.build.native import NativePlan
    seen=set();result=[];pending=deque([(bundle,0)])
    while pending:
        value,depth=pending.popleft()
        if id(value) in seen or depth>7:continue
        seen.add(id(value))
        if isinstance(value,NativePlan):result.append(value);continue
        if isinstance(value,dict):
            pending.extend((child,depth+1) for child in value.values())
        elif isinstance(value,(tuple,list)):
            pending.extend((child,depth+1) for child in value)
        elif hasattr(value,'__dict__') and type(value).__module__.startswith(('libero.','types')):
            for name,child in vars(value).items():
                if name not in ('registry','spec','metadata','native_model','built','robot'):
                    pending.append((child,depth+1))
    return tuple(result)


def prepare_cuda_physics(*args,cuda_build_root=None,mathdx_root=None,
                         components=None,dataflow_profile="compatible",profile=None,
                         optimized_dataflow=False,nworld=None,newton_execution_config=None,
                         algorithm_options=None,newton_algorithm_options=None,debug=False,
                         cache_collision_parameters=False,**kwargs):
    from libero.gpu.host.assembly.mathdx.factory import prepare_fused_physics as previous
    from libero.gpu.host.runtime.core.algorithm_config import normalize_algorithm_options
    algorithms=normalize_algorithm_options(algorithm_options,newton_algorithm_options=newton_algorithm_options)
    policy=algorithms.as_dict()
    if profile is not None:
        if dataflow_profile != "compatible" and dataflow_profile != profile:
            raise ValueError("conflicting dataflow_profile and profile")
        dataflow_profile=profile
    if dataflow_profile=="compatibility":dataflow_profile="compatible"
    if dataflow_profile not in ("compatible","fast"):raise ValueError("unknown dataflow profile")
    if nworld is not None:kwargs['nworld']=nworld
    kwargs.setdefault('capture_entry_witness', debug)
    selected=COMPONENTS if components is None else tuple(components)
    unknown=set(selected)-set(COMPONENTS)
    if unknown or len(set(selected))!=len(selected):
        raise ValueError('unknown or repeated CUDA components: '+repr(selected))
    root=Path(cuda_build_root or os.environ.get('LIBERO_CUDA_FUSION_BUILD_ROOT') or
              Path.cwd()/'handwritten-cuda-fusion-build').resolve()
    # Newton remains the accepted handwritten core; every full factor uses Dx.
    with startup_phase('arena'):
        bundle=previous(*args,cuda_build_root=root/'newton',mathdx_root=mathdx_root,
            diagnostic_profile=dataflow_profile,defer_cuda_core=('capacity' in selected),
            newton_execution_config=newton_execution_config,newton_algorithm_options=policy['newton'],**kwargs)
    from libero.gpu.host.build.native import world_guard_scope
    epoch_admitted=(optimized_dataflow and dataflow_profile=='fast' and not debug and cache_collision_parameters and
                    {'position','collision','collision_detect','constraints','forward'}<=set(selected))
    reuse_epoch=policy['forward_reuse'] in ('auto','epoch') and epoch_admitted
    if policy['forward_reuse']=='epoch' and not epoch_admitted:
        bundle.close()
        raise ValueError('epoch position reuse requires an optimized immutable-model native batch')
    prefix_guard=None
    if reuse_epoch:
        with wp.ScopedStream(bundle.stream):
            prefix_guard=wp.ones(bundle.runtime.forward.capacity,dtype=wp.int32,device=bundle.dynamics.device)
    bundle.cuda_position_reuse_guard=prefix_guard
    bundle.dataflow_profile=dataflow_profile
    bundle.debug=debug
    bundle.cuda_algorithm_options=algorithms
    try:
        manifests={}
        # Cold admission precedes F compilation. Only an admitted temporal C
        # front lets F omit the otherwise shared PCA bounds producer.
        if 'collision' in selected:
            from libero.gpu.host.bindings.collision import configure_temporal_front
            adapter=bundle.runtime.forward.c
            configure_temporal_front(getattr(adapter,'workspace',adapter),bundle.dynamics,
                dataflow_profile=dataflow_profile,**policy['collision'])
        for name in COMPONENTS:
            if name in selected:
                if name in ('osc', 'osc_gather') and getattr(bundle.dynamics, 'joint_controller', False):
                    manifests[name] = dict(installed=False, selected_controller='joint-native')
                    continue
                module=importlib.import_module('libero.gpu.host.bindings.'+name)
                options={}
                installer=module.install
                if name in ('dynamics','collision','constraints','post'):
                    options['dataflow_profile']=dataflow_profile
                if name=='collision':
                    options.update(policy['collision'])
                    options['diagnostics']=debug
                    options['static_parameters']=cache_collision_parameters
                    options['complete_only']=(optimized_dataflow and dataflow_profile=='fast' and
                        'collision_detect' in selected)
                if name=='post':options.update(policy['post'])
                if name=='collision_detect':options['diagnostic_profile']=dataflow_profile
                if name in ('position','osc') and optimized_dataflow:
                    installer=module.install_dataflow
                    options['dataflow_profile']=dataflow_profile
                    if name=='osc':options['diagnostic_profile']='compatibility' if dataflow_profile=='compatible' else 'fast'
                if name=='capacity':
                    options.update(mathdx_root=mathdx_root,dataflow_profile=dataflow_profile,
                        combine_entry=optimized_dataflow,newton_execution_config=newton_execution_config,
                        newton_algorithm_options=policy['newton'])
                    # S projection precedes retained before_constraint callbacks and
                    # optional entry witness copies. Keep this ordered boundary
                    # until a combined prefix owns both callback and copy semantics.
                if name=='capacity' and optimized_dataflow and 'post' in selected:
                    from libero.gpu.host.bindings.tail import prepare as prepare_tail
                    options['external_tail']=prepare_tail(bundle)
                migrated_prefix=False
                if name=='capacity' and optimized_dataflow and 'constraints' in selected and 'forward' in selected:
                    forward=bundle.runtime.forward
                    copy=forward._fusion_entry_copy
                    if forward.before_constraint is None and hasattr(copy,'pairs') and not copy.fallback and copy.slot is None:
                        from libero.gpu.host.bindings.constraints import projection_prefix
                        from libero.gpu.host.bindings.copies import entry_copy_kind
                        from libero.gpu.host.runtime.core.execution_config import newton_execution_config as resolve_copy_config
                        copy_threads=resolve_copy_config(execution_config=newton_execution_config).block_threads
                        schema=projection_prefix(forward)
                        schema['headers'].append((GPU_ROOT / 'cuda/runtime/entry_copy.cuh'))
                        for source,target in copy.pairs:
                            if source.size%forward.capacity:raise ValueError('entry witness world extent')
                            schema['stages'].append(dict(function='cf_entry_copy::'+entry_copy_kind(source,target,source.size//forward.capacity,copy_threads),
                                arguments=[('source',source),('target',target),('width',source.size//forward.capacity)],
                                collective=True,world_strides={'source':0,'target':0}))
                        options['external_prefix']=schema
                        migrated_prefix=True
                scope=(world_guard_scope(prefix_guard,('position_dataflow_','constraints_position'))
                       if reuse_epoch and name in ('position','constraints') else nullcontext())
                with startup_phase('bind_'+name),scope:
                    installed=installer(bundle,root/name,**options)
                if name=='capacity' and options.get('external_tail'):
                    from libero.gpu.host.bindings.tail import activate as activate_tail
                    manifests['newton_post_tail']=activate_tail(bundle,installed)
                if migrated_prefix:
                    from types import MethodType
                    forward._fusion_project_newton=MethodType(lambda owner:None,forward)
                    # Keep cold copy-root validation at the original call site;
                    # device copies execute after projection inside the N entry.
                    copy.launch=lambda stream:copy.validate_bindings()
                    forward._cuda_projection_in_newton=True
                    manifests['newton_entry_boundary']=dict(projection_and_entry_copy='combined',callback=None)
                elif name=='capacity' and optimized_dataflow:
                    manifests['newton_entry_boundary']=dict(projection_and_entry_copy='standalone',
                        reason='retained before_constraint callback or ordered alias fallback')
                manifests[name]=(installed if isinstance(installed,dict) else
                    dict(installed=True,manifest=installed.manifest))
        if 'collision' in selected and 'collision_detect' in selected:
            from libero.gpu.host.bindings.collision import install_combined
            schedule=policy['collision_schedule']
            if schedule=='auto':schedule='parallel' if bundle.runtime.forward.capacity<=32 else 'fused'
            scope=(world_guard_scope(prefix_guard,('collision_complete','collision_query_'))
                   if reuse_epoch else nullcontext())
            with scope:
                manifests['collision_complete']=install_combined(bundle,root/'collision_complete',
                    query_schedule=schedule,producer_blocks=policy['collision_workers'],world_guard=prefix_guard)
        # Capture after all component methods have been replaced. Capturing the
        # old bound functions would retain the old forward in the recovery arm.
        loop=bundle.runtime
        if optimized_dataflow:
            from libero.gpu.host.bindings.reset_preservation import install as install_reset_preservation
            manifests['reset_preservation']=install_reset_preservation(bundle,root/'reset_preservation')
        foundation=loop.forward.f
        # Validate retained owners after installing the selected components.
        foundation.validate_bindings()
        if reuse_epoch:
            # Establish the same explicit-begin cold contract as the exact
            # cache before capturing a prefix-only conditional body.
            _capture_reset_forward(loop)
            from libero.gpu.host.bindings.position_reuse import install as install_position_reuse
            manifests['reuse']=install_position_reuse(bundle,root/'position_reuse',prefix_guard)
        elif optimized_dataflow and 'forward' in selected and (debug or policy['forward_reuse']=='exact') and policy['forward_reuse']!='off':
            # Prefix-only cache graphs require their caller's explicit begin.
            # Record the complete native forward first, after native Newton is
            # installed, rather than relying on a legacy solver capture.
            _capture_reset_forward(loop)
            from libero.gpu.host.bindings.reuse import install_dataflow
            manifests['reuse']=install_dataflow(bundle,root/'reuse',diagnostic_profile=dataflow_profile)
        elif optimized_dataflow and 'forward' in selected:
            manifests['reuse']=dict(installed=False, reason='explicit off or no immutable-batch admission')
        foundation.validate_bindings()
        with startup_phase('reset_graph'):
            _capture_reset_forward(loop)
        bundle.fusion_batch='cuda-r02' if optimized_dataflow else 'cuda-r01'
        bundle.cuda_selection_manifest=manifests
        bundle.cuda_algorithm_manifest=dict(configuration=policy,
            default_policy='matches individual component installers',
            threshold_status='experimental configurable crossovers; not measured performance winners',
            components_with_policy=tuple(name for name in ('collision','post') if name in selected)+('newton',),
            numerical_qualification='independent functions only; no new full-program numerical acceptance claim')
        bundle.cuda_native_plans=_plans(bundle)
        bundle.cuda_manifests=[p.manifest for p in bundle.cuda_native_plans]
        bundle.cuda_manifests.append(bundle.persistent_cuda_core.manifest)
        detector=getattr(loop.forward.c.workspace.detect,'_cuda_detector_plan',None)
        if detector is not None:bundle.cuda_manifests.append(detector.manifest)
        def post_assembly_hook(components):
            from libero.gpu.host.bindings.position_scene import install_scene_from_environment
            install_scene_from_environment(components,root/'scene',dataflow_profile=dataflow_profile)
        if 'position' in selected:
            bundle.post_assembly_hook=post_assembly_hook
        bundle.cuda_requested_components=selected
        bundle.cuda_retained_components=tuple(k for k in COMPONENTS if k not in selected)
        def execution_trace():
            result={f"{index}:{p.manifest['name']}:{p.manifest['sha256'][:12]}":
                    p.execution_count.numpy().tolist()
                    for index,p in enumerate(bundle.cuda_native_plans)
                    if p.execution_count is not None}
            if detector is not None:result['collision_detect']=detector.execution_count.numpy().tolist()
            return result
        bundle.cuda_execution_trace=execution_trace
        return bundle
    except BaseException:
        bundle.close();raise


prepare_fused_physics=prepare_cuda_physics


def create(built):
    """Explicit environment route; production default remains separately named."""
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    result=assemble(built,physics_factory=prepare_cuda_physics,
        factory_entry='libero.gpu.host.assembly.components:create')
    from libero.gpu.host.bindings.position_scene import install_scene_from_environment
    try:
        install_scene_from_environment(result,Path.cwd()/'handwritten-cuda-fusion-build'/'scene')
    except BaseException:
        result.release()
        raise
    return result


def rebuild_components(built,previous):
    """Cold reconstruction selects CUDA again and reattaches scene callbacks."""
    from libero.gpu.host.runtime.step.shape_schema import RuntimeShape
    from libero.gpu.host.runtime.step import recovery_final_event_environment_factory as source
    RuntimeShape.from_native_metadata(built.metadata)
    source.default_integration_state(built.native_model)
    source.actual_decoder_record(built)
    previous.release()
    return create(built)


create.rebuild_components=rebuild_components
