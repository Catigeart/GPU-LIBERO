"""Cold capacity selection for handwritten Newton and device POTRF.

Arithmetic admission is not a claim that every task has been GPU tested.
Resources depending on the compiled descriptor must still pass compilation
and cudaFuncGetAttributes at installation, rather than a fixed NV whitelist.
"""
import libero.gpu.precision as scalar
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from libero.gpu.host.runtime.core.product_paths import nvcc_path

from libero.gpu.host.runtime.core.resource_budget import (
    PORTABLE_SHARED_LIMIT_BYTES, newton_shared_budget)

SHARED_LIMIT = PORTABLE_SHARED_LIMIT_BYTES
DEFAULT_INDEX = Path('/srv/libero-split-receive-20261004-r01/alltasks-evidence/20261009-r01/newton/allmodels-native-r01/index.json')



LAYOUT_VERSION = 1
PRIVATE_ALIGNMENT_BYTES = 32


def align_bytes(size, alignment=PRIVATE_ALIGNMENT_BYTES):
    """Cold physical storage rounding; sizes and offsets are in bytes."""
    size, alignment = int(size), int(alignment)
    if size < 0 or alignment < 1 or alignment & (alignment - 1):
        raise ValueError('nonnegative size and power-of-two alignment required')
    return (size + alignment - 1) // alignment * alignment


def array_layout(value, *, world_stride_elements, owner_name, shared_offset_bytes=None):
    """Describe the actual owner, never infer padded storage from logical shape.

    Dense public owners keep their physical strides. A resident private view uses
    a separately aligned shared offset; its data length remains the logical world
    length. Epoch/admission belongs to the existing sealed owner and launch.
    """
    itemsize = int(value.strides[-1])
    stride = None if world_stride_elements is None else int(world_stride_elements)
    return dict(layout_version=LAYOUT_VERSION, owner=owner_name,
                logical_shape=tuple(int(x) for x in value.shape),
                physical_strides_bytes=tuple(int(x) for x in value.strides),
                itemsize_bytes=itemsize, world_pitch_elements=stride,
                world_pitch_bytes=None if stride is None else stride * itemsize,
                field_offset_bytes=0, alignment_bytes=PRIVATE_ALIGNMENT_BYTES if shared_offset_bytes is not None else itemsize,
                shared_offset_bytes=shared_offset_bytes, allocation='shared-resident' if shared_offset_bytes is not None else 'existing-owner',
                lifetime='invocation-prefix-loop-tail' if shared_offset_bytes is not None else 'sealed-owner',
                epoch_policy='existing owner/adapter launch seal; cold rebuild on pointer/shape/stride change')


def qualify(nv, nM, *, rows=None, contacts=None, nworld=1, blas_backend='cublasdx',
            shared_limit_bytes=SHARED_LIMIT,newton_islands='off'):
    nv, nM, nworld = int(nv), int(nM), int(nworld)
    if nv < 1 or nM < nv or nworld < 1:
        raise ValueError('positive NV/world count and NM >= NV required')
    if rows is not None and int(rows) < 0:
        raise ValueError('negative row capacity')
    if contacts is not None and int(contacts) < 0:
        raise ValueError('negative contact capacity')
    budget = newton_shared_budget(nv, blas_backend=blas_backend, scalar_bytes=scalar.itemsize,
                                  shared_limit_bytes=shared_limit_bytes,newton_islands=newton_islands)
    matrix = budget['potrf_matrix_bytes']
    assembly = budget['hessian_workspace_bytes']
    blas_workspace = budget['blas_workspace_bytes']
    workspace = budget['dx_workspace_bytes']
    reserve = budget['static_reserve_bytes']
    tile_k = budget['hessian_tile_k']
    fits = budget['admitted_by_formula']
    return dict(nv=nv, nM=nM, rows=rows, contacts=contacts, nworld=nworld,
                route='handwritten-cusolverdx' if fits else 'unsupported-single-block-potrf',
                admitted_by_formula=fits, gpu_tested=False,
                potrf_matrix_bytes=matrix, shared_limit_bytes=budget['shared_limit_bytes'],
                resource_ledger_version=budget['ledger_version'],
                blas_backend=blas_backend,newton_islands=newton_islands,
                hessian_tile_k=tile_k if blas_backend=='cublasdx' else None,
                hessian_workspace_bytes=assembly,blas_workspace_bytes=blas_workspace,dx_workspace_bytes=workspace,
                static_reserve_bytes=reserve,
                hot_state_budget_bytes=budget['hot_state_budget_bytes'],
                admission='compile descriptor and inspect actual CUDA resources before launch',
                fallback='source regularized Cholesky only on failed POTRF or tiny pivot',
                limitation=None if fits else 'device-library workspace exceeds the current shared-memory ABI')


def registry(index=DEFAULT_INDEX):
    records = json.loads(Path(index).read_text())
    tasks, dimensions = [], {}
    for row in records:
        result = row.get('result', {})
        entry = qualify(row['nv'], row['nM'], rows=result.get('rows'), contacts=result.get('contacts'))
        entry.update(task=Path(row['task']).name, model_path=str(Path(row['task'])/'model.mjb'),
                     model_sha256=result.get('mjb_sha256'), native_reference=row.get('reference'),
                     native_iterations=result.get('newton_iterations'), reference_exit_code=row.get('exit_code'))
        tasks.append(entry)
        key = (entry['nv'], entry['nM'])
        dim = dimensions.setdefault(key, dict(nv=key[0], nM=key[1], tasks=[], models=set(),
                                              qualification=qualify(*key), representative=entry))
        dim['tasks'].append(entry['task']); dim['models'].add(entry['model_sha256'])
    dimensions = [dict(value, models=sorted(value['models'])) for _,value in sorted(dimensions.items())]
    return dict(task_count=len(tasks), unique_model_count=len({r['model_sha256'] for r in tasks}),
                dimension_count=len(dimensions), max_nv=max((r['nv'] for r in tasks),default=0),
                formula_fit_count=sum(r['admitted_by_formula'] for r in tasks),
                gpu_verified_count=0, tasks=tasks, dimensions=dimensions,
                scope='cold reference inventory; no task execution or GPU correctness claim')


def apply_gpu_report(inventory, report_path):
    """Attach bounded representative proofs without promoting untested siblings."""
    path=Path(report_path);report=json.loads(path.read_text())
    if not report.get('completed') or not report.get('passed'):
        raise ValueError('completed passing Newton report required')
    tested={}
    for case in report['cases']:
        if case['mode']!='normal':continue
        key=(case['nv'],case['nM'])
        task=Path(case['fixture_task']).name if 'fixture_task' in case else next(t['task'] for t in inventory['tasks'] if (t['nv'],t['nM'])==key)
        tested[key]=dict(task=task,resources=case['resources'],iterations=case['iterations'],
                         potrf_diagnostic=case['potrf_diagnostic'],blas_diagnostic=case['blas_diagnostic'])
    for task in inventory['tasks']:
        key=(task['nv'],task['nM'])
        if key in tested and task['task']==tested[key]['task']:
            task.update(gpu_tested=True,gpu_evidence=str(path))
    for dimension in inventory['dimensions']:
        key=(dimension['nv'],dimension['nM'])
        if key in tested:dimension['bounded_representative_gpu_test']=tested[key]
    inventory.update(gpu_verified_count=sum(t['gpu_tested'] for t in inventory['tasks']),
                     gpu_verified_dimension_count=len(tested),newton_gpu_report=str(path),
                     scope='cold all-task inventory plus bounded representative fixtures; untested siblings remain untested')
    return inventory


def _resolve_head_metadata(bundle, profile, combine_entry, config, explicit, *, environ=None):
    """Cold opt-in only; a sealed descriptor still requires N's late gate."""
    env = os.environ if environ is None else environ
    enabled = int(env.get('LIBERO_CUDA_S_ORDINAL_METADATA', '0'))
    if enabled not in (0, 1):
        raise ValueError('S ordinal metadata must be 0 or 1')
    if explicit is None and not enabled:
        return None
    ws = bundle.newton.solver.ws
    if (profile != 'fast' or ws.nworld != 8 or not combine_entry or config.strategy != 'single_cta'):
        raise ValueError('S ordinal metadata requires fast N8 single_cta combined N entry')
    from libero.gpu.host.runtime.newton.head_metadata import HeadMetadata
    if explicit is None:
        from libero.gpu.host.bindings.constraints import newton_head_metadata
        explicit = newton_head_metadata(bundle.runtime.forward)
    if type(explicit) is not HeadMetadata:
        raise TypeError('sealed HeadMetadata descriptor required')
    explicit.validate()
    if explicit.owner is not bundle.runtime.forward:
        raise ValueError('S ordinal metadata belongs to another forward owner')
    if (explicit.nworld, explicit.rows, explicit.contacts, explicit.nv) != (
            ws.nworld, ws.rows, ws.contacts, ws.qacc.shape[1]):
        raise ValueError('S ordinal metadata differs from N dimensions')
    return explicit


def install(bundle, root, *, mathdx_root=None,dataflow_profile="compatible",combine_entry=False,external_prefix=None,external_tail=None,newton_execution_config=None,newton_algorithm_options=None,head_metadata=None):
    """Install the parameterized existing handwritten CUDA core into a bundle."""
    from libero.gpu.host.runtime.newton.handwritten_cuda import HandwrittenCudaCore
    from libero.gpu.host.runtime.core.execution_config import newton_execution_config as resolve_newton_config
    from libero.gpu.host.runtime.newton.algorithm_options import normalize_newton_algorithm_options,effective_newton_algorithm_configuration
    config=resolve_newton_config(execution_config=newton_execution_config)
    algorithm_options=normalize_newton_algorithm_options(newton_algorithm_options)
    ws=bundle.newton.solver.ws
    head_metadata=_resolve_head_metadata(bundle,dataflow_profile,combine_entry,config,head_metadata)
    expected_algorithm=effective_newton_algorithm_configuration(algorithm_options,ws.qacc.shape[1])
    admission=qualify(ws.qacc.shape[1],ws.qM.shape[1],rows=ws.rows,contacts=ws.contacts,nworld=ws.nworld,
                      newton_islands=algorithm_options['newton_islands'])
    if not admission['admitted_by_formula']:
        raise ValueError(admission)
    existing=getattr(bundle,'persistent_cuda_core',None)
    if existing is not None and existing.manifest.get('cholesky_backend')=='cusolverdx' and existing.manifest.get('diagnostic_profile','compatible')==dataflow_profile and not combine_entry and existing.block_size==config.block_threads and existing.manifest.get('hessian_algorithm')==expected_algorithm and bool(existing.manifest.get('ordinal_head_metadata',{}).get('requested',False))==(head_metadata is not None):
        bundle.cuda_newton_manifest=dict(existing.manifest,capacity_qualification=admission,resource_qualification=inspect_resources(existing))
        return existing
    solver=bundle.newton.solver
    installed_launch=solver.launch_device
    # A changed cold policy must record the retained stage launcher, not execute
    # the previously installed ctypes core during metadata recording. Restore
    # the old callable on any construction/compile failure; installation below
    # is the only point that replaces the active launcher.
    recording_launch=(existing.original if existing is not None and
                      getattr(existing,'solver',None) is solver else installed_launch)
    solver.launch_device=recording_launch
    try:
        core_root = Path(root) / ('nv%d_nm%d' % (ws.qacc.shape[1], ws.qM.shape[1]))
        core=HandwrittenCudaCore(solver,core_root,cholesky_backend='cusolverdx',mathdx_root=mathdx_root,
            diagnostic_profile=dataflow_profile,adapter=bundle.newton if combine_entry else None,external_prefix=external_prefix,external_tail=external_tail,execution_config=config,head_metadata=head_metadata,
            **algorithm_options)
    finally:
        solver.launch_device=installed_launch
    resources=inspect_resources(core)
    core.install()
    if combine_entry:
        from libero.gpu.host.assembly.pipeline.newton import install_cuda_entry
        install_cuda_entry(bundle.newton,core)
    bundle.persistent_cuda_core=core
    bundle.cuda_newton_manifest=dict(core.manifest,capacity_qualification=admission,resource_qualification=resources)
    return core


def inspect_resources(core, *, shared_limit_bytes=SHARED_LIMIT):
    """Inspect linked CUDA metadata before admission, including vendor static SMEM."""
    declared=core.manifest.get('library_file')
    path=(Path(declared) if declared is not None else
          Path(core.root)/(core.manifest['sha256']+'.so')).resolve()
    loaded_name=getattr(getattr(core,'library',None),'_name',None)
    if declared is not None and not loaded_name:
        raise ValueError('actual Newton private library owner missing')
    if loaded_name is not None and Path(loaded_name).resolve()!=path:
        raise ValueError('Newton resource image differs from loaded private owner')
    binary_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    expected_sha=core.manifest.get('library_sha256')
    if declared is not None and expected_sha is None:
        raise ValueError('actual Newton private library SHA missing')
    if expected_sha is not None and binary_sha!=expected_sha:
        raise ValueError('Newton private library hash mismatch before resource inspection')
    inspector=Path(nvcc_path()).parent/'cuobjdump'
    text=subprocess.check_output([str(inspector),'--dump-resource-usage',str(path)],text=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=binary_sha:
        raise ValueError('Newton private library changed during resource inspection')
    match=re.search(r'Function [^\n]*handwritten_newton[^\n]*:\s*REG:(\d+) STACK:(\d+) SHARED:(\d+)',text)
    if not match:raise ValueError('complete Newton CUDA resource record missing')
    registers,stack,static=(int(match[i]) for i in (1,2,3))
    dynamic=int(core.manifest['shared_bytes'])
    shared_limit_bytes=int(shared_limit_bytes)
    if shared_limit_bytes < 1:raise ValueError("positive shared launch limit required")
    if static+dynamic>shared_limit_bytes:
        raise ValueError(('static + dynamic shared memory exceeds launch limit',static,dynamic,shared_limit_bytes))
    return dict(registers_per_thread=registers,stack_bytes=stack,static_shared_bytes=static,
                dynamic_shared_bytes=dynamic,total_shared_bytes=static+dynamic,
                shared_limit_bytes=shared_limit_bytes,
                library_file=str(path),library_sha256=binary_sha,
                private_owner_matched=loaded_name is not None,
                passed=True,scope='linked complete Newton kernel resources, no performance claim')
