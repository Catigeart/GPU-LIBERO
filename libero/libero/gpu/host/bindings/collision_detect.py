"""Direct handwritten CUDA collision detector, installed on live workspaces.

Independent detector: grid tiles produce all queries, a second kernel commits
the ordered successful prefix. Complete C retains the block collective helper.
All numerical bodies come from handwritten CUDA or frozen MuJoCo/libccd C,
never from Warp-generated C++. No dense BLAS/LAPACK operation occurs here.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import ctypes as ct
import types
import os
import numpy as np
import warp as wp
from libero.gpu.host.build.native import compile_library

MODULE = (GPU_ROOT / 'cuda/collision')
POINTER_NAMES = (
    'call_g1', 'call_g2', 'call_type1', 'call_type2', 'call_margin',
    'call_count', 'active', 'xpos', 'xmat', 'geom_size', 'geom_rbound',
    'geom_dataid', 'mesh_vert', 'mesh_vertadr', 'mesh_vertnum',
    'mesh_graphadr', 'mesh_graph', 'route_id', 'raw', 'raw_count',
    'det_status', 'scratch_i', 'native_raw', 'native_count', 'native_status',
    'native_diag', 'native_frame0', 'native_framewritten', 'fe_stage',
    'fe_code', 'fe_item', 'cuda_detector_execution_count',
    'primitive_raw', 'primitive_count', 'primitive_status',
    'query_order', 'query_tmp', 'bucket_offsets', 'task_bucket', 'task_start', 'task_count', 'mesh_edge_end', 'route_subgroup_widths')
INTEGER_NAMES = ('nworld', 'pairs', 'ngeom', 'nmesh', 'nvert', 'ngraph',
                 'mpr_iterations', 'multiccd')


class Params(ct.Structure):
    _fields_ = ([(n, ct.c_void_p) for n in POINTER_NAMES] +
                [(n, ct.c_int) for n in INTEGER_NAMES] +
                [('diagnostic_profile', ct.c_int), ('query_subgroup_width', ct.c_int),
                 ('box_private_capacity', ct.c_int), ('mpr_tolerance', scalar.ct_real)])


def source(*, device_only=False):
    """Inline maintained headers so every dependency invalidates the cache."""
    vector_source = (GPU_ROOT / 'cuda/common/vector.cuh').read_text()
    native = ((GPU_ROOT / 'cuda/collision/native/native_source.cuh')).read_text()
    support_reuse = (GPU_ROOT / 'cuda/collision/native/native_support_reuse.cuh').read_text()
    native = native.replace('#include "native_support_reuse.cuh"', support_reuse)
    primitive = ((GPU_ROOT / 'cuda/collision/collision_detect_primitive.cuh')).read_text()
    primitive = primitive.replace('#include "native/native_source.cuh"', '')
    adapter = ((GPU_ROOT / 'cuda/collision/collision_detect.cuh')).read_text().replace(
        '#include "collision_detect_primitive.cuh"', '')
    adapter = adapter.replace('#include "../common/warp.cuh"', '')
    warp_source = (GPU_ROOT / 'cuda/common/warp.cuh').read_text()
    kernel = ((GPU_ROOT / 'cuda/collision/collision_detect.cu')).read_text().replace(
        '#include "collision_detect.cuh"', '')
    if device_only:
        wrapper='''
namespace libero_collision_cuda {
__device__ void detect_prepare_pointer(int w, void* raw) { detect_prepare_compact(*(Params*)raw,w); }
__device__ void detect_commit_pointer(int w, void* raw) { detect_commit_world(*(Params*)raw,w); }
__device__ void detect_pointer(int w, void* raw) {
    detect_world(*(Params*)raw, w);
}
}
'''
        return '\n'.join((vector_source, native, warp_source, primitive, adapter, wrapper))
    return '\n'.join((vector_source, native, warp_source, primitive, adapter, kernel))


def device_source():
    """Header-only arithmetic and pointer stage for a collective C kernel."""
    return source(device_only=True)


class DetectorPlan:
    def __init__(self, workspace, build_dir, *, diagnostic_profile='compatibility', query_subgroup_width=32, box_private_capacity=50):
        self.workspace = workspace
        # The collision binding inventories every direct workspace array.
        # Keep instrumentation on this plan so installation preserves that ABI.
        self.execution_count=wp.zeros(workspace.nworld,dtype=wp.int64,device=workspace.device)
        if diagnostic_profile=='compatible':diagnostic_profile='compatibility'
        if diagnostic_profile not in ('compatibility', 'fast'):
            raise ValueError('unknown collision diagnostic profile')
        self.diagnostic_profile=diagnostic_profile
        if query_subgroup_width not in (0,32):
            raise ValueError('query_subgroup_width must be 0 or 32')
        self.query_subgroup_width=query_subgroup_width
        if isinstance(box_private_capacity,bool) or box_private_capacity!=50:
            raise ValueError('CG06 cold ABI admits original box capacity50 only')
        self.box_private_capacity=50
        self.route_subgroup_widths=wp.array(np.full(14,query_subgroup_width,dtype=np.int32),dtype=wp.int32,device=workspace.device)
        self.mesh_edge_end=workspace.mesh_edge_end
        self.primitive_raw=wp.empty(workspace.nworld*workspace.pairs*80,dtype=scalar.wp_real,device=workspace.device)
        self.primitive_count=wp.empty(workspace.nworld*workspace.pairs,dtype=wp.int32,device=workspace.device)
        self.primitive_status=wp.empty(workspace.nworld*workspace.pairs,dtype=wp.int32,device=workspace.device)
        self.query_order=wp.empty(workspace.nworld*workspace.pairs,dtype=wp.int32,device=workspace.device)
        self.query_tmp=wp.empty(workspace.nworld*workspace.pairs,dtype=wp.int32,device=workspace.device)
        self.bucket_offsets=wp.empty(workspace.nworld*16,dtype=wp.int32,device=workspace.device)
        self.task_capacity=(workspace.pairs+1)//2+15
        self.task_bucket=wp.empty(workspace.nworld*self.task_capacity,dtype=wp.int32,device=workspace.device)
        self.task_start=wp.empty(workspace.nworld*self.task_capacity,dtype=wp.int32,device=workspace.device)
        self.task_count=wp.empty(workspace.nworld,dtype=wp.int32,device=workspace.device)
        self.queue_offsets=wp.empty(4*(workspace.nworld+1),dtype=wp.int64,device=workspace.device)
        self.public_pointer_names=POINTER_NAMES[:-12]
        self.owners = tuple(getattr(workspace, n) for n in self.public_pointer_names)+(
            self.execution_count,self.primitive_raw,self.primitive_count,self.primitive_status,
            self.query_order,self.query_tmp,self.bucket_offsets,self.task_bucket,self.task_start,self.task_count,self.mesh_edge_end,self.route_subgroup_widths)
        self.device = wp.get_device(workspace.device)
        if any(not a.is_contiguous or a.device != self.device for a in self.owners):
            raise ValueError('detector requires contiguous arrays on one CUDA device')
        self.bindings = tuple((a.ptr, a.shape, a.dtype) for a in self.owners)
        if workspace.nworld <= 0 or workspace.pairs <= 0:
            raise ValueError('positive world and pair capacity required')
        self._mpr_tolerance = workspace.mpr_tolerance
        self.params = Params(*[a.ptr for a in self.owners],
                             *[getattr(workspace, n) for n in INTEGER_NAMES],
                             int(diagnostic_profile=='fast'),query_subgroup_width,box_private_capacity,
                             workspace.mpr_tolerance)
        self.library, self.manifest = compile_library('collision_detect', source(),
                build_dir, device=self.device, extra_flags=('-rdc=true',))
        self.library.collision_detect_params_size.argtypes=[]
        self.library.collision_detect_params_size.restype=ct.c_size_t
        if self.library.collision_detect_params_size()!=ct.sizeof(Params):
            raise ValueError('collision detector CUDA/Python parameter ABI mismatch')
        self.library.launch_collision_detect_tiled.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
        self.library.launch_collision_detect_tiled.restype = ct.c_int
        self.library.launch_collision_detect_compact_producer.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
        self.library.launch_collision_detect_compact_producer.restype=ct.c_int
        self.library.launch_collision_detect_parallel.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.library.launch_collision_detect_parallel.restype=ct.c_int
        self._queue_owner=self.queue_offsets
        self._queue_token=(self.queue_offsets.ptr,self.queue_offsets.shape,self.queue_offsets.dtype)
        self.device_params = wp.array(np.frombuffer(bytes(self.params), dtype=np.uint8).copy(),
                                      dtype=wp.uint8, device=self.device)
        self.manifest.update({'schema':'cuda-collision-detector-r01',
            'single_kernel': False, 'block_threads':64,
            'diagnostic_profile':diagnostic_profile,
            'primitive_producer':'parallel; ordered first-error prefix commit',
            'primitive_contact_capacity':8,
            'raw_mirror_mapping':'64-thread block-linear uint64 compatibility copy; all pairs incl inactive/unused; fast skips mirror',
            'prepare_thread_contract':64,
            'public_raw_mapping':'8 lanes per original call; uint64 prefix/frame copy',
            'mesh_mapping':'AoS retained; subgroup contiguous float runs and bit-preserving shuffle',
            'mesh_mirror_bytes':0,
            'grid_mapping':'prepare=(1,worlds); producer=(ceil(pairs/64),worlds,15 routes); commit=(1,worlds)',
            'query_subgroup_width':query_subgroup_width,
            'mesh_descriptor_bytes':workspace.mesh_descriptor['descriptor_bytes'],
            'private_contact_capacity':{'mpr':50,'capsulebox':8,'boxbox':50,'boxbox_fallback':50},
            'parallel_private_contact_capacity':{'mpr':5,'capsulebox':8,'boxbox':50},
            'parallel_mpr_bound':'CPU 2.3.7 initial contact + four perturbation trials; public raw50 unchanged',
            'cold_abi_only':True,
            'route_subgroup_widths':'reserved pointer; actual query_subgroup_width admits0/32 only',
            'query_order':'stable route radix buckets, original-slot scatter',
            'scratch_bytes_per_world':8*workspace.pairs+64+8*self.task_capacity+4,
            'compact_task_capacity':self.task_capacity,
            'parallel_queue_bytes':self.queue_offsets.size*8,
            'parallel_grid_mapping':'four route families; compact live world tiles; global worker grid',
            'capacity':{n:getattr(workspace,n) for n in INTEGER_NAMES},
            'baseline_launches': ['raw-copy','native-detector','warp-public-detector'],
            'route_semantics':'live route_id controls supported and gated paths',
            'device_util':'libero_collision_cuda::detect_world(Params&, int)'})

    def validate(self, stream=None):
        """Validate the retained pointer ABI without issuing a GPU launch."""
        if any(a is not b for a,b in zip((self.execution_count,self.primitive_raw,
                self.primitive_count,self.primitive_status,self.query_order,
                self.query_tmp,self.bucket_offsets,self.task_bucket,self.task_start,self.task_count,self.mesh_edge_end,self.route_subgroup_widths),self.owners[-12:])):
            raise ValueError('detector retained scratch/counter rebound; reinstall')
        if self.bindings != tuple((a.ptr,a.shape,a.dtype) for a in self.owners):
            raise ValueError('detector bindings changed; reinstall')
        if any(getattr(self.workspace,n) is not a for n,a in zip(self.public_pointer_names,self.owners[:-12])):
            raise ValueError('detector arrays rebound; reinstall')
        if any(getattr(self.workspace,n) != getattr(self.params,n) for n in INTEGER_NAMES) or \
                self.workspace.mpr_tolerance != self._mpr_tolerance or \
                self.params.mpr_tolerance != scalar.ct_real(self._mpr_tolerance).value:
            raise ValueError('detector capacity/options changed; reinstall')
        if self.diagnostic_profile not in ('compatibility','fast') or \
                self.params.diagnostic_profile != int(self.diagnostic_profile=='fast'):
            raise ValueError('detector diagnostic profile changed; reinstall')
        if self.query_subgroup_width not in (0,32) or self.params.query_subgroup_width != self.query_subgroup_width:
            raise ValueError('detector query subgroup width changed; reinstall')
        if self.workspace.mesh_edge_end is not self.mesh_edge_end or len(self.workspace._mesh_descriptor_owners)!=5 or any(a is not b for a,b in zip(self.workspace._mesh_descriptor_owners,(self.workspace.mesh_vertadr,self.workspace.mesh_vertnum,self.workspace.mesh_graphadr,self.workspace.mesh_graph,self.mesh_edge_end))):
            raise ValueError('cold mesh descriptor owner changed; reinstall')
        if self.box_private_capacity!=50 or self.params.box_private_capacity!=50:
            raise ValueError('CG06 cold ABI box capacity changed; rebuild')
        stream = stream or wp.get_stream(self.device)
        if wp.get_device(stream.device) != self.device:
            raise ValueError('detector stream belongs to another device')
        return stream

    def launch_compact_producer(self, blocks, stream=None, world_guard=None):
        if isinstance(blocks,bool) or not isinstance(blocks,int) or not 1<=blocks<=2147483647:
            raise ValueError('positive integer producer blocks required')
        stream=self.validate(stream)
        with wp.ScopedDevice(self.device):
            code=self.library.launch_collision_detect_compact_producer(self.device_params.ptr,
                self.workspace.nworld,blocks,stream.cuda_stream,world_guard.ptr if world_guard is not None else None)
        if code:raise RuntimeError('CUDA compact detector launch failed: '+str(code))

    def launch_parallel_producer(self, blocks, stream=None, world_guard=None):
        stream=self.validate(stream)
        if self.queue_offsets is not self._queue_owner or (self.queue_offsets.ptr,self.queue_offsets.shape,self.queue_offsets.dtype)!=self._queue_token:
            raise ValueError('collision route queue owner changed')
        with wp.ScopedDevice(self.device):
            code=self.library.launch_collision_detect_parallel(self.device_params.ptr,
                self.workspace.nworld,blocks,self.queue_offsets.ptr,
                world_guard.ptr if world_guard is not None else None,stream.cuda_stream)
        if code:raise RuntimeError('CUDA parallel detector launch failed: '+str(code))

    def launch(self, stream=None):
        stream=self.validate(stream)
        with wp.ScopedDevice(self.device):
            code=self.library.launch_collision_detect_tiled(self.device_params.ptr,
                    self.workspace.nworld, self.workspace.pairs, stream.cuda_stream)
        if code:
            raise RuntimeError('CUDA detector launch failed: '+str(code))


def install_detector(workspace, build_dir=None, *, diagnostic_profile='compatibility', query_subgroup_width=32, box_private_capacity=50):
    if diagnostic_profile=='compatible':diagnostic_profile='compatibility'
    if isinstance(box_private_capacity,bool) or box_private_capacity!=50:
        raise ValueError('CG06 cold ABI admits original box capacity50 only')
    if query_subgroup_width not in (0,32):raise ValueError('CG06 cold ABI admits subgroup0/32 only')
    if hasattr(workspace, '_cuda_detector_plan'):
        if workspace._cuda_detector_plan.diagnostic_profile != diagnostic_profile:
            raise ValueError('detector diagnostic profile changed; reinstall workspace')
        if workspace._cuda_detector_plan.query_subgroup_width != query_subgroup_width:
            raise ValueError('detector query subgroup width changed; reinstall workspace')
        return workspace._cuda_detector_plan
    if build_dir is None:
        root=os.environ.get('LIBERO_CUDA_FUSION_BUILD_ROOT') or os.environ.get('P06C_NATIVE_BUILD_ROOT')
        if not root:
            raise ValueError('build_dir or LIBERO_CUDA_FUSION_BUILD_ROOT required')
        build_dir=Path(root)/'collision_detect'
    plan=DetectorPlan(workspace,build_dir,diagnostic_profile=diagnostic_profile,query_subgroup_width=query_subgroup_width,box_private_capacity=box_private_capacity)
    workspace._cuda_detector_original_launch=workspace.launch
    workspace._cuda_detector_plan=plan
    def launch(self,stream=None):
        return self._cuda_detector_plan.launch(stream)
    workspace.launch=types.MethodType(launch,workspace)
    return plan


def install(runtime, build_root=None, *, diagnostic_profile='compatibility', query_subgroup_width=32, box_private_capacity=50):
    """Accept DetectorWorkspace, CollisionChainWorkspace or its runtime wrapper."""
    if hasattr(runtime,'runtime'):
        runtime=runtime.runtime
    if hasattr(runtime,'forward'):
        runtime=runtime.forward.c.workspace
    if hasattr(runtime,'workspace'):
        runtime=runtime.workspace
    ws=getattr(runtime,'collision',runtime)
    ws=getattr(ws,'chain',ws)
    ws=getattr(ws,'detect',ws)
    return install_detector(ws,build_root,diagnostic_profile=diagnostic_profile,query_subgroup_width=query_subgroup_width,box_private_capacity=box_private_capacity)
