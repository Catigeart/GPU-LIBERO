"""Optional connected dense factor/solve using constant-size MathDx tiles.

Python binds arrays and compiles handwritten CUDA. It translates no arithmetic.
This API handles an already assembled H and RHS. It is not a replacement for
the persistent Newton Hessian builder, line search, or stopping protocol.
"""
import libero.gpu.precision as scalar
from pathlib import Path
import ctypes as ct
import warp as wp

from libero.gpu.host.build.native import NativePlan

GPU_ROOT = Path(__file__).resolve().parents[3]


def _overlap(a, b):
    a_bytes=a.size*wp.types.type_size_in_bytes(a.dtype)
    b_bytes=b.size*wp.types.type_size_in_bytes(b.dtype)
    return a.ptr < b.ptr + b_bytes and b.ptr < a.ptr + a_bytes


class BlockedFactorPlan:
    """Sealed, graph-compatible factor plus two triangular solves.

    Matrix arrays are FP64 [world,max_nv,ld], ld >= max_nv. Vector arrays are
    FP64 [world,max_nv]. ``n``, ``active``, ``status`` are int32 [world]. Runtime
    n can differ between worlds and launches; values outside [0,max_nv] fail on
    the device without publishing an output. RHS is the desired solve RHS;
    a Newton caller supplies -gradient if that is its direction convention.

    Only active lower factor cells and vector prefixes are committed. All three
    public outputs commit together after finite checks. A prior nonzero status
    or inactive world preserves all public/private diagnostics. Status codes:
    1 dimensions, 2 nonfinite input, 3 nonfinite factor or solve. Factor may alias
    H and either vector output may alias RHS. The two output vectors must be
    disjoint. Independent calls require independent plans/private workspace.
    """
    def __init__(self, H, rhs, factor, direction, forward, n, active, status,
                 build_root, *, block_size=128, tile=16):
        if tile not in (16,32) or block_size not in (32,64,128,256):
            raise ValueError('tile 16/32 and block size 32/64/128/256 required')
        if not isinstance(H,wp.array) or len(H.shape)!=3:
            raise ValueError('H must be a Warp matrix [world,max_nv,ld]')
        worlds,capacity,ld=H.shape
        if capacity<1 or worlds<1 or ld<capacity:
            raise ValueError('positive dimensions and ld >= max_nv required')
        device=H.device
        bindings=[H,rhs,factor,direction,forward,n,active,status]
        shapes=[H.shape,(worlds,capacity),H.shape,(worlds,capacity),(worlds,capacity),
                (worlds,),(worlds,),(worlds,)]
        for index,(array,shape) in enumerate(zip(bindings,shapes)):
            dtype=scalar.wp_real if index<5 else wp.int32
            if not isinstance(array,wp.array) or not array.is_contiguous or array.shape!=shape or \
                    array.dtype!=dtype or array.device!=device or not array.device.is_cuda:
                raise ValueError('blocked factor binding shape/dtype/device/contiguity mismatch')
        # Exact sanctioned aliases only. Arbitrary subviews overlap differently.
        for i,left in enumerate(bindings):
            for j,right in enumerate(bindings[:i]):
                if not _overlap(left,right):continue
                permitted={i,j} in ({0,2},{1,3},{1,4})
                if not permitted or left.ptr!=right.ptr or left.shape!=right.shape:
                    raise ValueError('unsupported overlapping blocked factor bindings')
        self.max_nv=capacity;self.ld=ld;self.nworld=worlds;self.device=device
        self.working=wp.empty(H.shape,dtype=scalar.wp_real,device=device)
        self.working_x=wp.empty((worlds,capacity),dtype=scalar.wp_real,device=device)
        self.working_y=wp.empty((worlds,capacity),dtype=scalar.wp_real,device=device)
        self.diagnostic=wp.zeros((worlds,8),dtype=wp.int32,device=device)
        self._bindings=bindings
        self._tokens=tuple((x.ptr,x.shape,x.dtype,x.device) for x in bindings)
        args=bindings[:5]+[n,ld,active,status,self.working,self.working_x,
                          self.working_y,self.diagnostic]
        shared_bytes=4*tile*tile*8+64
        header=GPU_ROOT/'cuda/newton/blocked_factor.cuh'
        bridge=GPU_ROOT/'cuda/newton/blocked_factor_dx_bridge.cu'
        certificate=r'''
struct Params;
__global__ void native_kernel(Params);
extern "C" int blocked_resources(int* shared,int* registers,int* local) {
    cudaFuncAttributes attributes;
    cudaError_t error=cudaFuncGetAttributes(&attributes,native_kernel);
    if(error!=cudaSuccess)return int(error);
    *shared=int(attributes.sharedSizeBytes);*registers=attributes.numRegs;
    *local=int(attributes.localSizeBytes);return 0;
}
'''
        self.plan=NativePlan('connected_blocked_factor',[header,certificate],
            [('libero_blocked::factor_direction',args,'block')],worlds,build_root,
            block_size=block_size,mathdx=True,device=device,bridges=[bridge],
            defines={'BF_TILE':tile,'BF_MAX_NV':capacity,'BF_SHARED_BYTES':shared_bytes,
                     'DX_THREADS':block_size,'DX_SM':device.arch*10},track_execution=False)
        function=self.plan.library.blocked_resources
        function.argtypes=[ct.POINTER(ct.c_int)]*3;function.restype=ct.c_int
        sizes=[ct.c_int() for _ in range(3)]
        with wp.ScopedDevice(device):error=function(*(ct.byref(x) for x in sizes))
        if error:raise RuntimeError('blocked compiled resource query failed: '+str(error))
        if sizes[0].value>49152:
            raise ValueError('linked blocked kernel exceeds portable 48 KiB shared ceiling')
        self.manifest=dict(self.plan.manifest)
        self.manifest.update(experimental=True,scope='already assembled connected H factor/direction',
            max_nv=capacity,leading_dimension=ld,tile=tile,portable_shared_ceiling=49152,
            compiled_static_shared_bytes=sizes[0].value,compiled_registers_per_thread=sizes[1].value,
            compiled_local_bytes_per_thread=sizes[2].value,
            private_global_bytes_per_world=(capacity*ld+2*capacity)*8,
            standard_math={'potrf':'cuSOLVERDx','trsm':'cuSOLVERDx','gemm':'cuBLASDx'},
            regularization='1e-15 source-order whole factor fallback from original H',
            integration='independent callable plan; persistent Newton and Hessian assembly unchanged')

    def validate_bindings(self):
        if tuple((x.ptr,x.shape,x.dtype,x.device) for x in self._bindings)!=self._tokens:
            raise ValueError('blocked public array bindings changed; rebuild the plan')
        self.plan.validate_bindings()

    def launch(self, stream=None):
        self.validate_bindings()
        self.plan.launch(stream=stream)
