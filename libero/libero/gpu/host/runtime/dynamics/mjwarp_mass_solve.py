"""Load-only FP32 backsubstitution for the pinned LIBERO mass-block layouts."""
import ctypes as ct
import hashlib
from pathlib import Path


class MassSolveInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','blocks','threads','factor_stride',
                                  'diagonal_stride','rhs_stride','solution_stride')]
    _fields_ += [(k,ct.c_void_p) for k in ('factor_address','block_start','factor','diagonal_inverse','rhs','solution')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/dynamics/mjwarp_mass_solve.cu','cuda/dynamics/mjwarp_mass_solve.cuh','cmake/MjwarpMassSolve.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()


def unsupported_reason(model,data):
    """Unknown layouts keep the complete upstream caller, before any CUDA load."""
    if model.nv<=0:return 'empty mass dimension'
    if data.qLD.ndim!=2 or data.qLD.shape[1]!=model.qLD_block_total:
        return 'unported sparse LDL mass region or factor layout'
    if model.block_dim.small_cholesky!=64 or model.block_dim.cholesky_solve!=64:
        return 'unqualified mass block dimension'
    if not model.M_tiles:return 'no qualified mass blocks'
    for tile in model.M_tiles:
        if tile.elemid.size==0 and tile.size not in (1,6):return 'unqualified scalar mass block size'
        if tile.elemid.size!=0 and tile.size!=9:return 'unqualified tile mass block size'
    return None


class MassSolvePlan:
    """Qualify immutable layout metadata once; borrow fresh factors on each call."""

    def __init__(self,model,data,library,stream):
        import numpy as np
        import warp as wp
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        reason=unsupported_reason(model,data)
        if reason:raise ValueError(reason)
        if data.nworld<1:raise ValueError('invalid mass world count')
        self.library_path=Path(library).resolve()
        if not self.library_path.is_file():raise MissingCompiledProduct('Build flash_mjwarp_mass_solve_fp32 offline')
        self.library=ct.CDLL(str(self.library_path))
        self.library.flash_mjwarp_mass_solve_source_id.restype=ct.c_char_p
        self.library.flash_mjwarp_mass_solve_arguments_bytes.restype=ct.c_size_t
        if self.library.flash_mjwarp_mass_solve_source_id().decode()!=source_id():
            raise MissingCompiledProduct('Mass solve binary is stale; rebuild its CMake target offline')
        if self.library.flash_mjwarp_mass_solve_arguments_bytes()!=ct.sizeof(MassSolveInputs):
            raise ValueError('Mass solve C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_mass_solve_arch()!=stream.device.arch:
            raise MissingCompiledProduct('Mass solve binary architecture differs')
        n,nv=data.nworld,model.nv
        for name,a,dtype in [('factor',data.qLD,wp.float32),('diagonal',data.qLDiagInv,wp.float32),
                             ('factor_address',model.qLD_block_adr,wp.int32)]:
            if a.device!=stream.device or not a.is_contiguous or a.dtype!=dtype:
                raise ValueError('Mass solve array device/layout/dtype differs: '+name)
        if data.qLD.shape[0]!=n or tuple(data.qLDiagInv.shape)!=(n,nv) or tuple(model.qLD_block_adr.shape)!=(nv,):
            raise ValueError('Mass solve factor/diagonal shape differs')
        # These are STATIC model indices, read before capture, not downloaded
        # physical factors. No numerical factor values are cached by the plan.
        address=model.qLD_block_adr.numpy();coverage=np.zeros(nv,np.int32);groups=[]
        for tile in model.M_tiles:
            a=tile.adr
            if a.device!=stream.device or not a.is_contiguous or a.dtype!=wp.int32 or a.ndim!=1:
                raise ValueError('Mass solve block-start layout differs')
            starts=a.numpy();kind=0 if tile.elemid.size==0 else 1
            for start in starts:
                start=int(start);size=int(tile.size)
                if start<0 or start+size>nv:raise ValueError('Mass solve block exceeds DOFs')
                adr=int(address[start])
                if not np.all(address[start:start+size]==adr):raise ValueError('Mass solve block factor map differs')
                if adr==-2 and kind==0:pass
                elif adr<0 or adr+size*size>model.qLD_block_total:
                    raise ValueError('Mass solve block factor address exceeds storage')
                coverage[start:start+size]+=1
            if len(starts):groups.append(dict(kind=kind,size=tile.size,starts=a,count=len(starts)))
        if not np.all(coverage==1):raise ValueError('Mass solve blocks must partition every DOF exactly once')
        self.model,self.data,self.stream,self.groups=model,data,stream,groups

    @property
    def identity(self):
        return dict(source_id=source_id(),library_sha256=hashlib.sha256(self.library_path.read_bytes()).hexdigest(),
                    groups=[dict(kind=g['kind'],size=g['size'],count=g['count']) for g in self.groups],
                    factor_values='borrow-current-data-no-cache')

    def bind(self,solution,rhs):
        return MassSolve(self,solution,rhs)


class MassSolve:
    """A reusable pointer binding; exact in-place solve is supported as upstream."""

    def __init__(self,plan,solution,rhs):
        import warp as wp
        m,d,s=plan.model,plan.data,plan.stream
        for name,a in [('rhs',rhs),('solution',solution)]:
            if a.device!=s.device or a.dtype!=wp.float32 or not a.is_contiguous or a.ndim!=2 or (
                    a.shape[0]!=d.nworld or a.shape[1]<m.nv):
                raise ValueError('Mass solve vector device/layout/dtype/shape differs: '+name)
        # Partial overlaps are not the upstream exact-alias contract.
        if solution.ptr!=rhs.ptr and max(solution.ptr,rhs.ptr)<min(
                solution.ptr+solution.capacity,rhs.ptr+rhs.capacity):
            raise ValueError('Mass solve vectors partially overlap')
        if solution.ptr==rhs.ptr and solution.shape!=rhs.shape:
            raise ValueError('In-place mass solve vector shapes differ')
        self.plan,self.solution,self.rhs=plan,solution,rhs
        self.arguments=[]
        for g in plan.groups:
            a=MassSolveInputs(worlds=d.nworld,nv=m.nv,blocks=g['count'],threads=64,
                factor_stride=d.qLD.shape[1],diagonal_stride=d.qLDiagInv.shape[1],
                rhs_stride=rhs.shape[1],solution_stride=solution.shape[1],factor_address=m.qLD_block_adr.ptr,
                block_start=g['starts'].ptr,factor=d.qLD.ptr,diagonal_inverse=d.qLDiagInv.ptr,
                rhs=rhs.ptr,solution=solution.ptr)
            self.arguments.append((a,g['kind'],g['size']))
        self.launch=plan.library.flash_mjwarp_mass_solve_launch
        self.launch.argtypes=[ct.POINTER(MassSolveInputs),ct.c_int,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    def enqueue(self):
        for a,kind,size in self.arguments:
            error=self.launch(ct.byref(a),kind,size,self.plan.stream.cuda_stream)
            if error:raise RuntimeError('Handwritten mass solve launch failed: CUDA error '+str(error))
