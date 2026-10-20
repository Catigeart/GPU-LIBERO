"""Load-only FP32 mass factorization and fused factor/solve on current arrays."""
import ctypes as ct
import hashlib
from pathlib import Path


class MassFactorInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','nv','blocks','matrix_width','factor_width','rhs_width','solution_width')]
    _fields_ += [(k,ct.c_void_p) for k in ('matrix_row','factor_address','block_start','gather','matrix','rhs',
                                         'factor','diagonal_inverse','solution')]


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/dynamics/mjwarp_mass_factor.cu','cuda/dynamics/mjwarp_mass_factor.cuh','cmake/MjwarpMassFactor.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()


def unsupported_reason(model,data):
    if model.nv<=0:return 'empty mass dimension'
    if data.qLD.ndim!=2 or data.qLD.shape[1]!=model.qLD_block_total:return 'unported sparse LDL mass region or factor layout'
    if (model.block_dim.small_cholesky!=64 or model.block_dim.cholesky_factorize!=32 or
            model.block_dim.cholesky_factorize_solve!=32):return 'unqualified factor block dimension'
    if not model.M_tiles:return 'no qualified mass blocks'
    for tile in model.M_tiles:
        if tile.elemid.size==0 and tile.size not in (1,6):return 'unqualified scalar mass block size'
        if tile.elemid.size!=0 and tile.size!=9:return 'unqualified tile mass block size'
    return None


class MassFactorPlan:
    """Validate static CSR/block maps once, before capture; cache no matrix values."""
    def __init__(self,model,data,library,stream):
        import numpy as np
        import warp as wp
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        reason=unsupported_reason(model,data)
        if reason:raise ValueError(reason)
        self.model,self.data,self.stream=model,data,stream
        self.path=Path(library).resolve()
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_mjwarp_mass_factor_fp32 offline')
        self.library=ct.CDLL(str(self.path))
        self.library.flash_mjwarp_mass_factor_source_id.restype=ct.c_char_p
        self.library.flash_mjwarp_mass_factor_arguments_bytes.restype=ct.c_size_t
        if self.library.flash_mjwarp_mass_factor_source_id().decode()!=source_id():
            raise MissingCompiledProduct('Mass factor binary is stale; rebuild offline')
        if self.library.flash_mjwarp_mass_factor_arguments_bytes()!=ct.sizeof(MassFactorInputs):
            raise ValueError('Mass factor C/Python ABI differs')
        if not stream.device.is_cuda or self.library.flash_mjwarp_mass_factor_arch()!=stream.device.arch:
            raise MissingCompiledProduct('Mass factor architecture differs')
        nv,nc=model.nv,model.nC
        if data.nworld<1 or nc<nv:raise ValueError('Mass factor dimensions differ')
        def static(a,shape):
            if a.device!=stream.device or a.dtype!=wp.int32 or not a.is_contiguous or tuple(a.shape)!=shape:
                raise ValueError('Mass factor index array layout differs')
            return a.numpy()
        row=static(model.M_rowadr,(nv,));nnz=static(model.M_rownnz,(nv,));col=static(model.M_colind,(nc,))
        address=static(model.qLD_block_adr,(nv,))
        if (np.any(nnz<1) or not np.array_equal(row,np.r_[0,np.cumsum(nnz)[:-1]]) or int(nnz.sum())!=nc):
            raise ValueError('Mass factor CSR rows differ')
        pairs={}
        for i in range(nv):
            columns=col[row[i]:row[i]+nnz[i]]
            if columns[-1]!=i or np.any(columns<0) or np.any(columns>i) or np.any(np.diff(columns)<=0):
                raise ValueError('Mass factor CSR columns differ')
            for j,k in enumerate(columns):pairs[i,int(k)]=int(row[i]+j)
        coverage=np.zeros(nv,np.int32);regions=np.zeros(model.qLD_block_total,np.int32)
        owners=np.full(nv,-1,np.int32);groups=[]
        for tile in model.M_tiles:
            size=int(tile.size);starts=static(tile.adr,(tile.adr.size,));kind=int(tile.elemid.size!=0)
            gather=static(tile.elemid,(len(starts)*size*size,)) if kind else None
            for b,first in enumerate(starts):
                first=int(first);last=first+size;adr=int(address[first]) if 0<=first<nv else -1
                if first<0 or last>nv or not np.all(address[first:last]==adr):
                    raise ValueError('Mass factor block partition differs')
                coverage[first:last]+=1;owners[first:last]=len(groups)
                if adr==-2 and not kind:
                    if not np.all(nnz[first:last]==1) or not np.array_equal(row[first:last],row[first]+np.arange(size)):
                        raise ValueError('Compact mass block layout differs')
                else:
                    if adr<0 or adr+size*size>model.qLD_block_total:raise ValueError('Mass factor block exceeds output storage')
                    regions[adr:adr+size*size]+=1
                    if kind:
                        expected=np.array([pairs.get((max(first+i,first+j),min(first+i,first+j)),nc)
                                           for i in range(size) for j in range(size)])
                        actual=gather[b*size*size:(b+1)*size*size]
                        valid=(actual==expected)|((expected==nc)&((actual<0)|(actual>=nc)))
                        if not np.all(valid):raise ValueError('Mass factor gather differs from CSR matrix')
                    elif any(nnz[first+i]!=i+1 or row[first+i]!=row[first]+i*(i+1)//2 or
                             not np.array_equal(col[row[first+i]:row[first+i]+i+1],np.arange(first,first+i+1)) for i in range(size)):
                        raise ValueError('Scalar mass triangle layout differs')
            if len(starts):groups.append(dict(kind=kind,size=size,count=len(starts),starts=tile.adr,gather=tile.elemid))
        if not np.all(coverage==1) or np.any(regions>1):raise ValueError('Mass factor blocks overlap or omit DOFs')
        # Distinct blocks may not contain off-block mass entries, even if the
        # caller supplied internally consistent but incomplete gather indices.
        for (i,j) in pairs:
            if address[i]!=address[j] or owners[i]!=owners[j]:raise ValueError('Mass matrix crosses independent blocks')
        self.groups=groups

    @property
    def identity(self):
        return dict(source_id=source_id(),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    groups=[{k:g[k] for k in ('kind','size','count')} for g in self.groups],
                    matrix_values='borrow-current-input-no-cache')

    def bind(self,matrix,factor,diagonal,solution=None,rhs=None):
        return MassFactor(self,matrix,factor,diagonal,solution,rhs)


class MassFactor:
    def __init__(self,plan,matrix,factor,diagonal,solution,rhs):
        import warp as wp
        m,d,s=plan.model,plan.data,plan.stream
        if (solution is None)!=(rhs is None):raise ValueError('Factor solve needs both vectors')
        n=d.nworld;solve=rhs is not None
        arrays=dict(matrix=matrix,factor=factor,diagonal=diagonal)
        shapes=dict(matrix=(n,m.nC),factor=(n,m.qLD_block_total),diagonal=(n,m.nv))
        if solve:
            arrays.update(solution=solution,rhs=rhs)
            for key,a in (('solution',solution),('rhs',rhs)):
                if a.ndim!=2 or a.shape[0]!=n or a.shape[1]<m.nv:raise ValueError('Mass factor vector shape differs')
                shapes[key]=tuple(a.shape)
        for key,a in arrays.items():
            if (a.device!=s.device or a.dtype!=wp.float32 or not a.is_contiguous or tuple(a.shape)!=shapes[key]):
                raise ValueError('Mass factor array device/layout/dtype differs: '+key)
        for out in ('factor','diagonal')+(('solution',) if solve else ()):
            a=arrays[out]
            for key,b in arrays.items():
                if key==out:continue
                if not a.capacity or not b.capacity:continue
                if out=='solution' and key=='rhs' and a.ptr==b.ptr and a.shape==b.shape:continue
                if max(a.ptr,b.ptr)<min(a.ptr+a.capacity,b.ptr+b.capacity):
                    raise ValueError('Mass factor arrays overlap: '+out+'/'+key)
        self.plan,self.arrays,self.solve=plan,arrays,solve
        self.arguments=[]
        for g in plan.groups:
            a=MassFactorInputs(worlds=n,nv=m.nv,blocks=g['count'],matrix_width=m.nC,factor_width=m.qLD_block_total,
                rhs_width=rhs.shape[1] if solve else 0,solution_width=solution.shape[1] if solve else 0,
                matrix_row=m.M_rowadr.ptr,factor_address=m.qLD_block_adr.ptr,block_start=g['starts'].ptr,
                gather=g['gather'].ptr,matrix=matrix.ptr,rhs=rhs.ptr if solve else None,
                factor=factor.ptr,diagonal_inverse=diagonal.ptr,solution=solution.ptr if solve else None)
            self.arguments.append((a,g['kind'],g['size']))
        self.launch=plan.library.flash_mjwarp_mass_factor_launch
        self.launch.argtypes=[ct.POINTER(MassFactorInputs),ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    def enqueue(self):
        for a,kind,size in self.arguments:
            error=self.launch(ct.byref(a),kind,size,int(self.solve),self.plan.stream.cuda_stream)
            if error:raise RuntimeError('Mass factor launch failed: CUDA error '+str(error))
