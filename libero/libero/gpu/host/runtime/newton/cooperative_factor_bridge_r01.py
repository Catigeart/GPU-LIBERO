"""Independent complete factor_update diagnostic loader, no runtime grant."""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import ctypes as ct
import hashlib,json,traceback
import warp as wp
from .cooperative_rank_bridge_r01 import CooperativeRankModule, OPTIONS, SHIM, BoundRank
from .s07_bridge import default_nvrtc, S07DeviceBridge
from .factor_capture import FIELDS

SOURCE_SHA='3c93486f97a2500f92d009f4bb17c12385df7bd861fd9e63a5ae44d5da7c7c1d'
PAIRS={(21,86),(22,87),(24,89),(27,107),(30,110),(33,128),(36,131),
       (37,132),(39,149),(43,153),(45,170),(51,191),(57,212)}
FLOAT_FIELDS={'J','D','qM','contact_H','H','Hcone','local','ltj','vec'}

def layout(nv,nm,nworld,rows,contacts):
    if any(type(v) is not int for v in (nv,nm,nworld,rows,contacts)) or (nv,nm) not in PAIRS or min(nworld,rows,contacts)<1:
        raise ValueError('factor diagnostic dimension scope')
    widths=(1,1,1,1,rows*nv,rows,rows,rows,nm,nv,nv,nv,rows,contacts,contacts*36,1,nv*nv,nv*nv,1,1,36,6*nv,nv,1,1)
    sizes={name:(width if name in ('madr','parentid','simplenum') else nworld*width) for name,width in zip(FIELDS,widths)}
    if any(v>=2**31 for v in sizes.values()):raise ValueError('factor int32 offset overflow')
    return sizes,(nworld,1,1),(64,1,1)

class CooperativeFactorModule(CooperativeRankModule):
    def __init__(self,build_root,device='cuda:0'):
        self.device=wp.get_device(device)
        if not self.device.is_cuda or self.device.arch!=120:raise ValueError('only actual sm120 diagnostic')
        source=(GPU_ROOT / 'host/runtime/newton/cooperative_factor_update_r01.cu').read_bytes()
        if hashlib.sha256(source).hexdigest()!=SOURCE_SHA:raise ValueError('factor source drift')
        self.root=Path(build_root).resolve();self.root.mkdir(parents=True,exist_ok=False)
        (self.root/'source.cu').write_bytes(source)
        self.record=dict(scope='complete factor_update diagnostic candidate, source/actual numeric acceptance pending',source_sha256=SOURCE_SHA,loader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),compiler_implementation_sha256=hashlib.sha256((GPU_ROOT / 'host/runtime/newton/cooperative_rank_bridge_r01.py').read_bytes()).hexdigest(),options=[v.decode() for v in OPTIONS],shim_sha256=hashlib.sha256(SHIM).hexdigest(),nvrtc_path=default_nvrtc(),block=[64,1,1],accepted=False)
        self._cuda=None
        try:self.ptx=self._compile(source);self._load()
        except BaseException:self.record['failure']=traceback.format_exc();raise
        finally:(self.root/'record.json').write_text(json.dumps(self.record,indent=2)+'\n')
    def _load(self):
        self._cuda=S07DeviceBridge._driver(self)
        self.context=ct.c_void_p();rc=self._cuda.cuCtxGetCurrent(ct.byref(self.context))
        if rc or not self.context.value:raise RuntimeError(('factor context',rc))
        self._module=ct.c_void_p();self._ptx_buffer=ct.create_string_buffer(self.ptx)
        rc=self._cuda.cuModuleLoadData(ct.byref(self._module),ct.cast(self._ptx_buffer,ct.c_void_p))
        if rc:raise RuntimeError(('factor module',rc))
        self.function=ct.c_void_p();rc=self._cuda.cuModuleGetFunction(ct.byref(self.function),self._module,b'n_cooperative_factor_update_r01')
        if rc:raise RuntimeError(('factor function',rc))
    def bind(self,arrays,*,nv,nm,nworld,rows,contacts,stream):
        sizes,grid,block=layout(nv,nm,nworld,rows,contacts)
        if set(arrays)!=set(FIELDS):raise ValueError('factor exact field set')
        owners=tuple(arrays[name] for name in FIELDS)
        spans=[]
        for name,a in zip(FIELDS,owners):
            dtype=scalar.wp_real if name in FLOAT_FIELDS else wp.int32
            if a.dtype!=dtype or a.device!=self.device or a.ndim!=1 or a.size!=sizes[name] or not a.is_contiguous:raise ValueError('factor typed/device/shape/contiguity')
            spans.append((int(a.ptr),int(a.ptr)+a.size*(8 if name in FLOAT_FIELDS else 4)))
        for i,(a,b) in enumerate(spans):
            if any(a<d and c<b for c,d in spans[i+1:]):raise ValueError('factor array owners overlap')
        if stream.device!=self.device:raise ValueError('factor stream ownership')
        boxes=[ct.c_void_p(int(a.ptr)) for a in owners]+[ct.c_int(v) for v in (rows,contacts,nv,nm,nworld)]
        parameters=(ct.c_void_p*len(boxes))(*(ct.addressof(v) for v in boxes))
        return BoundRank(self,owners,stream,grid,block,boxes,parameters)
