"""Cold-bound exact two-field transport into fresh private applied-force roots.

This independent ABI deliberately owns no old D/Newton/forward publications.
Its two private consumer-valid flags revoke before any payload store. Existing
runtime force-write/downstream invalidation is a separate unavailable contract.
"""
import libero.gpu.precision as scalar
from pathlib import Path
from threading import RLock

def _token(a):
    return(id(a),a.ptr,a.dtype,tuple(a.shape),tuple(a.strides),a.size,a.device)

class AppliedForceWrite:
    def __init__(self,owner,source,*,capacity,nv,nbody,stream,wp,native,source_root,credential,fused=False,qualification=None):
        if any(type(v) is not int or v<=0 or v>=2**31 for v in (capacity,nv,nbody)):
            raise ValueError('positive int32 applied-force dimensions required')
        if capacity*max(nv,6*nbody)>=2**31:raise ValueError('applied-force extent exceeds int32 ABI')
        if type(fused) is not bool:raise TypeError('explicit cold transport variant required')
        if not callable(getattr(credential,'validate',None)):raise TypeError('qualified cold source credential required')
        self._owner,self._source,self._stream=owner,source,stream
        self._wp,self._credential=wp,credential
        self._capacity,self._nv,self._nbody=capacity,nv,nbody
        self._failed=False;self._lock=RLock();self._plans=()
        self._fields=('qfrc_applied','xfrc_applied','requested','active','first_error','force_valid','full_valid')
        self._arrays=self._current()
        self._check_layout(self._arrays)
        self._sealed=tuple(_token(a) for a in self._arrays)
        credential.validate()
        root=Path(source_root)
        sources=[root/'cuda/common/vector.cuh',root/'cuda/runtime/copies.cuh',root/'cuda/runtime/applied_force_write.cuh']
        q,x,qs,xs,req,act,err,fv,full=self._arrays
        if fused:
            from .applied_force_policy import select
            fused=select(capacity,nv,nbody,Path(source_root),qualification)
        if fused:
            self._plans=(native('force_both',sources,[('cf_force_write::both',
                [qs,q,xs,x,req,act,err,fv,full,nv,6*nbody],'block')],capacity),)
        else:
            self._plans=tuple(native('force_single',sources,[('cf_force_write::single',
                [src,dst,req,act,err,fv,full,width],'block')],capacity)
                for src,dst,width in ((qs,q,nv),(xs,x,6*nbody)))
        self._variant='composed_one_launch' if fused else 'original_two_payload_launches'
    def _current(self):
        if getattr(self._owner,'stream',None) is not self._stream or getattr(self._source,'stream',None) is not self._stream:
            raise ValueError('applied-force foreign stream owner')
        return(getattr(self._owner,'qfrc_applied',None),getattr(self._owner,'xfrc_applied',None),getattr(self._source,'qfrc_applied',None),
            getattr(self._source,'xfrc_applied',None),*(getattr(self._owner,n,None) for n in self._fields[2:]))
    def _check_layout(self,arrays):
        W,nv,nb=self._capacity,self._nv,self._nbody
        shapes=((W,nv),(W,nb,6),(W,nv),(W,nb,6))+((W,),)*5
        spans=[]
        for i,(a,shape) in enumerate(zip(arrays,shapes)):
            dtype=scalar.wp_real if i<4 else self._wp.int32;width=scalar.itemsize if i<4 else 4
            if not isinstance(a,self._wp.array) or a.dtype!=dtype or tuple(a.shape)!=shape:
                raise ValueError('applied-force exact shape/dtype owner required')
            count=1;strides=[]
            for extent in reversed(shape):strides.insert(0,count*width);count*=extent
            if (a.device!=self._stream.device or tuple(a.strides)!=tuple(strides)
                    or a.size!=count or not a.ptr or a.ptr%width):
                raise ValueError('applied-force exact device/contiguous/span ABI required')
            lo,hi=int(a.ptr),int(a.ptr)+count*width
            if any(lo<other_hi and other_lo<hi for other_lo,other_hi in spans):
                raise ValueError('applied-force source/target/status spans must be disjoint')
            spans.append((lo,hi))
    def validate(self):
        if self._failed:raise ValueError('applied-force failed submission requires fresh owner')
        self._credential.validate()
        arrays=self._current();self._check_layout(arrays)
        if tuple(_token(a) for a in arrays)!=self._sealed:
            raise ValueError('applied-force source/target owner changed')
    def enqueue(self):
        with self._lock:
            self.validate()
            try:
                for plan in self._plans:plan.launch(stream=self._stream)
            except BaseException:self._failed=True;raise
    @property
    def manifest(self):
        return dict(abi='fresh-private-applied-force-copy-v1',capacity=self._capacity,nv=self._nv,nbody=self._nbody,
            variant=self._variant,launches=len(self._plans),installed_default=False,raw_handles_exposed=False,
            production_downstream_invalidation_qualified=False)
