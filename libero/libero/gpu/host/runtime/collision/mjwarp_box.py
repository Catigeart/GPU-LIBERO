"""Independent sphere/capsule-box and opt-in primitive box-box contact binding."""
import ctypes as ct
from .mjwarp_plane import PlaneContacts,ContactInputs,qualify as qualify_plane


def qualify(model,data,*,implementation='upstream'):
    if implementation not in ('upstream','cuda'):raise ValueError('box_collision must be upstream or cuda')
    a=qualify_plane(model,data,implementation=implementation)
    if a['fallback']:a['fallback']=a['fallback'].replace('plane collision','box collision')
    return a


class BoxContacts(PlaneContacts):
    component='box'
    families=['sphere-box','capsule-box','primitive-box-box']
    def __init__(self,model,data,context,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        super().__init__(model,data,context,library,stream,admission=admission)
        self.boxbox=int(model.has_boxbox_margin or bool(model.opt.disableflags & types.DisableBit.NATIVECCD))
        index=lambda i,j:i*(2*len(types.GeomType)-i-1)//2+j
        families=[(2,6),(3,6)]+([(6,6)] if self.boxbox else [])
        self.enabled=any(model.geom_pair_type_count[index(i,j)] for i,j in families)
        self.launch.argtypes=[ct.POINTER(ContactInputs),ct.c_int,ct.c_void_p]
    @property
    def identity(self):return dict(super().identity,boxbox_enabled=bool(self.boxbox),has_candidate_families=bool(self.enabled))
    def enqueue(self):
        if not self.enabled:return
        error=self.launch(ct.byref(self.arguments),self.boxbox,self.stream.cuda_stream)
        if error:raise RuntimeError('Box collision launch failed: CUDA error '+str(error))
