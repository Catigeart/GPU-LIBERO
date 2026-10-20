"""Device-only copies between runtime raw fields and packed observation sources."""
import libero.gpu.precision as scalar
import warp as wp

@wp.kernel
def _copy_field(source: wp.array(dtype=scalar.wp_real), target: wp.array(dtype=scalar.wp_real),
                source_stride: int, target_stride: int, offset: int, length: int):
    world,j=wp.tid()
    target[world*target_stride+offset+j]=source[world*source_stride+j]

class DeviceFieldPacker:
    """Reuse one contiguous FP64 source; each update is exclusively device copies.

    layout maps names to (offset, per_world_length). Input fields must be contiguous
    FP64 Warp arrays with exactly nworld*length elements. Caller supplies the legal
    per-field publications and uses the same stream as the physics scheduler.
    """
    def __init__(self,layout,nworld=1,device="cuda:0"):
        self.layout=dict(layout);self.nworld=nworld;self.device=device
        self.stride=max((o+n for o,n in layout.values()),default=0)
        occupied=set()
        for name,(offset,length) in layout.items():
            if offset<0 or length<0:raise ValueError(name)
            region=set(range(offset,offset+length))
            if occupied.intersection(region):raise ValueError(f"overlapping source layout: {name}")
            occupied.update(region)
        self.buffer=wp.zeros(nworld*self.stride,dtype=scalar.wp_real,device=device)

    def validate(self,fields):
        for name,(offset,length) in self.layout.items():
            source=fields[name]
            if source.dtype!=scalar.wp_real or source.size!=self.nworld*length:
                raise ValueError(f"field shape/type mismatch: {name}")
            if source.device!=self.buffer.device:raise ValueError(f"field device mismatch: {name}")
            if source.ndim!=1:raise ValueError(f"flatten device field before packing: {name}")
        return fields

    def copy_from(self,fields,stream=None):
        self.validate(fields)
        for name,(offset,length) in self.layout.items():
            source=fields[name]
            if length:
                wp.launch(_copy_field,dim=(self.nworld,length),inputs=[source,self.buffer,length,self.stride,offset,length],device=self.device,stream=stream)
        return self.buffer