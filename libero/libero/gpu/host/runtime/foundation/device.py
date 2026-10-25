"""Device-only shared admission, epochs and sticky first-error.

One thread owns a world. Source producer statuses keep their original codes.
This module routes metadata; it never computes physics or writes producer data.
"""
from enum import IntEnum
import warp as wp
from .schema import checked_product

wp.set_module_options({"enable_backward":False,"fast_math":False,"fuse_fp":True})

class SharedError(IntEnum):
    OK=0
    BAD_ACTIVE=1
    BAD_COUNT=2
    CAPACITY=3
    UNSUPPORTED_ROUTE=4
    NONFINITE=5
    NUMERIC_DOMAIN=6
    STALE_EPOCH=7
    BINDING=8
    WATCHDOG=9

@wp.func
def _first(w:int,kind:int,stage:int,local:int,item:int,
           event:wp.int64,errors:wp.array(dtype=wp.int32),
           first_stage:wp.array(dtype=wp.int32),
           first_local:wp.array(dtype=wp.int32),
           first_item:wp.array(dtype=wp.int32),
           first_event:wp.array(dtype=wp.int64)):
    if errors[w]==0:
        errors[w]=kind
        first_stage[w]=stage
        first_local[w]=local
        first_item[w]=item
        first_event[w]=event

@wp.kernel
def _gate(admission:wp.array(dtype=wp.int32),
          count:wp.array(dtype=wp.int32),
          producer_epoch:wp.array(dtype=wp.int64),
          expected_epoch:wp.array(dtype=wp.int64),
          producer_valid:wp.array(dtype=wp.int32),
          event:wp.array(dtype=wp.int64),
          errors:wp.array(dtype=wp.int32),
          first_stage:wp.array(dtype=wp.int32),
          first_local:wp.array(dtype=wp.int32),
          first_item:wp.array(dtype=wp.int32),
          first_event:wp.array(dtype=wp.int64),
          active:wp.array(dtype=wp.int32),
          valid:wp.array(dtype=wp.int32),
          capacity:int,stage:int):
    w=wp.tid()
    active[w]=0
    valid[w]=0
    if errors[w]!=0 or admission[w]==0:
        return
    if admission[w]!=1:
        _first(w,1,stage,admission[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if count[w]<0:
        _first(w,2,stage,count[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if count[w]>capacity:
        _first(w,3,stage,count[w],capacity,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    if producer_valid[w]!=1 or producer_epoch[w]!=expected_epoch[w] or producer_epoch[w]<wp.int64(0) or expected_epoch[w]<wp.int64(0):
        _first(w,7,stage,producer_valid[w],-1,event[w],errors,first_stage,
               first_local,first_item,first_event)
        return
    active[w]=1

@wp.kernel
def _reset(mask:wp.array(dtype=wp.int32),event:wp.array(dtype=wp.int64),
           errors:wp.array(dtype=wp.int32),first_stage:wp.array(dtype=wp.int32),
           first_local:wp.array(dtype=wp.int32),first_item:wp.array(dtype=wp.int32),
           first_event:wp.array(dtype=wp.int64),active:wp.array(dtype=wp.int32),
           valid:wp.array(dtype=wp.int32),valid_epoch:wp.array(dtype=wp.int64),stage:int):
    w=wp.tid()
    if mask[w]==1:
        errors[w]=0
        first_stage[w]=0
        first_local[w]=0
        first_item[w]=-1
        first_event[w]=wp.int64(-1)
        active[w]=0
        valid[w]=0
        valid_epoch[w]=wp.int64(0)
    elif mask[w]!=0:
        _first(w,1,stage,mask[w],-1,event[w],errors,first_stage,first_local,first_item,first_event)
        active[w]=0
        valid[w]=0

@wp.kernel
def _next_epoch(admission:wp.array(dtype=wp.int32),event:wp.array(dtype=wp.int64),
                errors:wp.array(dtype=wp.int32),first_stage:wp.array(dtype=wp.int32),
                first_local:wp.array(dtype=wp.int32),first_item:wp.array(dtype=wp.int32),
                first_event:wp.array(dtype=wp.int64),epoch:wp.array(dtype=wp.int64),
                active:wp.array(dtype=wp.int32),valid:wp.array(dtype=wp.int32),stage:int):
    w=wp.tid()
    active[w]=0
    valid[w]=0
    if errors[w]!=0 or admission[w]==0:return
    if admission[w]!=1:
        _first(w,1,stage,admission[w],-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    if epoch[w]<wp.int64(0):
        _first(w,7,stage,1,-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    if epoch[w]==wp.int64(9223372036854775807):
        _first(w,7,stage,2,-1,event[w],errors,first_stage,first_local,first_item,first_event)
        return
    epoch[w]=epoch[w]+wp.int64(1)

class SharedProtocol:
    def __init__(self,nworld,capacity,*,device="cuda:0",stream=None):
        if type(nworld)!=int or nworld<1:
            raise ValueError("positive integer world capacity required")
        if type(capacity)!=int or capacity<0:
            raise ValueError("nonnegative static capacity")
        checked_product([nworld,max(1,capacity)])
        self.nworld=nworld
        self.capacity=capacity
        self.device=wp.get_device(device)
        if not self.device.is_cuda: raise ValueError("CUDA required")
        self.stream=stream or wp.get_stream(self.device)
        if self.stream.device!=self.device: raise ValueError("cross-device stream")
        self.active=wp.zeros(nworld,dtype=wp.int32,device=self.device)
        self.valid=wp.zeros(nworld,dtype=wp.int32,device=self.device)
        self.valid_epoch=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        self.forward_epoch=wp.zeros(nworld,dtype=wp.int64,device=self.device)
        for name in ("errors","first_stage","first_local","first_item"):
            setattr(self,name,wp.zeros(nworld,dtype=wp.int32,device=self.device))
        self.first_event=wp.zeros(nworld,dtype=wp.int64,device=self.device)

    def _require(self,arrays):
        from .binding import array_token
        for name,array,dtype in arrays:
            token=array_token(array)
            if array.shape!=(self.nworld,) or token[4]!=dtype or not array.is_contiguous or array.device!=self.device:
                raise ValueError("shared device field dtype/shape/stride/device mismatch: "+name)

    def _state(self):
        self._require([(n,getattr(self,n),"int64" if n in ("first_event","valid_epoch","forward_epoch") else "int32") for n in ("errors","first_stage","first_local","first_item","first_event","active","valid","valid_epoch","forward_epoch")])

    @staticmethod
    def _stage(stage):
        if type(stage)!=int or not 0<stage<=2**31-1:raise ValueError("positive int32 frozen stage ID")

    def _errors(self):
        return [self.errors,self.first_stage,self.first_local,self.first_item,self.first_event]

    def gate(self,admission,count,producer_epoch,expected_epoch,producer_valid,event,*,stage):
        self._stage(stage);self._state()
        self._require([("admission",admission,"int32"),("count",count,"int32"),
            ("producer_epoch",producer_epoch,"int64"),("expected_epoch",expected_epoch,"int64"),
            ("producer_valid",producer_valid,"int32"),("event",event,"int64")])
        wp.launch(_gate,dim=self.nworld,inputs=[
            admission,count,producer_epoch,expected_epoch,producer_valid,event,
            *self._errors(),self.active,self.valid,self.capacity,stage],
            device=self.device,stream=self.stream)

    def reset_at_boundary(self,mask,*,event=None,stage=1):
        self._stage(stage);self._state()
        event=self.forward_epoch if event is None else event
        self._require([("reset_mask",mask,"int32"),("event",event,"int64")])
        wp.launch(_reset,dim=self.nworld,inputs=[mask,event,*self._errors(),self.active,self.valid,self.valid_epoch,stage],
                  device=self.device,stream=self.stream)

    def begin_forward(self,admission,*,event=None,stage=1):
        self._stage(stage);self._state()
        event=self.forward_epoch if event is None else event
        self._require([("admission",admission,"int32"),("event",event,"int64")])
        wp.launch(_next_epoch,dim=self.nworld,
                  inputs=[admission,event,*self._errors(),self.forward_epoch,self.active,self.valid,stage],
                  device=self.device,stream=self.stream)
