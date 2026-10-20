"""Sparse actual BADQACC event evidence, no physics/source error mutation."""
import libero.gpu.precision as scalar
import warp as wp
from .recovery_runtime_r02 import CurrentWrittenRecoveringRuntime
from .shape_runtime import _token
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})

@wp.kernel(enable_backward=False)
def _select(active:wp.array(dtype=wp.int32),qa:wp.array(dtype=scalar.wp_real,ndim=2),
        candidate:wp.array(dtype=scalar.wp_real,ndim=2), candidate_attempt:wp.array(dtype=wp.int64),completed:wp.array(dtype=wp.int32),
        origin:wp.array(dtype=wp.int32),stage:wp.array(dtype=wp.int32),
        attempt:wp.array(dtype=wp.int64),written:wp.array(dtype=wp.int64),
        expected:wp.array(dtype=wp.int64),state:wp.array(dtype=wp.int64),
        forward:wp.array(dtype=wp.int64),count:wp.array(dtype=wp.int32),
        slot:wp.array(dtype=wp.int32),overflow:wp.array(dtype=wp.int32),
        metadata:wp.array(dtype=wp.int64,ndim=2),capacity:int,nv:int):
    w=wp.tid()
    slot[w]=-1
    bad=int(-1)
    same=int(1)
    if active[w]!=1:return
    for j in range(nv):
        value=qa[w,j]
        if value!=candidate[w,j]:same=0
        if bad<0 and (wp.isnan(value) or value>scalar.wp_real(1e10) or value<scalar.wp_real(-1e10)):bad=j
    if bad<0:return
    if completed[w]!=1 or origin[w]!=1 or candidate_attempt[w]!=attempt[w]:return
    row=count[w]
    if row>=capacity:
        overflow[w]=1
        return
    slot[w]=row
    count[w]=row+1
    metadata[w*capacity+row,0]=wp.int64(bad)
    metadata[w*capacity+row,1]=attempt[w]
    metadata[w*capacity+row,2]=written[w]
    metadata[w*capacity+row,3]=expected[w]
    metadata[w*capacity+row,4]=state[w]
    metadata[w*capacity+row,5]=forward[w]
    metadata[w*capacity+row,6]=wp.int64(stage[w])
    metadata[w*capacity+row,7]=candidate_attempt[w]
    metadata[w*capacity+row,8]=wp.int64(same)

@wp.kernel(enable_backward=False)
def _copy_f64(source:wp.array(dtype=scalar.wp_real),slot:wp.array(dtype=wp.int32),
        target:wp.array(dtype=scalar.wp_real,ndim=2),width:int,capacity:int):
    w,j=wp.tid()
    row=slot[w]
    if row>=0:target[w*capacity+row,j]=source[w*width+j]

@wp.kernel(enable_backward=False)
def _copy_i32(source:wp.array(dtype=wp.int32),slot:wp.array(dtype=wp.int32),
        target:wp.array(dtype=wp.int32,ndim=2),width:int,capacity:int):
    w,j=wp.tid()
    row=slot[w]
    if row>=0:target[w*capacity+row,j]=source[w*width+j]

class EventRecoveringRuntime(CurrentWrittenRecoveringRuntime):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.capture_events = bool(self.forward.entry_witness)
        if not self.capture_events:
            return
        n=self.forward.n
        if not hasattr(n,"final_event"):raise ValueError("actual final producer event required")
        W=self.euler.capacity
        self.event_capacity=8
        self.event_count=wp.zeros(W,dtype=wp.int32,device=self.stream.device)
        self.event_slot=wp.full(W,-1,dtype=wp.int32,device=self.stream.device)
        self.event_overflow=wp.zeros(W,dtype=wp.int32,device=self.stream.device)
        self.event_metadata=wp.zeros((W*8,9),dtype=wp.int64,device=self.stream.device)
        self.event_sources={name:getattr(self.euler,name).reshape((-1,)) for name in
            ("qpos","qvel","time","ctrl","qacc","qacc_warmstart")}
        self.event_sources["N_candidate"]=n.final_event.candidate.reshape((-1,))
        if self.controller is not None:
            for name,array in self.controller.public_arrays().items():
                if array.dtype==scalar.wp_real:
                    self.event_sources["controller."+name]=array.reshape((-1,))
            for name in ("q0","action"):
                array=getattr(self.controller.controller,name)
                self.event_sources["controller."+name]=array.reshape((-1,))
        self.event_before={name:wp.zeros((W*8,array.size//W),dtype=scalar.wp_real,device=self.stream.device)
            for name,array in self.event_sources.items()}
        self.event_after={name:wp.zeros((W*8,array.size//W),dtype=scalar.wp_real,device=self.stream.device)
            for name,array in self.event_sources.items()}
        self.event_after_euler={name:wp.zeros((W*8,self.event_sources[name].size//W),dtype=scalar.wp_real,device=self.stream.device)
            for name in ("qpos","qvel","time")}
        self.event_warnings=wp.zeros((W*8,8),dtype=wp.int32,device=self.stream.device)
        self.event_lastinfo=wp.zeros((W*8,8),dtype=wp.int32,device=self.stream.device)
        self._event_roots=list(self.event_sources.values())+list(self.event_before.values())+list(self.event_after.values())+list(self.event_after_euler.values())+[
            self.event_count,self.event_slot,self.event_overflow,self.event_metadata,self.event_warnings,self.event_lastinfo]
        self._event_solver=n.final_event
        self._event_tokens=self._event_owner_tokens()

    def _event_owner_tokens(self):
        roots=[self.event_count,self.event_slot,self.event_overflow,self.event_metadata,self.event_warnings,self.event_lastinfo]
        return (tuple(_token(a) for a in roots),tuple((kind,name,_token(a)) for kind,mapping in
            (("sources",self.event_sources),("before",self.event_before),("after",self.event_after),
             ("after_euler",self.event_after_euler)) for name,a in mapping.items()))

    def _copy_event(self,outputs):
        for name,target in outputs.items():
            source=self.event_sources[name]
            wp.launch(_copy_f64,(self.euler.capacity,source.size//self.euler.capacity),inputs=[source,self.event_slot,target,source.size//self.euler.capacity,self.event_capacity],device=self.stream.device,stream=self.stream)

    def _admit_current_acceleration(self):
        super()._admit_current_acceleration()
        if not self.capture_events:
            return
        if self._event_owner_tokens()!=self._event_tokens or self.forward.n.final_event is not self._event_solver:
            raise ValueError("event retained owner drift")
        n=self.forward.n;event=n.final_event
        wp.launch(_select,self.euler.capacity,inputs=[self.recovery.active,self.euler.qacc,event.candidate,event.candidate_attempt,
            event.candidate_completed,event.candidate_origin,event.source_stage,n.n_attempt,n.n_written,
            self.forward.expected_newton,self.state_epoch,self.forward.d.forward_epoch,self.event_count,
            self.event_slot,self.event_overflow,self.event_metadata,8,self.runtime_shape.nv],
            device=self.stream.device,stream=self.stream)
        self._copy_event(self.event_before)

    def _integrate(self):
        # This method runs after the real conditional physical recovery forward.
        # Controller goals/history are only copied, never recalculated here.
        if self.capture_events:
            self._copy_event(self.event_after)
            for source,target in ((self.recovery.warning_number,self.event_warnings),
                                  (self.recovery.warning_lastinfo,self.event_lastinfo)):
                wp.launch(_copy_i32,(self.euler.capacity,8),inputs=[source.reshape((-1,)),self.event_slot,target,8,self.event_capacity],device=self.stream.device,stream=self.stream)
        super()._integrate()
        if self.capture_events:
            self._copy_event(self.event_after_euler)

    def download_recovery_events(self):
        if not self.capture_events:
            raise RuntimeError('recovery event recording requires debug=True')
        wp.synchronize_stream(self.stream)
        return dict(event_count=self.event_count.numpy(),event_overflow=self.event_overflow.numpy(),
            metadata=self.event_metadata.numpy(),warnings=self.event_warnings.numpy(),lastinfo=self.event_lastinfo.numpy(),
            **{"before."+k:v.numpy() for k,v in self.event_before.items()},
            **{"after_forward."+k:v.numpy() for k,v in self.event_after.items()},
            **{"after_euler."+k:v.numpy() for k,v in self.event_after_euler.items()})
