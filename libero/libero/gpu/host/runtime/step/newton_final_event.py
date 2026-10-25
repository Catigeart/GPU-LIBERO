"""Independent cold final-Newton evidence; no bad publication/error override.

Only successful original empty/finalize exits produce completed=1. A finite
check rejection at finalize is recorded separately as rejected_final_entry;
intermediate errors cannot masquerade as a completed native acceleration.
"""
import libero.gpu.precision as scalar
import warp as wp
from libero.gpu.host.runtime.newton import live_copy_api as original
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})

@wp.kernel(enable_backward=False)
def _final_event(qacc:wp.array(dtype=scalar.wp_real,ndim=2), empty:wp.array(dtype=wp.int32),
        newton_world:wp.array(dtype=wp.int32), solver_exit:wp.array(dtype=wp.int32),
        error:wp.array(dtype=wp.int32), finalize_status:wp.array(dtype=wp.int32),
        attempted:wp.array(dtype=wp.int64), candidate:wp.array(dtype=scalar.wp_real,ndim=2),
        stamp:wp.array(dtype=wp.int64), completed:wp.array(dtype=wp.int32),
        origin:wp.array(dtype=wp.int32), stage:wp.array(dtype=wp.int32),
        rejected_final_entry:wp.array(dtype=wp.int32), nv:int):
    w=wp.tid()
    completed[w]=0;origin[w]=0;stage[w]=0;stamp[w]=attempted[w]
    rejected_final_entry[w]=0
    if empty[w]==1 and error[w]==0:
        completed[w]=1;origin[w]=1;stage[w]=1
    elif newton_world[w]==1 and solver_exit[w]==1:
        stage[w]=2
        if error[w]==0 and finalize_status[w]==0:
            completed[w]=1;origin[w]=1
        elif finalize_status[w]==3:
            # Entry witness only: original GPU finalize did NOT complete.
            # This cannot trigger warning/reset without separate source proof.
            rejected_final_entry[w]=1
    if completed[w]==1 or rejected_final_entry[w]==1:
        for j in range(nv):candidate[w,j]=qacc[w,j]


def _token(a):return (id(a),a.ptr,a.shape,a.dtype,str(a.device))

class FinalEventAdapter(original.ColdNewtonRuntimeAdapter):
    def __init__(self,workspace,*,rules,solver_factory):
        super().__init__(workspace,rules=rules,solver_factory=solver_factory)
        original_factory=self._solver_factory
        owner=self
        class EventSolver(original_factory):
            def __init__(self,ws):
                super().__init__(ws)
                self.candidate=wp.zeros(ws.qacc.shape,dtype=scalar.wp_real,device=ws.device)
                self.candidate_attempt=wp.zeros(ws.nworld,dtype=wp.int64,device=ws.device)
                self.candidate_completed=wp.zeros(ws.nworld,dtype=wp.int32,device=ws.device)
                self.candidate_origin=wp.zeros(ws.nworld,dtype=wp.int32,device=ws.device)
                self.source_stage=wp.zeros(ws.nworld,dtype=wp.int32,device=ws.device)
                self.rejected_final_entry=wp.zeros(ws.nworld,dtype=wp.int32,device=ws.device)
                self._event_roots=(self.candidate,self.candidate_attempt,self.candidate_completed,
                    self.candidate_origin,self.source_stage,self.rejected_final_entry,owner.n_attempt)
                self._event_tokens=tuple(_token(a) for a in self._event_roots)
            def launch_device(self):
                roots=(self.candidate,self.candidate_attempt,self.candidate_completed,
                    self.candidate_origin,self.source_stage,self.rejected_final_entry,owner.n_attempt)
                if tuple(_token(a) for a in roots)!=self._event_tokens:raise ValueError("final event owner changed")
                super().launch_device()
                w=self.ws
                wp.launch(_final_event,w.nworld,inputs=[w.qacc,w.empty,w.newton_world,w.solver_exit,
                    w.err_code,w.s08d_status,owner.n_attempt,self.candidate,self.candidate_attempt,
                    self.candidate_completed,self.candidate_origin,self.source_stage,
                    self.rejected_final_entry,w.qacc.shape[1]],device=w.device)
        self._solver_factory=EventSolver


def prepare_final_event_runtime(**kwargs):
    adapter = original.prepare_live_copy_runtime(adapter_type=FinalEventAdapter, **kwargs)
    adapter.final_event=adapter.solver
    return adapter
