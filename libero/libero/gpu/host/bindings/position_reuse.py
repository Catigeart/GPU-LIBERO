"""Versioned position reuse for the immutable-model native batch owner only.

No input snapshots, hashes, host predicates or solver-history shortcuts. Each
prefix kernel has an entry guard, so a miss cannot alter a cached world's
scratch or public metadata. Velocity, force and Newton still run every time.
"""
from pathlib import Path
from types import MethodType
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from .reuse import _position_prefix, _enqueue, _step1


class EpochPositionReuse:
    position_only=True

    def __init__(self,bundle,root,guard):
        self.forward=f=bundle.runtime.forward
        self.stream=f.stream
        self.work=guard
        W=f.capacity;d=f.d;sw=f.s.workspace
        with wp.ScopedStream(self.stream):
            self.saved=wp.zeros(W,dtype=wp.int64,device=d.device)
            self.cached=wp.zeros(W,dtype=wp.int32,device=d.device)
            self.hit=wp.zeros(W,dtype=wp.int32,device=d.device)
            self.any=wp.zeros(1,dtype=wp.int32,device=d.device)
            self.math=wp.zeros(W,dtype=wp.uint64,device=d.device)
            self.consumed=wp.zeros(W,dtype=wp.uint64,device=d.device)
        # store() advances this only after a successful, reusable prefix.
        # It is also the persistent Newton coefficient version, so it remains
        # live in production; math is the existing counter alias for reports.
        self.generation=self.math
        source=GPU_ROOT/'cuda/runtime/position_reuse.cuh'
        self.plans={}
        def plan(name,args,worlds=W):
            p=NativePlan('position_reuse_'+name,[source],
                [('position_reuse::'+name,args,'lane0')],worlds,Path(root)/name)
            self.plans[name]=p
            return p
        self.check=plan('check',[d.active,d.first_error,d.state_epoch,self.saved,
            self.cached,sw.status,sw.committed,f.c.workspace.fe_code,
            d.stages.position.factor.warning_count,f.c.workspace.finalize.warnings,self.work,self.hit,
            d.forward_epoch,d.valid['position'],d.written['position'],f.c_mask,
            f.transport.active,self.consumed])
        self.predicate=plan('predicate',[self.work,W,self.any],worlds=1)
        self.store=plan('store',[self.work,d.first_error,d.valid['position'],sw.status,
            sw.committed,f.c.workspace.fe_code,d.stages.position.factor.warning_count,
            f.c.workspace.finalize.warnings,d.state_epoch,self.saved,self.cached,self.math])
        with wp.ScopedCapture(stream=self.stream) as cap:self.branch()
        self.graph=cap.graph

    def miss_body(self):
        _position_prefix(self.forward)
        self.store.launch(self.stream)

    def branch(self):
        wp.capture_if(self.any,on_true=self.miss_body,stream=self.stream)

    def enqueue(self):
        self.check.launch(self.stream)
        self.predicate.launch(self.stream)
        if self.stream.is_capturing:self.branch()
        else:wp.capture_launch(self.graph,stream=self.stream)


def install(bundle,root,guard):
    f=bundle.runtime.forward
    cache=EpochPositionReuse(bundle,root,guard)
    f._dataflow_reuse=cache
    f.enqueue=MethodType(_enqueue,f)
    f.enqueue_step1=MethodType(_step1,f)
    if bundle.controller is not None:
        c=bundle.controller
        c.forward_refresh=f.enqueue
        c._callbacks=(c.forward_refresh,c.gather,c.live_bias)
    bundle.cuda_position_reuse_plans=cache.plans
    return dict(installed=True,mode='state_epoch',per_world=True,
        reused='kinematics/mass/factor/collision/S-position',
        uncached='velocity/reference/bias/force/Newton/warmstart/post',
        key='owned state_epoch; immutable model and mocap; guarded prefix kernels',
        snapshot_bytes=0,plans={k:p.manifest for k,p in cache.plans.items()})
