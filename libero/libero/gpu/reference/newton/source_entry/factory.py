"""Explicit handwritten candidate; existing production entry is unchanged."""
from pathlib import Path
import warp as wp
from libero.gpu.host.assembly.pipeline.factory import prepare_fused_physics as previous
from libero.gpu.host.runtime.newton.handwritten_cuda import HandwrittenCudaCore


def prepare_fused_physics(*args,cuda_build_root=None,shared_state=True,**kwargs):
    bundle=previous(*args,**kwargs)
    try:
        root=Path(cuda_build_root) if cuda_build_root else Path.cwd()/'handwritten-newton-build'
        core=HandwrittenCudaCore(bundle.newton.solver,root,shared_state=shared_state).install()
        loop=bundle.runtime
        with wp.ScopedCapture(stream=loop.stream,force_module_load=True) as capture:
            wp.capture_if(loop.recovery.reset_mask,on_true=loop._reset_forward,stream=loop.stream)
        loop._reset_forward_graph=capture.graph
        bundle.persistent_cuda_core=core;bundle.cuda_newton_manifest=core.manifest
        return bundle
    except BaseException:
        bundle.close();raise
