"""Opt-in CUDA Newton integration, preserving the batch-three adapters."""
from pathlib import Path
import warp as wp
from libero.gpu.host.assembly.pipeline.factory import prepare_fused_physics as previous
from libero.gpu.host.runtime.newton.persistent_cuda import PersistentCudaCore


def prepare_fused_physics(*args,cuda_build_root=None,**kwargs):
    bundle=previous(*args,**kwargs)
    try:
        root=Path(cuda_build_root) if cuda_build_root else Path.cwd()/'persistent-newton-build'
        core=PersistentCudaCore(bundle.newton.solver,root).install()
        # Replace the cold conditional graph as well, so recovery uses the
        # same solver. Capture records device launches without executing them.
        loop=bundle.runtime
        with wp.ScopedCapture(stream=loop.stream,force_module_load=True) as capture:
            wp.capture_if(loop.recovery.reset_mask,on_true=loop._reset_forward,stream=loop.stream)
        loop._reset_forward_graph=capture.graph
        bundle.persistent_cuda_core=core
        bundle.cuda_newton_manifest=core.manifest
        return bundle
    except BaseException:
        bundle.close()
        raise
