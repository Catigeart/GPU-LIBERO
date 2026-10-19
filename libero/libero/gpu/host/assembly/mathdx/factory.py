"""Opt-in device-library full Cholesky; retained incremental updates."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import hashlib
from pathlib import Path
import warp as wp
from libero.gpu.host.assembly.pipeline.factory import prepare_fused_physics as previous
from libero.gpu.host.runtime.newton.handwritten_cuda import HandwrittenCudaCore


def prepare_fused_physics(*args,cuda_build_root=None,mathdx_root=None,diagnostic_profile="compatible",defer_cuda_core=False,newton_execution_config=None,newton_algorithm_options=None,**kwargs):
    from libero.gpu.host.runtime.newton.algorithm_options import normalize_newton_algorithm_options
    algorithm_options=normalize_newton_algorithm_options(newton_algorithm_options)
    bundle=previous(*args,newton_diagnostic_profile=diagnostic_profile,**kwargs)
    try:
        root=Path(cuda_build_root) if cuda_build_root else Path.cwd()/'cusolverdx-newton-build'
        core=None
        if not defer_cuda_core:
            core=HandwrittenCudaCore(bundle.newton.solver,root,
                cholesky_backend='cusolverdx',mathdx_root=mathdx_root,
                diagnostic_profile=diagnostic_profile,execution_config=newton_execution_config,
                **algorithm_options).install()
        loop=bundle.runtime
        if not defer_cuda_core:
            with wp.ScopedCapture(stream=loop.stream,force_module_load=True) as capture:
                wp.capture_if(getattr(loop,'reset_predicate',loop.recovery.reset_mask),on_true=loop._reset_forward,stream=loop.stream)
            loop._reset_forward_graph=capture.graph
        if core is not None:
            bundle.persistent_cuda_core=core;bundle.cuda_newton_manifest=core.manifest
        root=(GPU_ROOT / 'host/runtime/newton')
        paths=[Path(__file__),(GPU_ROOT / 'host/assembly/mathdx/__init__.py'),(GPU_ROOT / 'host/runtime/newton/handwritten_cuda.py'),(GPU_ROOT / 'host/runtime/newton/persistent_cuda.py'),
               (GPU_ROOT / 'reference/newton/newton_linesearch/kernel.cu'),*sorted(((GPU_ROOT / 'cuda/newton')).glob('*.cuh')),
               (GPU_ROOT / 'cuda/newton/potrf_dx_bridge.cu')]
        for path in paths:bundle.fusion_source_manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return bundle
    except BaseException:
        bundle.close();raise
