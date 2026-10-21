"""Select storage metadata for shared pointer-only CUDA bindings.

The independent owner never imports Warp. The reference owner retains its
authentic array types so existing validation and alias checks stay effective.
No numerical launch or fallback is selected here.
"""


def array_api(owner):
    if getattr(owner,'_cuda_native',False):
        from libero.gpu.host.backends.cuda_fp32 import device
        return device
    import warp
    return warp


def type_api(owner):
    if getattr(owner,'_cuda_native',False):
        from libero.gpu.host.backends.cuda_fp32 import constants
        return constants
    from mujoco_warp._src import types
    return types


def collision_table(owner):
    if getattr(owner,'_cuda_native',False):
        from libero.gpu.host.backends.cuda_fp32.constants import MJ_COLLISION_TABLE
        return MJ_COLLISION_TABLE
    from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE
    return MJ_COLLISION_TABLE
