"""Resolve public backend choices without importing a physical engine."""
from libero.gpu import precision as scalar


def resolve_engine(engine='auto', precision=None):
    if engine not in ('auto', 'native', 'mjwarp', 'cuda_fp32'):
        raise ValueError('physics_engine must be auto, native, mjwarp or cuda_fp32')
    if precision is not None and precision not in ('fp32', 'float32', 'fp64', 'float64'):
        raise ValueError('precision must be fp32 or fp64')
    bits = scalar.bits if precision is None else (32 if precision in ('fp32', 'float32') else 64)
    if engine == 'auto':
        return 'cuda_fp32' if bits == 32 else 'native'
    if engine == 'mjwarp' and precision in ('fp64', 'float64'):
        raise ValueError('MJWarp requires FP32; use physics_engine="native" for FP64')
    if engine == 'cuda_fp32' and precision in ('fp64','float64'):
        raise ValueError('Independent cuda_fp32 requires FP32')
    return engine
