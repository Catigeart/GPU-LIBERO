"""Cold receipt/admission boundaries for explicit independent C construction."""
from contextlib import contextmanager


def prepare_collision_admission(static,*,context=None,options=None):
    from .cold_geometry import CollisionColdAdmission
    return CollisionColdAdmission(static, context=context, options=options)


@contextmanager
def collision_build_receipts_scope():
    # H04 remains globally off. This ABI requests complete receipts explicitly.
    from libero.gpu.host.build.native import strict_build_receipts_scope
    with strict_build_receipts_scope(enabled=True):yield
