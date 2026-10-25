"""Explicit shape-wide recovery callbacks with retryable cleanup."""
from . import recovery_final_event_environment as original
from .recovery_final_event_factory import prepare_final_event_recovery_physics


def make_boundary_callbacks(bundle, built):
    from .recovery_cleanup_factory_r02 import assemble_prepared
    return original.make_boundary_callbacks(
        bundle, built,
        replacement_factory=getattr(bundle, 'cold_recovery_prepare', prepare_final_event_recovery_physics),
        replacement_assembler=assemble_prepared,
        admission=getattr(bundle, 'domain_admission', None),
        factory_entry=getattr(bundle, 'domain_factory_entry',
            'libero.gpu.host.adapters.environment.production_factory_r02:create'),
        retain_cleanup=True)


_written = original._written
_model_commit = original._model_commit
