"""Metadata contracts for the GPU backend; no physics or source certification."""
from .contracts import BufferBinding, BufferSpec, ContractError, DeviceWorldState, FirstError, GraphBinding, HostBoundaryPolicy, ModelDimensions, ModelGeneration, ModelSpec, ParameterUpdate, Publication, StateWrite

from .registry import Admission, CapabilityRegistry, KernelCandidate
from .census_registry import describe_census_record

__all__ = [name for name in globals() if not name.startswith('_')]




