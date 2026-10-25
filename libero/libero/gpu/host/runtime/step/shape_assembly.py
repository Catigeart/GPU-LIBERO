"""Internal constructor choices for the resident FP64 physics assembly."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ShapeAssembly:
    """Optional constructor overrides; omitted fields use the base assembly.

    This is a local construction argument. It owns no resources, registration,
    admission, or mutable runtime state.
    """

    dynamics_factory: Callable | None = None
    post_factory: Callable | None = None
    cold_factories: Callable | None = None
    foundation_type: Callable | None = None
    forward_type: Callable | None = None
    controller_type: Callable | None = None
    euler_type: Callable | None = None
    runtime_factory: Callable | None = None
    constraint_workspace_type: Callable | None = None
