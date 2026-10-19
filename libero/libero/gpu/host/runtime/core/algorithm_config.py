"""Frozen cold configuration for equivalent CUDA algorithm implementations.

Thresholds are experimental choices, not measured performance winners. Solver
tolerances, iteration limits, public outputs and publication rules remain owned
by the existing mathematical contract.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from numbers import Integral, Real


def _integer(name, value, *, minimum=0, maximum=2147483647, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(name + ' must be an integer' + (' or None' if optional else ''))
    if not minimum <= value <= maximum:
        raise ValueError(name + ' is outside its supported integer range')
    return int(value)


def _boolean(name, value):
    if not isinstance(value, bool):
        raise TypeError(name + ' must be bool')
    return value


def _policy(cls, value, name):
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(name + ' must be a frozen policy or mapping')
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError('unknown ' + name + ' options: ' + repr(sorted(unknown, key=str)))
    return cls(**dict(value))


@dataclass(frozen=True)
class NewtonAlgorithmOptions:
    cone_hessian_strategy: str = 'auto'
    cone_batch_threshold: int | None = None
    rank_rebuild_threshold: int | None = None
    newton_islands: str = 'off'

    def __post_init__(self):
        if self.cone_hessian_strategy not in ('rank', 'batch', 'auto'):
            raise ValueError('cone_hessian_strategy must be rank, batch or auto')
        if self.newton_islands not in ('off', 'exact'):
            raise ValueError('newton_islands must be off or exact')
        for name in ('cone_batch_threshold', 'rank_rebuild_threshold'):
            object.__setattr__(self, name, _integer(name, getattr(self, name), optional=True))


@dataclass(frozen=True)
class CollisionAlgorithmOptions:
    temporal: bool = False
    absolute_padding: float = 0.005
    relative_padding: float = 0.01
    max_age: int = 32
    max_cache_bytes: int = 8 * 1024**2

    def __post_init__(self):
        _boolean('temporal', self.temporal)
        for name in ('absolute_padding', 'relative_padding'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise TypeError(name + ' must be a finite real number')
            object.__setattr__(self, name, float(value))
        if self.absolute_padding <= 0 or not 0 <= self.relative_padding <= 1:
            raise ValueError('temporal padding domain')
        object.__setattr__(self, 'max_age', _integer('max_age', self.max_age, minimum=1))
        object.__setattr__(self, 'max_cache_bytes',
                           _integer('max_cache_bytes', self.max_cache_bytes, minimum=1, maximum=2**63-1))


@dataclass(frozen=True)
class PostAlgorithmOptions:
    contact_index_mode: str = 'auto'
    radix_min_contacts: int = 256
    contact_topology_cache: bool = True

    def __post_init__(self):
        if self.contact_index_mode not in ('auto', 'scan', 'radix'):
            raise ValueError('contact_index_mode must be auto, scan or radix')
        object.__setattr__(self, 'radix_min_contacts',
                           _integer('radix_min_contacts', self.radix_min_contacts))
        _boolean('contact_topology_cache', self.contact_topology_cache)


@dataclass(frozen=True)
class AlgorithmOptions:
    """Nested policy consumed explicitly by prepare_cuda_physics.

    A mapping has the same ``newton``, ``collision``, ``post`` keys and nested
    field names. Defaults match the individual installers. ``baseline()``
    selects R07 N/C/Post options; ``experimental()`` requests exact islands
    and temporal broadphase, subject to their cold admission and fallback.
    """
    newton: NewtonAlgorithmOptions = field(default_factory=NewtonAlgorithmOptions)
    collision: CollisionAlgorithmOptions = field(default_factory=CollisionAlgorithmOptions)
    post: PostAlgorithmOptions = field(default_factory=PostAlgorithmOptions)
    collision_schedule: str = 'auto'
    collision_workers: int = 128
    forward_reuse: str = 'auto'

    def __post_init__(self):
        if self.collision_schedule not in ('auto','fused','compact','parallel'):
            raise ValueError('collision_schedule must be auto/fused/compact/parallel')
        if self.forward_reuse not in ('auto','off','epoch','exact'):
            raise ValueError('forward_reuse must be auto/off/epoch/exact')
        object.__setattr__(self,'collision_workers',_integer('collision_workers',self.collision_workers,minimum=1))
        for name, cls in (('newton', NewtonAlgorithmOptions),
                          ('collision', CollisionAlgorithmOptions), ('post', PostAlgorithmOptions)):
            object.__setattr__(self, name, _policy(cls, getattr(self, name), name))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def baseline(cls):
        return cls(newton=NewtonAlgorithmOptions(cone_hessian_strategy='rank', rank_rebuild_threshold=0),
                   post=PostAlgorithmOptions(contact_index_mode='scan', contact_topology_cache=False),
                   collision_schedule='fused',forward_reuse='off')

    @classmethod
    def experimental(cls):
        return cls(newton=NewtonAlgorithmOptions(newton_islands='exact'),
                   collision=CollisionAlgorithmOptions(temporal=True))


def normalize_algorithm_options(options=None, *, newton_algorithm_options=None):
    """Copy/validate input before constructing CUDA owners or compiling code."""
    if options is not None and newton_algorithm_options is not None:
        raise ValueError('algorithm_options and newton_algorithm_options cannot both be supplied')
    result = AlgorithmOptions() if options is None else _policy(AlgorithmOptions, options, 'algorithm')
    if newton_algorithm_options is not None:
        result = AlgorithmOptions(newton=_policy(NewtonAlgorithmOptions, newton_algorithm_options, 'newton'),
                                  collision=result.collision, post=result.post)
    return result
