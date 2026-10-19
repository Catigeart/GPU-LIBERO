"""Cold typed S head bindings; availability is not numeric authentication.

This module does not import Warp, read device values, or allocate device views.
The original producers retain every array owner. A consumer must additionally
validate current row coverage and numeric equivalence on device for each launch.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


_FIELDS = (
    'source_counts', 'source_ncon', 'compact_heads', 'compact_count',
    'compact_valid', 'head_type', 'head_id', 'head_dim', 'head_mu',
    'head_friction', 'valid',
)
_FLOAT_FIELDS = frozenset(('head_mu', 'head_friction'))


def _token(array):
    return (id(array), int(array.ptr), tuple(array.shape), tuple(array.strides),
            array.dtype, str(array.device), bool(array.is_contiguous), int(array.size))


def _shape(name, worlds, rows):
    if name == 'source_counts':
        return (worlds, 5)
    if name == 'head_friction':
        return (worlds, 5, rows)
    if name in ('source_ncon', 'compact_count', 'compact_valid', 'valid'):
        return (worlds,)
    return (worlds, rows)


def _size(shape):
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def _contiguous_strides(shape, itemsize):
    strides = []
    for dimension in reversed(shape):
        strides.append(itemsize)
        itemsize *= dimension
    return tuple(reversed(strides))


@dataclass(frozen=True, eq=False)
class HeadMetadata(Mapping):
    """Sealed original owners with explicit row/ordinal/world pitches.

    ``source_counts`` has fields (ne, nf, nlimit, nefc, nnzJ). ``compact_count``
    is G, not the number n of written rows. Numeric head entries are valid only
    at type7 ordinals and friction components below d-1. Capacity R remains the
    plane pitch when n/G change. ``valid`` is only producer availability.
    """
    owner: object = field(repr=False)
    nworld: int
    rows: int
    contacts: int
    nv: int
    _arrays: Mapping = field(repr=False)
    _tokens: Mapping = field(repr=False)
    _provider: object = field(repr=False)
    _validate_owner: object = field(repr=False)
    _array_type: object = field(repr=False)
    _dtypes: Mapping = field(repr=False)
    _dimensions: tuple = field(repr=False)
    world_strides: Mapping

    @classmethod
    def create(cls, *, owner, nworld, rows, contacts, nv, provider,
               validate_owner, array_type, int32_dtype, float64_dtype):
        if cls is not HeadMetadata:
            raise TypeError('exact HeadMetadata type required')
        dimensions = (nworld, rows, contacts, nv)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError('positive integer head metadata dimensions required')
        if not callable(provider) or not callable(validate_owner):
            raise TypeError('current owner provider and validation callback required')
        validate_owner()
        arrays = dict(provider())
        if set(arrays) != set(_FIELDS):
            raise ValueError('complete head metadata field set required')
        dtypes = {'int32': int32_dtype, 'float64': float64_dtype}
        for name, array in arrays.items():
            cls._check_array(name, array, nworld, rows, array_type, dtypes)
        devices = {str(array.device) for array in arrays.values()}
        if len(devices) != 1:
            raise ValueError('head metadata requires one device')
        strides = {name: _size(_shape(name, nworld, rows)[1:]) for name in _FIELDS}
        result = cls(owner, *dimensions, MappingProxyType(arrays),
                     MappingProxyType({k: _token(v) for k, v in arrays.items()}),
                     provider, validate_owner, array_type, MappingProxyType(dtypes),
                     dimensions, MappingProxyType(strides))
        result.validate()
        return result

    @staticmethod
    def _check_array(name, array, worlds, rows, array_type, dtypes):
        logical = _shape(name, worlds, rows)
        size = _size(logical)
        floating = name in _FLOAT_FIELDS
        dtype = dtypes['float64' if floating else 'int32']
        itemsize = 8 if floating else 4
        if (not isinstance(array, array_type) or array.dtype != dtype or
                not array.is_contiguous or int(array.size) != size or
                tuple(array.shape) not in (logical, (size,))):
            raise ValueError('typed contiguous head metadata layout required: ' + name)
        if int(array.ptr) <= 0 or int(array.ptr) % itemsize:
            raise ValueError('head metadata requires a naturally aligned resident pointer: ' + name)
        if tuple(array.strides) != _contiguous_strides(tuple(array.shape), itemsize):
            raise ValueError('head metadata physical strides changed: ' + name)

    def __getitem__(self, name):
        return self._arrays[name]

    def __iter__(self):
        return iter(_FIELDS)

    def __len__(self):
        return len(_FIELDS)

    def validate(self):
        """Reject owner/layout drift without reading any numeric device value."""
        if type(self) is not HeadMetadata:
            raise TypeError('exact HeadMetadata type required')
        if (self.nworld, self.rows, self.contacts, self.nv) != self._dimensions:
            raise ValueError('head metadata dimensions changed')
        self._validate_owner()
        current = dict(self._provider())
        if set(current) != set(_FIELDS) or set(self._arrays) != set(_FIELDS):
            raise ValueError('head metadata field set changed')
        expected_strides = {name: _size(_shape(name, self.nworld, self.rows)[1:])
                            for name in _FIELDS}
        if dict(self.world_strides) != expected_strides:
            raise ValueError('head metadata world strides changed')
        for name, array in current.items():
            self._check_array(name, array, self.nworld, self.rows,
                              self._array_type, self._dtypes)
            if array is not self._arrays[name] or _token(array) != self._tokens[name]:
                raise ValueError('head metadata owner/layout changed: ' + name)

    def manifest(self):
        """Logical layout only; no input values, readiness claim, or pointer dump."""
        self.validate()
        return dict(schema='sealed-S-head-metadata-v1', nworld=self.nworld,
                    rows=self.rows, contacts=self.contacts, nv=self.nv,
                    source_count_fields=['ne', 'nf', 'nlimit', 'nefc', 'nnzJ'],
                    fields={name: dict(logical_shape=list(_shape(name, self.nworld, self.rows)),
                        dtype='float64' if name in _FLOAT_FIELDS else 'int32',
                        physical_strides_bytes=list(self._arrays[name].strides),
                        world_stride_elements=self.world_strides[name]) for name in _FIELDS},
                    numeric_input_copy=False,
                    admission='original N admission and per-launch current equivalence required',
                    valid_scope='producer availability only; G is not a public row extent')
