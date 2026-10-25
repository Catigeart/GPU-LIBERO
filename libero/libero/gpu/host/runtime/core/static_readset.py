"""Owned bytes-backed NumPy readsets; ordinary readonly arrays are insufficient."""
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np

def immutable_array(value):
    copied=np.asarray(value,order='C')
    if copied.dtype.hasobject: raise ValueError('object readsets cannot be frozen as bytes')
    return np.frombuffer(copied.tobytes(order='C'),dtype=copied.dtype).reshape(copied.shape)

def _token(array):
    return (id(array),array.dtype.str,array.dtype.descr if array.dtype.fields else None,
            tuple(array.shape),tuple(array.strides),bool(array.flags.c_contiguous),
            bool(array.flags.writeable),int(array.__array_interface__['data'][0]))

def _bytes_owner(array):
    seen=set();value=array
    while isinstance(value,np.ndarray):
        if id(value) in seen: return None
        seen.add(id(value));value=value.base
    return value if type(value) is bytes else None

class StaticReadsetSeal:
    """Digest encoding comes from the original consumer, never a new schema.

    Replacing arrays, mutating shape/dtype/strides, or ordinary mutable storage
    uses the original strict hash. Only retained bytes-backed arrays reuse it.
    This seals host static data, never GPU in-place content or upload readiness.
    """
    def __init__(self, arrays, identity, digest):
        self.identity=identity;self.digest=digest
        self.rows=tuple((k,a,_token(a),_bytes_owner(a)) for k,a in arrays.items())
        self.enabled=all(owner is not None and not a.flags.writeable for _,a,_,owner in self.rows)
    def valid(self, arrays, identity):
        return (self.enabled and identity==self.identity and
                tuple(arrays)==tuple(k for k,_,_,_ in self.rows) and
                all(arrays[k] is a and _token(a)==token and _bytes_owner(a) is owner
                    for k,a,token,owner in self.rows))
    def current_digest(self, arrays, identity, strict):
        return self.digest if self.valid(arrays,identity) else strict()


def native_field_digest(name, array):
    """Native metadata encoding, checked once when importing a model snapshot."""
    import hashlib
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(name.encode('utf-8') + b'\0')
    h.update(a.dtype.str.encode('ascii') + b'\0')
    h.update(str(a.shape).encode('ascii') + b'\0')
    h.update(a.tobytes())
    return h.hexdigest()


class PreparedModelArrays(Mapping):
    """One detached immutable host snapshot shared by a model's components."""
    def __init__(self, arrays, metadata, spec):
        self.identity = spec.identity_key
        self._digests = dict(metadata['array_sha256'])
        fields = {}
        for name, value in arrays.items():
            a = immutable_array(value)
            if native_field_digest(name, a) != self._digests.get(name):
                raise ValueError('native field digest differs: ' + name)
            fields[name] = a
        self._fields = MappingProxyType(fields)
        self._tokens = {name: _token(a) for name, a in fields.items()}
        self.select(metadata, spec, ())

    def __getitem__(self, name):
        a = self._fields[name]
        if _token(a) != self._tokens[name]:
            raise ValueError('prepared native field layout changed: ' + name)
        return a

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def select(self, metadata, spec, fields):
        if (spec.identity_key != self.identity or metadata['model_sha256'] != spec.model_sha256 or
                metadata['static_arrays_sha256'] != spec.static_arrays_sha256):
            raise ValueError('prepared native model identity differs')
        if any(metadata['dimensions'].get(k) != v for k, v in spec.dimensions.as_dict().items()):
            raise ValueError('native dimensions differ')
        result = {}
        for name in fields:
            if metadata['array_sha256'].get(name) != self._digests.get(name):
                raise ValueError('prepared native field metadata differs: ' + name)
            result[name] = self[name]
        return result


def native_arrays(arrays, metadata, spec, fields):
    """Standalone callers import a snapshot; prepared components borrow it."""
    fields = tuple(fields)
    if not isinstance(arrays, PreparedModelArrays):
        arrays = PreparedModelArrays({name: arrays[name] for name in fields}, metadata, spec)
    return arrays.select(metadata, spec, fields)
