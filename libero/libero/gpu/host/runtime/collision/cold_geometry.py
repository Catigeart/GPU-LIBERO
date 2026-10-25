"""Resident collision model checks, independent of source/build certificates.

Static values, model context, capacities and close state remain checked. Loaded
library ABI, device buffers and graph ownership belong to the plan/binding layer.
"""
import hashlib
import json
import struct
import numpy as np
from ..core.static_readset import immutable_array, StaticReadsetSeal


def _encoded(value):
    """Strict typed byte encoding used both before and after sealing."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:raise ValueError('object static field is unsupported')
        meta=json.dumps((value.dtype.str,value.dtype.descr if value.dtype.fields else None,
            tuple(value.shape)),separators=(',',':')).encode()
        raw=np.ascontiguousarray(value).tobytes()
        return b'a'+struct.pack('<QQ',len(meta),len(raw))+meta+raw
    if isinstance(value,np.generic):return b'n'+_encoded(np.asarray(value))
    if value is None:return b'z'
    if type(value) is bool:return b'b'+bytes([value])
    if type(value) is int:return b'i'+_encoded(str(value))
    if type(value) is float:return b'f'+struct.pack('<d',value)
    if type(value) is str:
        raw=value.encode();return b's'+struct.pack('<Q',len(raw))+raw
    if type(value) is bytes:return b'r'+struct.pack('<Q',len(value))+value
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('static/context keys must be strings')
        return b'd'+struct.pack('<Q',len(value))+b''.join(_encoded(k)+_encoded(value[k]) for k in sorted(value))
    if isinstance(value,(tuple,list)):
        return (b't' if type(value) is tuple else b'l')+struct.pack('<Q',len(value))+b''.join(map(_encoded,value))
    raise TypeError('opaque static/context field: '+type(value).__name__)


def strict_static_digest(value):return hashlib.sha256(_encoded(value)).hexdigest()


def _freeze(value,immutable_array):
    if isinstance(value,np.ndarray):return immutable_array(value)
    if isinstance(value,dict):return {k:_freeze(v,immutable_array) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_freeze(v,immutable_array) for v in value)
    if isinstance(value,list):return [_freeze(v,immutable_array) for v in value]
    _encoded(value)
    return value


def _arrays(value,prefix='',result=None):
    result={} if result is None else result
    if isinstance(value,np.ndarray):result[prefix]=value
    elif isinstance(value,dict):
        for k in sorted(value):_arrays(value[k],prefix+'/'+k,result)
    elif isinstance(value,(tuple,list)):
        for k,v in enumerate(value):_arrays(v,prefix+'/'+str(k),result)
    return result


def _array_skeleton(value):
    if isinstance(value,np.ndarray):return ('array',value.dtype.str,value.dtype.descr if value.dtype.fields else None,tuple(value.shape))
    if isinstance(value,dict):return {k:_array_skeleton(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_array_skeleton(v) for v in value)
    if isinstance(value,list):return [_array_skeleton(v) for v in value]
    return value


class CollisionColdAdmission:
    def __init__(self, static, *, context=None, options=None):
        self.static = _freeze(static, immutable_array)
        self.context_source = {} if context is None else context
        self.options = {} if options is None else dict(options)
        self.context_digest = strict_static_digest(self.context_source)
        self.skeleton_digest = strict_static_digest(_array_skeleton(self.static))
        self.static_digest = strict_static_digest(self.static)
        self.plans = ()
        self.workspace = None
        self.sealed = False
        self.revoked = False
        self.static_seal = None
        self._seal_identity = None
        self._plan_owners = ()
        self.complete_context = (isinstance(self.context_source, dict) and all(
            k in self.context_source for k in
            ('model_identity', 'spec_identity', 'generation', 'field_permissions')))

    def attach(self, workspace):
        if self.workspace is not None:
            raise ValueError('admission owner is one-shot')
        self.workspace = workspace
        workspace._collision_cold_admission = self
        return workspace

    def _identity(self):
        ws = self.workspace
        dimensions = {} if ws is None else {k: getattr(ws, k) for k in
            ('nworld', 'ngeom', 'nbody', 'slots', 'call_capacity', 'pair_capacity',
             'stage_slots', 'req_slots', 'contact_capacity') if hasattr(ws, k)}
        return strict_static_digest(dict(context=self.context_source,
            dimensions=dimensions, device=None if ws is None else str(ws.device),
            options=self.options, static_skeleton=_array_skeleton(self.static)))

    def validate(self, *, require_installed=True):
        if self.revoked:
            raise ValueError('collision cold admission was revoked')
        if require_installed and not self.sealed:
            raise ValueError('collision cold admission is not sealed')
        if strict_static_digest(self.context_source) != self.context_digest:
            raise ValueError('collision model/spec/context changed')
        if strict_static_digest(_array_skeleton(self.static)) != self.skeleton_digest:
            raise ValueError('collision static metadata/scalars changed')
        identity = self._identity()
        current = (self.static_seal.current_digest(_arrays(self.static), identity,
            strict=lambda: strict_static_digest(self.static)) if self.static_seal is not None
            else strict_static_digest(self.static))
        if current != self.static_digest:
            raise ValueError('collision host static readset changed')
        if self.sealed and (identity != self._seal_identity or
                len(self.plans) != len(self._plan_owners) or
                any(a is not b for a, b in zip(self.plans, self._plan_owners))):
            raise ValueError('collision installed plan/context changed')
        return True

    def seal(self, plans):
        if self.sealed:
            raise ValueError('collision admission seal is one-shot')
        self.validate(require_installed=False)
        plans = tuple(plans)
        if not plans:
            raise ValueError('actual installed plans required')
        self.plans = self._plan_owners = plans
        self._seal_identity = self._identity()
        if self.complete_context:
            self.static_seal = StaticReadsetSeal(
                _arrays(self.static), self._seal_identity, self.static_digest)
        self.sealed = True
        self.validate()
        return self.manifest()

    def manifest(self):
        return dict(schema='resident-collision-model-v1', sealed=self.sealed,
            host_static_digest=self.static_digest,
            host_static_reuse=bool(self.static_seal and self.static_seal.enabled),
            complete_context=self.complete_context, installed_plans=len(self.plans),
            gpu_static_write_or_upload_permission=False)

    def revoke(self, reason='resident arena released'):
        """The owning arena releases GPU work before closing this model binding."""
        self.revoked = True
        self.sealed = False
        self.plans = self._plan_owners = ()
