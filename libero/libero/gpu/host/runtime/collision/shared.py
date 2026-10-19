"""Retained device edges for the collision chain.

These objects inspect host allocation metadata only. They do not read poses,
counts, epochs, or contacts back to the host. F supplies its device epoch gate
as the active input and owns any packed-to-SoA transport.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import warp as wp

FLOAT_FIELDS = {
    'dist': (0, 1), 'pos': (1, 3), 'frame': (4, 9),
    'includemargin': (13, 1), 'friction': (14, 5), 'solref': (19, 2),
    'solreffriction': (21, 2), 'solimp': (23, 5), 'mu': (28, 1), 'H': (29, 36),
}
INT_FIELDS = {'dim': 0, 'geom1': 1, 'geom2': 2, 'exclude': 3, 'efc_address': 4}


class DeviceBindingError(ValueError):
    pass


def array_token(array):
    if not isinstance(array, wp.array) or not array.device.is_cuda:
        raise DeviceBindingError('actual CUDA Warp array required')
    # Warp represents genuine zero-length optional CUDA arguments by ptr=None.
    # Preserve their object/dtype/shape identity without inventing allocation.
    pointer = int(array.ptr or 0)
    if pointer == 0 and (array.size != 0 or int(array.capacity) != 0):
        raise DeviceBindingError('nonempty CUDA argument has null allocation')
    return (id(array), pointer, tuple(array.shape), tuple(array.strides),
            array.dtype, str(array.device), int(array.capacity),
            bool(array.is_contiguous), id(getattr(array, '_ref', None)))


def _path(root, path):
    if not isinstance(path, str) or not path or any(not p for p in path.split('.')):
        raise DeviceBindingError('explicit source attribute path required')
    parents = [root]
    value = root
    for part in path.split('.'):
        try:
            value = value[part] if isinstance(value, dict) else getattr(value, part)
        except (KeyError, AttributeError, TypeError) as exc:
            raise DeviceBindingError('source attribute path is missing: ' + path) from exc
        parents.append(value)
    return parents


def _allocation(array):
    """Require Warp's retained parent chain and authentic allocation deleter."""
    array_token(array)
    chain, seen = [], set()
    while isinstance(array, wp.array):
        if id(array) in seen:
            raise DeviceBindingError('cyclic allocation parent chain')
        seen.add(id(array))
        chain.append(array)
        parent = getattr(array, '_ref', None)
        if parent is None:
            break
        if not isinstance(parent, wp.array):
            raise DeviceBindingError('native external allocation needs separate ABI proof')
        array = parent
        array_token(array)
    allocator = getattr(array, '_allocator', None)
    deleter = getattr(array, 'deleter', None)
    if (allocator is None or deleter is None
            or getattr(deleter, '__self__', None) is not allocator
            or int(array.ptr) <= 0 or int(array.capacity) <= 0):
        raise DeviceBindingError('actual owning Warp allocation required')
    return array, chain


def _origin_token(root):
    return (array_token(root), id(root._allocator), id(root.deleter),
            id(root.deleter.__self__))


def _extent(array):
    if array.size == 0 or any(int(x) <= 0 for x in array.shape):
        raise DeviceBindingError('nonempty physical input required')
    width = 8 if array.dtype == wp.float64 else 4 if array.dtype == wp.int32 else None
    if width is None or any(int(s) <= 0 for s in array.strides):
        raise DeviceBindingError('scalar float64/int32 positive strides required')
    return width + sum((int(n) - 1) * int(s) for n, s in zip(array.shape, array.strides))


def _consumer_modules(ws):
    return {'chain': ws, 'tree': ws.tree, 'dispatch': ws.dispatch,
            'detect': ws.detect, 'finalize': ws.finalize,
            'params': ws.finalize.params}


def _argument_arrays(ws):
    # Exact array objects supplied by this chain and its four consumers.
    # No traversal of the externally retained D owner or native build metadata.
    modules = _consumer_modules(ws)
    result = {}
    for prefix, obj in modules.items():
        for name, value in vars(obj).items():
            if isinstance(value, wp.array):
                result[prefix + '.' + name] = value
            elif isinstance(value, dict):
                for key, array in value.items():
                    if isinstance(array, wp.array):
                        result[prefix + '.' + name + '.' + str(key)] = array
    return result



def _handwritten_plans(ws):
    detector = getattr(ws.detect, '_cuda_detector_plan', None)
    complete = getattr(ws, '_cuda_collision_complete', None)
    schedule=getattr(ws,'_cuda_collision_query_schedule','fused')
    if detector is None or (schedule=='fused' and complete is None):
        raise DeviceBindingError('handwritten_only backend is not cold-installed')
    plans = (detector, complete) if schedule=='fused' else (detector,)
    if schedule in ('compact','parallel'):
        compact = ws._cuda_collision_compact
        plans += (compact['front'], compact['tail'])
    return plans


def _plan_exports(plan, detector):
    return ((plan.library.launch_collision_detect_tiled, plan.library.launch_collision_detect_compact_producer,plan.library.launch_collision_detect_parallel)
        if detector else (plan.library.launch,))


def _method_token(method):
    function = getattr(method, '__func__', method)
    return (id(getattr(method, '__self__', None)), id(function), id(getattr(function, '__code__', None)),
        id(getattr(function, '__defaults__', None)), tuple(id(v) for v in (getattr(function, '__defaults__', None) or ())),
        id(getattr(function, '__kwdefaults__', None)), tuple((k,id(v)) for k,v in sorted((getattr(function, '__kwdefaults__', None) or {}).items())),
        tuple(id(c.cell_contents) for c in (getattr(function, '__closure__', None) or ())))


def _handwritten_methods(ws):
    return tuple((id(plan),tuple((name,_method_token(getattr(plan,name))) for name in
        (('validate','launch','launch_compact_producer','launch_parallel_producer') if i == 0 else ('validate_bindings','launch'))))
        for i,plan in enumerate(_handwritten_plans(ws)))


def _admission_methods(admission):
    seal = admission.static_seal
    return (id(admission),id(admission.files),tuple(_method_token(getattr(admission,name)) for name in
        ('validate','authority_token','_identity','_verify_files')),
        tuple(_method_token(getattr(admission.files,name)) for name in ('digest','verification')),
        None if seal is None else (id(seal),_method_token(seal.valid),_method_token(seal.current_digest)))


def _native_retained(ws):
    if getattr(ws, '_detector_backend', 'legacy') == 'legacy':
        native = ws.detect.native
        return (native, native['library'], native['library'].launch_native)
    plans = _handwritten_plans(ws)
    return tuple(value for plan in plans for value in (plan, plan.library)) + tuple(
        function for i, plan in enumerate(plans) for function in _plan_exports(plan, i == 0)) + tuple(
        owner for plan in plans for owner in plan.owners)


def _native_execution(ws):
    if getattr(ws, '_detector_backend', 'legacy') == 'legacy':
        return ws.detect.native['execution_binding']
    if getattr(ws.detect, '_cuda_detector_plan', None) is None or getattr(ws, '_cuda_collision_complete', None) is None:
        return {'abi': 'handwritten_only-v1', 'cold_pending': True, 'plans': []}
    return {'abi': 'handwritten_only-v1', 'cold_pending': False, 'plans': [p.manifest for p in _handwritten_plans(ws)]}


def _native_token(ws):
    if getattr(ws, '_detector_backend', 'legacy') == 'handwritten_only':
        admission = getattr(ws, '_collision_cold_admission', None)
        if admission is None:
            raise DeviceBindingError('handwritten_only requires its retained cold admission')
        admission.validate()
        if ws.detect.native is not None:
            raise DeviceBindingError('handwritten_only unexpectedly retained baseline native')
        records = []
        for i, plan in enumerate(_handwritten_plans(ws)):
            if i == 0:
                plan.validate()
                parameter_abi = tuple((name, ctype.__name__, ct.sizeof(ctype)) for name, ctype in plan.params._fields_)
                params_owner = id(plan.device_params), int(plan.device_params.ptr)
            else:
                plan.validate_bindings()
                parameter_abi = tuple((name, ctype.__name__, ct.sizeof(ctype)) for name, ctype in plan.params._fields_)
                params_owner = id(plan.params), ct.addressof(plan.params)
            # Actual loaded library/export, pointer owners, and compiled source identity.
            owners = tuple((id(a), int(a.ptr or 0), tuple(a.shape), tuple(a.strides),
                a.dtype.__name__, str(a.device), id(getattr(a, '_ref', None)))
                for a in plan.owners)
            exports = tuple((id(function), int(ct.cast(function, ct.c_void_p).value),
                tuple((id(t), t.__name__) for t in (function.argtypes or ())),
                id(function.restype), getattr(function.restype, '__name__', None))
                for function in _plan_exports(plan, i == 0))
            methods = tuple((name, _method_token(getattr(plan, name))) for name in
                (('validate', 'launch', 'launch_compact_producer') if i == 0 else ('validate_bindings', 'launch')))
            records.append((id(plan), id(plan.library), int(plan.library._handle),
                exports, methods,
                parameter_abi, params_owner, bytes(plan.params).hex(), owners, str(plan.device),
                hashlib.sha256(json.dumps(plan.manifest, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()))
        return ('handwritten_only-v1', tuple(records))
    native = ws.detect.native
    library = native['library']
    execution = native['execution_binding']
    function = library.launch_native
    # This is the loaded export used by native.launch_native, rather than a
    # provider's descriptive module label. No kernel or physics call occurs.
    return (id(native), id(library), int(library._handle), id(function),
            int(ct.cast(function, ct.c_void_p).value),
            tuple(native['adapter_abi']), native['device'],
            native['build_path'], native['build_sha256'], id(execution),
            hashlib.sha256(json.dumps(execution, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())

class DeviceInputBinding:
    """One immutable source graph, retained for eager and captured launches."""

    NAMES = ('xpos', 'xmat', 'xipos', 'ximat', 'active')

    def __init__(self, ws, owner, paths, stream):
        if owner is None or not isinstance(paths, dict) or set(paths) != set(self.NAMES):
            raise DeviceBindingError('one retained source root and all five paths required')
        if not isinstance(stream, wp.Stream) or not stream.device.is_cuda:
            raise DeviceBindingError('explicit CUDA stream required')
        device = wp.get_device(ws.device)
        if stream.device != device:
            raise DeviceBindingError('stream/device mismatch')
        self.owner, self.paths, self.stream = owner, dict(paths), stream
        self._stream_token = (id(stream), str(stream.device), int(stream.cuda_stream))
        self.source = {}
        self.flat = {}
        self.records = {}
        self._retained = [owner, stream]
        for name in self.NAMES:
            parents = _path(owner, paths[name])
            array = parents[-1]
            array_token(array)
            dtype = wp.int32 if name == 'active' else wp.float64
            count = 1 if name == 'active' else (ws.ngeom if name in ('xpos', 'xmat') else ws.nbody)
            width = 1 if name == 'active' else (3 if name in ('xpos', 'xipos') else 9)
            shapes = {(ws.nworld * count * width,)}
            if name != 'active':
                shapes.update(((ws.nworld, count * width), (ws.nworld, count, width)))
            if (array.dtype != dtype or array.device != device
                    or tuple(array.shape) not in shapes or not array.is_contiguous):
                raise DeviceBindingError('source dtype/shape/contiguity/device mismatch: ' + name)
            allocation, chain = _allocation(array)
            offset = int(array.ptr) - int(allocation.ptr)
            extent = _extent(array)
            itemsize = 4 if dtype == wp.int32 else 8
            if offset < 0 or offset % itemsize or offset + extent > int(allocation.capacity):
                raise DeviceBindingError('actual input allocation range mismatch: ' + name)
            flat = array.flatten()
            if int(flat.ptr) != int(array.ptr) or flat.dtype != dtype or not flat.is_contiguous:
                raise DeviceBindingError('flatten changed source allocation: ' + name)
            self.source[name], self.flat[name] = array, flat
            self.records[name] = {
                'path_ids': tuple(id(p) for p in parents),
                'source_token': array_token(array), 'allocation': allocation,
                'allocation_token': _origin_token(allocation),
                'view_ids': tuple(id(p) for p in chain), 'offset': offset,
            }
            self._retained.extend(parents + chain + [allocation, flat])
        # Install only after every source edge has passed. No partial binding
        # survives a failure above.
        ws.xpos, ws.xmat, ws.active = self.flat['xpos'], self.flat['xmat'], self.flat['active']
        ws.tree.xipos, ws.tree.ximat = self.flat['xipos'], self.flat['ximat']
        ws.detect.xpos, ws.detect.xmat = ws.xpos, ws.xmat
        ws.tree.geom_pos, ws.tree.geom_mat = ws.xpos, ws.xmat
        ws.dispatch.xpos, ws.dispatch.xmat = ws.xpos, ws.xmat
        self._arguments = _argument_arrays(ws)
        self._argument_tokens = {name: array_token(a) for name, a in self._arguments.items()}
        self._retained.extend(self._arguments.values())
        self._scalars = (ws.nworld, ws.ngeom, ws.nbody, ws.npair_static,
                         ws.nexclude, ws.slots, ws.call_capacity, ws.pair_capacity,
                         ws.stage_slots, ws.req_slots, ws.contact_capacity)
        self.graphs = []
        self._modules = _consumer_modules(ws)
        self._module_scalars = self._scalar_state(ws)
        self._retained.extend(self._modules.values())
        self._native = None if getattr(ws, '_detector_backend', 'legacy') == 'handwritten_only' else _native_token(ws)
        self._admission = getattr(ws, '_collision_cold_admission', None)
        self._admission_methods = None if self._admission is None else _admission_methods(self._admission)
        self._admission_authority = None
        self._native_methods = None
        if self._native is not None:
            self._retained.extend(_native_retained(ws))

    @staticmethod
    def _scalar_state(ws):
        return {prefix + '.' + name: (type(value), value)
                for prefix, obj in _consumer_modules(ws).items()
                for name, value in vars(obj).items()
                if not name.startswith('_') and type(value) in (int, float, bool, str)}

    def seal_backend(self, ws):
        if self._native is not None or ws._launch_started or self.graphs:
            raise DeviceBindingError('backend seal is one-shot before launch/capture')
        self.validate(ws, self.stream, allow_unsealed=True)
        token = _native_token(ws)  # requires actual complete installed plans
        self._native = token
        self._native_methods = _handwritten_methods(ws)
        self._admission_methods = _admission_methods(self._admission)
        self._admission_authority = self._admission.authority_token()
        self._retained.extend(_native_retained(ws))
        self._arguments = _argument_arrays(ws)
        self._argument_tokens = {name: array_token(a) for name, a in self._arguments.items()}
        self._retained.extend(self._arguments.values())
        self._module_scalars = self._scalar_state(ws)
        return self.manifest(ws)

    def validate(self, ws, stream, *, allow_unsealed=False):
        pending = self._native is None
        if pending and (not allow_unsealed or ws._launch_started or self.graphs):
            raise DeviceBindingError('handwritten_only backend must be sealed before launch/capture')
        if getattr(ws, '_detector_backend', 'legacy') == 'handwritten_only':
            admission = getattr(ws, '_collision_cold_admission', None)
            # Check actual owner/method/proof replacement before any mutable callback.
            if admission is None or admission is not self._admission:
                raise DeviceBindingError('handwritten_only requires its retained cold admission')
            current_methods = _admission_methods(admission)
            if current_methods[:4] != self._admission_methods[:4] or (not pending and current_methods != self._admission_methods):
                raise DeviceBindingError('collision admission methods/files/seal replaced')
            if not pending and (admission.authority_token() != self._admission_authority or
                    _handwritten_methods(ws) != self._native_methods):
                raise DeviceBindingError('collision admission authority/plan methods changed')
            admission.validate(require_installed=not pending)
        if (id(stream), str(stream.device), int(stream.cuda_stream)) != self._stream_token:
            raise DeviceBindingError('bound stream replaced')
        scalars = (ws.nworld, ws.ngeom, ws.nbody, ws.npair_static,
                   ws.nexclude, ws.slots, ws.call_capacity, ws.pair_capacity,
                   ws.stage_slots, ws.req_slots, ws.contact_capacity)
        modules = _consumer_modules(ws)
        if (set(modules) != set(self._modules)
                or any(obj is not self._modules[name] for name, obj in modules.items())
                or (not pending and self._scalar_state(ws) != self._module_scalars)
                or (pending and any(self._scalar_state(ws).get(name) != token for name, token in self._module_scalars.items()))):
            raise DeviceBindingError('collision model/consumer/scalar replaced')
        if scalars != self._scalars:
            raise DeviceBindingError('bound model dimensions/capacities replaced')
        if not pending and _native_token(ws) != self._native:
            raise DeviceBindingError('actual native library/function/build replaced')
        for name in self.NAMES:
            parents = _path(self.owner, self.paths[name])
            array = parents[-1]
            record = self.records[name]
            allocation, chain = _allocation(array)
            if (tuple(id(p) for p in parents) != record['path_ids']
                    or array_token(array) != record['source_token']
                    or allocation is not record['allocation']
                    or _origin_token(allocation) != record['allocation_token']
                    or tuple(id(p) for p in chain) != record['view_ids']):
                raise DeviceBindingError('source owner/array/view replaced: ' + name)
        actual = _argument_arrays(ws)
        if ((not pending and set(actual) != set(self._argument_tokens))
                or any(name not in actual or array_token(actual[name]) != token for name, token in self._argument_tokens.items())):
            raise DeviceBindingError('collision consumer array/alias replaced')
        return True

    def manifest(self, ws, *, allow_unsealed=False):
        self.validate(ws, self.stream, allow_unsealed=allow_unsealed)
        return {
            'source_root_object_id': id(self.owner), 'stream': self._stream_token,
            'epoch': 'actual resident F gate; no host epoch is inferred here',
            'actual_native_token': self._native,
            'inputs': {name: {
                'path': self.paths[name], 'source_object_id': id(self.source[name]),
                'allocation_object_id': id(self.records[name]['allocation']),
                'source_pointer': int(self.source[name].ptr),
                'consumer_pointer': int(self.flat[name].ptr),
                'allocation_pointer': int(self.records[name]['allocation'].ptr),
                'allocation_bytes': int(self.records[name]['allocation'].capacity),
                'offset_bytes': self.records[name]['offset'],
                'dtype': self.source[name].dtype.__name__,
                'shape': tuple(self.source[name].shape),
                'strides_bytes': tuple(self.source[name].strides),
                'keepers': self.records[name]['path_ids'] + self.records[name]['view_ids'],
            } for name in self.NAMES},
        }


class ContactFieldViews:
    """Real strided fields over C's retained contact slabs, without GPU copies.

    These views are not contiguous SoA. A consumer requiring SoA must use an
    explicit device transport with the authentic read/write masks.
    """

    def __init__(self, ws):
        self.owner = ws
        self.contact_f, self.contact_i = ws.finalize.contact_f, ws.finalize.contact_i
        self.ncon = ws.finalize.ncon
        self.nworld, self.capacity = ws.nworld, ws.contact_capacity
        for array, dtype, total in ((self.contact_f, wp.float64, self.nworld * self.capacity * 65),
                                    (self.contact_i, wp.int32, self.nworld * self.capacity * 5),
                                    (self.ncon, wp.int32, self.nworld)):
            if array.dtype != dtype or array.shape != (total,) or not array.is_contiguous:
                raise DeviceBindingError('canonical contact slab/count schema mismatch')
            allocation, _ = _allocation(array)
            offset = int(array.ptr) - int(allocation.ptr)
            if offset < 0 or offset + _extent(array) > int(allocation.capacity):
                raise DeviceBindingError('contact slab exceeds its real allocation')
        self.fields = {}
        for name, (offset, width) in FLOAT_FIELDS.items():
            shape = (self.nworld, self.capacity) if width == 1 else (self.nworld, self.capacity, width)
            strides = (self.capacity * 520, 520) if width == 1 else (self.capacity * 520, 520, 8)
            self.fields[name] = self._view(self.contact_f, offset * 8, shape, strides, wp.float64)
        for name, offset in INT_FIELDS.items():
            self.fields[name] = self._view(self.contact_i, offset * 4,
                (self.nworld, self.capacity), (self.capacity * 20, 20), wp.int32)
        self._tokens = (array_token(self.contact_f), array_token(self.contact_i), array_token(self.ncon))
        self._origins = tuple(_origin_token(_allocation(a)[0]) for a in
                              (self.contact_f, self.contact_i, self.ncon))
        self._field_tokens = {name: array_token(a) for name, a in self.fields.items()}

    @staticmethod
    def _view(parent, offset, shape, strides, dtype):
        array_token(parent)
        view = wp.array(ptr=int(parent.ptr) + offset, shape=shape, strides=strides,
                        dtype=dtype, device=parent.device, copy=False)
        view._ref = parent
        if _extent(view) + offset > int(parent.capacity):
            raise DeviceBindingError('contact field exceeds actual slab allocation')
        return view

    def validate(self):
        ws = self.owner
        if (ws.nworld != self.nworld or ws.contact_capacity != self.capacity
                or (array_token(ws.finalize.contact_f), array_token(ws.finalize.contact_i),
                    array_token(ws.finalize.ncon)) != self._tokens
                or tuple(_origin_token(_allocation(a)[0]) for a in
                         (self.contact_f, self.contact_i, self.ncon)) != self._origins
                or set(self.fields) != set(self._field_tokens)
                or any(array_token(a) != self._field_tokens[name]
                       for name, a in self.fields.items())):
            raise DeviceBindingError('contact allocation/view replaced')
        for name, array in self.fields.items():
            parent = self.contact_f if name in FLOAT_FIELDS else self.contact_i
            if getattr(array, '_ref', None) is not parent:
                raise DeviceBindingError('contact field keeper replaced')
        return True

    def manifest(self):
        self.validate()
        return {'layout': 'C split slabs float64[W,C,65], int32[W,C,5]; fields are strided',
            'nworld': self.nworld, 'contact_capacity': self.capacity,
            'slabs': {'float': array_token(self.contact_f)[:4], 'int': array_token(self.contact_i)[:4]},
            'ncon_pointer': int(self.ncon.ptr),
            'fields': {name: {'pointer': int(a.ptr), 'dtype': a.dtype.__name__,
                'shape': tuple(a.shape), 'strides_bytes': tuple(a.strides),
                'offset_bytes': int(a.ptr) - int((self.contact_f if name in FLOAT_FIELDS else self.contact_i).ptr),
                'contiguous': bool(a.is_contiguous)}
                for name, a in self.fields.items()}}


def resident_metadata(ws):
    """Actual reachable Warp allocation inventory, without any device reads.

    Native module constant/local/shared bytes and CUDA Graph/context/cache
    resources are different domains. This inventory is not a total VRAM bound.
    """
    arrays = _argument_arrays(ws)
    roots, entries = {}, {}
    public_ids = {id(a) for a in ws.outputs.values()}
    for name, array in arrays.items():
        array_token(array)
        root, chain, seen = array, [], set()
        while isinstance(root, wp.array):
            if id(root) in seen:
                raise DeviceBindingError('cyclic resident allocation parent')
            seen.add(id(root)); chain.append(root)
            parent = getattr(root, '_ref', None)
            if parent is None:
                break
            if not isinstance(parent, wp.array):
                raise DeviceBindingError('external allocation lacks authentic ABI')
            root = parent
        if root.size:
            actual_root, _ = _allocation(array)
            if actual_root is not root:
                raise DeviceBindingError('resident root inconsistency')
        width = wp.types.type_size_in_bytes(array.dtype)
        extent = 0 if array.size == 0 else width + sum(
            (int(n) - 1) * int(st) for n, st in zip(array.shape, array.strides))
        offset = int(array.ptr or 0) - int(root.ptr or 0)
        if offset < 0 or offset + extent > int(root.capacity):
            raise DeviceBindingError('resident view exceeds authentic allocation')
        key = str(id(root))
        if key not in roots:
            roots[key] = {'object_id': id(root), 'pointer': int(root.ptr or 0),
                'capacity_bytes': int(root.capacity), 'device': str(root.device),
                'allocator_object_id': id(getattr(root, '_allocator', None)),
                'owns_deallocation': getattr(root, 'deleter', None) is not None,
                'reachable_paths': [], 'has_public_output_alias': False}
        roots[key]['reachable_paths'].append(name)
        roots[key]['has_public_output_alias'] |= id(array) in public_ids
        entries[name] = {'object_id': id(array), 'pointer': int(array.ptr or 0),
            'dtype': array.dtype.__name__, 'shape': tuple(array.shape),
            'strides_bytes': tuple(array.strides), 'contiguous': bool(array.is_contiguous),
            'physical_view_bytes': extent, 'root_id': key,
            'allocation_offset_bytes': offset, 'keeper_chain': [id(a) for a in chain]}
    return {'schema': 'p06c-actual-reachable-warp-resident-v1',
        'nworld': ws.nworld, 'arrays': entries, 'roots': roots,
        'unique_root_bytes': sum(r['capacity_bytes'] for r in roots.values()),
        'native_execution_binding': _native_execution(ws),
        'scope': 'actual prepared C reachable Warp owners only; no counts/contacts/epochs downloaded; not whole F capacity, CUDA context/cache/Graph or total VRAM guarantee'}
