"""Resident forward-input admission over one batched SharedProtocol(W) gate.

The dynamics backend owns the only forward clock and all physical computation.
Euler and forward share the exact qpos/qvel allocations. This adapter never
uploads dynamic intermediates, calls a CPU forward, or adds a refresh forward.
"""
import libero.gpu.precision as scalar
from pathlib import Path
import hashlib
import warp as wp
from ..core.static_readset import immutable_array, StaticReadsetSeal
from .device import SharedProtocol
from .binding import array_token, allocation_origin, origin_token
from .schema import checked_product

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

def _origin_record(a):
    root, chain = allocation_origin(a)
    return (root, tuple(chain), origin_token(root))

def _check_origin(a, record):
    root, chain = allocation_origin(a)
    if root is not record[0] or len(chain) != len(record[1]) or any(x is not y for x,y in zip(chain,record[1])) or origin_token(root) != record[2]:
        raise ValueError('resident allocation owner/refchain replaced')

@wp.kernel(enable_backward=False)
def _copy_admission(source: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32), world: int):
    target[world] = source[0]

@wp.kernel(enable_backward=False)
def _copy_event(source: wp.array(dtype=wp.int64), target: wp.array(dtype=wp.int64), world: int):
    target[world] = source[0]

class FoundationForwardAdapter:
    """Prepare once, dispatch at the caller's original explicit forward times.

    backend must implement validate_bindings(), begin_forward(mask),
    launch_position(stream=None), launch_velocity(stream=None). Its qpos, qvel, state_epoch,
    stream and device roots remain stable; it owns forward_epoch and publishes
    actual position/velocity valid/written only after its real device work.
    source_records are exact source licences, not a replacement source guard.
    expected_state_epoch and state_valid must be independently prepared device
    arrays owned by the step state producer. No host publication is accepted.
    """
    def __init__(self, backend, *, mask, expected_state_epoch, state_valid,
                 nq, nv, source_records, stream):
        if type(nq) is not int or type(nv) is not int or nq <= 0 or nv <= 0:
            raise ValueError('positive model nq/nv required')
        if not source_records:
            raise ValueError('pinned original backend/model source licence required')
        self.source_records = tuple((str(Path(r['path']).resolve()), r['sha256']) for r in source_records)
        for path, digest in self.source_records:
            if len(digest) != 64 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('original runtime source drift: ' + path)
        self._source_records = self.source_records
        self.backend, self._backend, self.stream = backend, backend, stream
        self.nq, self.nv = nq, nv
        if backend.stream is not stream or not stream.device.is_cuda:
            raise ValueError('retained actual CUDA backend stream required')
        self.state_epoch = backend.state_epoch
        self.expected_state_epoch, self.state_valid = expected_state_epoch, state_valid
        self.qpos, self.qvel = backend.qpos, backend.qvel
        self.nworld = self.state_epoch.size
        checked_product([self.nworld, max(nq, nv)])
        if self.state_epoch.shape != (self.nworld,) or self.nworld < 1:
            raise ValueError('actual world epoch shape')
        for a in (self.state_epoch, expected_state_epoch):
            self._layout(a, wp.int64, (self.nworld,))
        self._layout(state_valid, wp.int32, (self.nworld,))
        for a, extent in ((self.qpos, nq), (self.qvel, nv)):
            if a.shape not in ((self.nworld, extent), (self.nworld * extent,)):
                raise ValueError('model state world-major shape')
            self._layout(a, scalar.wp_real, a.shape)
        p, e = self.state_epoch, expected_state_epoch
        if max(int(p.ptr), int(e.ptr)) < min(int(p.ptr)+p.size*8, int(e.ptr)+e.size*8):
            raise ValueError('independent intended epoch aliases actual producer')
        self._roots = tuple((n, getattr(self, n), array_token(getattr(self, n))) for n in
                            ('qpos','qvel','state_epoch','expected_state_epoch','state_valid'))
        self._stream = (id(stream), int(stream.cuda_stream), int(stream.device.context))
        self._backend_methods = tuple((n, getattr(type(backend), n),
                                       getattr(type(backend),n).__code__, getattr(type(backend),n).__defaults__) for n in
                                      ('validate_bindings','begin_forward','launch_position','launch_velocity'))
        with wp.ScopedStream(stream):
            self.admitted = wp.zeros(self.nworld, dtype=wp.int32, device=stream.device)
            self.errors = wp.zeros(self.nworld, dtype=wp.int32, device=stream.device)
            self.first_stage = wp.zeros(self.nworld, dtype=wp.int32, device=stream.device)
            self.first_local = wp.zeros(self.nworld, dtype=wp.int32, device=stream.device)
            self.first_item = wp.full(self.nworld, -1, dtype=wp.int32, device=stream.device)
            self.first_event = wp.zeros(self.nworld, dtype=wp.int64, device=stream.device)
            self.count = wp.full(self.nworld, nq, dtype=wp.int32, device=stream.device)
            self.gates = (SharedProtocol(self.nworld, nq, device=str(stream.device), stream=stream),)
        self._admitted, self._count, self._gates = self.admitted, self.count, self.gates
        self._gate_roots = tuple(tuple((n,a,array_token(a)) for n,a in vars(g).items() if isinstance(a,wp.array)) for g in self.gates)
        self._error_roots = tuple((n,getattr(self,n),array_token(getattr(self,n))) for n in
                                  ('errors','first_stage','first_local','first_item','first_event'))
        self._tokens = (array_token(self.admitted), array_token(self.count))
        self._views = ((self.count, self.state_epoch, expected_state_epoch, state_valid),)
        self._view_tokens = tuple(tuple(array_token(a) for a in row) for row in self._views)
        self._layout(mask, wp.int32, (self.nworld,))
        self._mask, self._mask_token = mask, array_token(mask)
        self._mask_views = (mask,)
        self._mask_view_tokens = tuple(array_token(v) for v in self._mask_views)
        self._origins = tuple((a,_origin_record(a)) for a in 
                              [*(a for _,a,_ in self._roots),self._mask,self.admitted,self.count,
                               *(a for row in self._views for a in row),*self._mask_views,
                               *(a for rows in self._gate_roots for _,a,_ in rows),
                               *(a for _,a,_ in self._error_roots)])
        self._begun = False
        self.validate_bindings()

    def _layout(self, a, dtype, shape):
        if not isinstance(a, wp.array) or a.dtype != dtype or a.shape != shape or not a.is_contiguous or a.device != self.stream.device:
            raise ValueError('resident state dtype/shape/stream device mismatch')

    def validate_bindings(self):
        if self.backend is not self._backend or self.source_records != self._source_records:
            raise ValueError('runtime backend/source licence replaced')
        if (id(self.stream), int(self.stream.cuda_stream), int(self.stream.device.context)) != self._stream or self.backend.stream is not self.stream:
            raise ValueError('runtime stream/context replaced')
        for name, method, code, defaults in self._backend_methods:
            if getattr(getattr(self.backend, name), '__func__', None) is not method or method.__code__ is not code or method.__defaults__ is not defaults:
                raise ValueError('qualified backend method replaced: ' + name)
        self.backend.validate_bindings()
        for a, record in self._origins:
            _check_origin(a, record)
        for name, root, token in self._roots:
            if getattr(self, name) is not root or array_token(root) != token:
                raise ValueError('current resident root replaced: ' + name)
            if name in ('qpos','qvel','state_epoch') and getattr(self.backend, name) is not root:
                raise ValueError('backend state allocation changed: ' + name)
        if self.admitted is not self._admitted or self.count is not self._count or self.gates is not self._gates or (array_token(self.admitted), array_token(self.count)) != self._tokens:
            raise ValueError('prepared F admission allocation replaced')
        if array_token(self._mask) != self._mask_token or tuple(array_token(v) for v in self._mask_views) != self._mask_view_tokens:
            raise ValueError('retained caller mask allocation/view changed')
        for row, tokens in zip(self._views, self._view_tokens):
            if tuple(array_token(a) for a in row) != tokens:
                raise ValueError('retained world epoch view changed')
        for name, root, token in self._error_roots:
            if getattr(self,name) is not root or array_token(root) != token:
                raise ValueError('F source error aggregate replaced')
        for gate, roots in zip(self.gates, self._gate_roots):
            for name, root, token in roots:
                if getattr(gate,name) is not root or array_token(root) != token:
                    raise ValueError('prepared original F1 state replaced: ' + name)
            gate._state()
            if gate.stream is not self.stream:
                raise ValueError('F1 gate stream changed')

    def begin_forward(self, mask=None, *, stream=None):
        """Exactly one backend begin for one caller-requested complete forward.

        Only actual state lineage is gated here. Explicit caller mask may be a
        conditional controller refresh; masked worlds are never revived.
        """
        if stream is not None and stream is not self.stream:
            raise ValueError('forward retained stream required')
        self.validate_bindings()
        mask = self._mask if mask is None else mask
        self._layout(mask, wp.int32, (self.nworld,))
        if mask is not self._mask or array_token(mask) != self._mask_token:
            raise ValueError('forward mask must update its retained device allocation')
        gate = self.gates[0]
        count, actual, expected, valid = self._views[0]
        gate.gate(mask, count, actual, expected, valid, actual, stage=101)
        wp.copy(self.admitted,gate.active,stream=self.stream)
        for name, root, _ in self._error_roots:
            wp.copy(root,getattr(gate,name),stream=self.stream)
        self.backend.begin_forward(self.admitted, stream=self.stream)
        self._begun = True
        return self.admitted

    def launch_position(self, *, stream=None):
        self._phase_stream(stream)
        self.backend.launch_position(stream=self.stream)
        return self.backend

    def launch_velocity(self, *, stream=None):
        self._phase_stream(stream)
        self.backend.launch_velocity(stream=self.stream)
        return self.backend

    def _phase_stream(self, stream):
        if stream is not None and stream is not self.stream:
            raise ValueError('forward phase retained stream required')
        if not self._begun:
            raise ValueError('caller must begin its explicit forward first')
        self.validate_bindings()

    def reset_at_boundary(self, mask):
        """Explicit lifecycle reset only; no automatic sticky-error erasure."""
        self._layout(mask, wp.int32, (self.nworld,))
        self.validate_bindings()
        gate = self.gates[0]
        gate.reset_at_boundary(mask, event=self.state_epoch, stage=101)
        for name, root, _ in self._error_roots:
            wp.copy(root,getattr(gate,name),stream=self.stream)

    def source_catalog(self):
        self.validate_bindings()
        return {'sources': self.source_records, 'nworld': self.nworld, 'nq': self.nq, 'nv': self.nv,
                'roots': {n: token for n, _, token in self._roots},
                'forward_clock_owner': 'actual dynamics backend; F does not increment it',
                'hot_host_copies': False, 'extra_forward_calls': False,
                'scope': 'current resident input admission; backend original guards and actual output publication remain mandatory'}
# This holder is a new explicit candidate, not an impersonation of fixed3eea.
# Original body kernels remain unchanged; a family first runs in diagnostic
# registry scope and needs independent numerical/write/source acceptance before
# production. No native MjData or forward is constructed here.
import numpy as np
from libero.gpu.reference.position.body_kinematics.workspace import FrozenModel, STATIC, _read_set_hash


def _native_field_digest(name, array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(name.encode('utf-8') + b'\0')
    h.update(a.dtype.str.encode('ascii') + b'\0')
    h.update(str(a.shape).encode('ascii') + b'\0')
    h.update(a.tobytes())
    return h.hexdigest()


def _pin_native_metadata(metadata, spec, records, metadata_record, fields):
    import json
    path = str(Path(metadata_record['path']).resolve())
    digest = metadata_record['sha256']
    if (path,digest) not in records or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('authoritative native metadata source pin missing')
    original = json.loads(Path(path).read_text())
    for name in ('model_sha256','static_arrays_sha256','dimensions','decoder_pin'):
        if metadata[name] != original[name]:
            raise ValueError('native metadata differs from pinned original: '+name)
    for name in fields:
        if metadata['array_sha256'].get(name) != original['array_sha256'].get(name):
            raise ValueError('native field metadata differs from pinned original: '+name)

def _model_admission(registry, candidate_id, spec, mode, base_spec):
    # The derived descriptor and its exact native readset are never rewritten.
    # Only an independently qualified, explicitly supported generation can
    # transfer production permission from an immutable base identity.
    if base_spec is not None and mode == 'production':
        admission = registry.admit_generation(candidate_id, base_spec, spec,
                                               required_scope='component-runtime')
        if admission.execution_allowed and (admission.acceptance is None or
                admission.acceptance.execution_scope != 'component-runtime'):
            raise ValueError('generation requires component-runtime qualification')
        return admission
    # A finite private/math licence must never become public runtime permission.
    # Diagnostic execution remains available for actual new-model qualification.
    return registry.admit(candidate_id, spec, mode=mode,
                          required_scope='component-runtime' if mode == 'production' else None)

class PreparedKinematicModel(FrozenModel):
    """Count-parametric original K model with an exact native static readset.

    Cold preparation only. metadata is actual model-only decoder output, arrays
    its copied native leaves. registry/model identity/source candidate admission
    is mandatory; a structural diagnostic licence is never numerical acceptance.
    Parameter changes retain base_spec and require exact derived native readsets
    plus independent generation admission; the old base guard is unchanged.
    """
    def __init__(self, arrays, metadata, spec, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', base_spec=None):
        if not source_records:
            raise ValueError('native static decoding/source records required')
        self.source_records = tuple((str(Path(r['path']).resolve()), r['sha256']) for r in source_records)
        for path, digest in self.source_records:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('native static/source bytes drift: ' + path)
        _pin_native_metadata(metadata,spec,self.source_records,metadata_record,STATIC)
        self.spec, self.registry, self.candidate_id, self.mode = spec, registry, candidate_id, mode
        if metadata['model_sha256'] != spec.model_sha256 or metadata['static_arrays_sha256'] != spec.static_arrays_sha256:
            raise ValueError('actual model/static identity differs from structural spec')
        if metadata['decoder_pin']['source_sha256'] not in {digest for _,digest in self.source_records}:
            raise ValueError('original model-only decoder source pin missing')
        if spec.model_sha256 not in {digest for _,digest in self.source_records}:
            raise ValueError('actual model MJB file source pin missing')
        self.base_spec = base_spec
        self._base_identity = None if base_spec is None else base_spec.identity_key
        self.admission = _model_admission(registry,candidate_id,spec,mode,base_spec)
        candidate = registry.candidates[candidate_id]
        if candidate.source_sha256 not in {digest for _,digest in self.source_records}:
            raise ValueError('registered original kernel source pin missing')
        if not self.admission.execution_allowed:
            raise ValueError('kinematic candidate admission: ' + '; '.join(self.admission.reasons))
        self.dims = {name:getattr(spec.dimensions,name) for name in ('nq','nbody','njnt','nmocap')}
        if any(metadata['dimensions'][name] != value for name,value in self.dims.items()):
            raise ValueError('native dimensions differ')
        self.arrays = {}
        for name in STATIC:
            a = np.array(arrays[name], copy=True, order='C')
            if metadata['array_sha256'].get(name) != _native_field_digest(name,a):
                raise ValueError('actual native kinematic field digest differs: ' + name)
            a = immutable_array(a)
            self.arrays[name] = a
        self.kind = 'runtime-native-qualified-candidate'
        self.field_sha256 = _read_set_hash(self.arrays,self.dims)
        self.spec_sha256 = spec.identity_key
        self._identity = (spec.identity_key, self.field_sha256, tuple(sorted(self.dims.items())))
        self._readset_seal = StaticReadsetSeal(self.arrays,tuple(sorted(self.dims.items())),self.field_sha256)
        self._static_validated = False
        self.validate()
        self._static_validated = True

    def validate(self):
        n,a = self.dims,self.arrays
        dimensions=tuple(sorted(n.items()))
        seal=getattr(self,'_readset_seal',None)
        digest=seal.current_digest(a,dimensions,lambda:_read_set_hash(a,n)) if seal is not None else _read_set_hash(a,n)
        if (self.spec.identity_key, digest, dimensions) != self._identity:
            raise ValueError('prepared family kinematic readset replaced')
        if (None if self.base_spec is None else self.base_spec.identity_key) != self._base_identity:
            raise ValueError('immutable generation base replaced')
        admission = _model_admission(self.registry,self.candidate_id,self.spec,self.mode,self.base_spec)
        if not admission.execution_allowed:
            raise ValueError('prepared model independent admission changed')
        if getattr(self,'_static_validated',False) and seal is not None and seal.valid(a,dimensions):
            return
        if n['nq'] < 1 or n['nbody'] < 1 or any(type(v) is not int or v<0 for v in n.values()):
            raise ValueError('kinematic model dimensions')
        checked_product([n['nbody'],9])
        for name,(dtype,shape) in STATIC.items():
            value = a[name]
            if value.dtype != np.dtype(dtype) or value.shape != shape(n) or not value.flags.c_contiguous:
                raise ValueError('kinematic native shape/dtype: ' + name)
            if dtype == 'float64' and not np.isfinite(value).all():
                raise ValueError('nonfinite static kinematic input: ' + name)
        if a['body_parentid'][0] != 0 or np.any(a['body_jntnum']<0):
            raise ValueError('world/negative joint topology')
        if np.any(a['body_mocapid'] < -1) or np.any(a['body_mocapid'] >= n['nmocap']):
            raise ValueError('mocap topology range')
        if np.any(~np.isin(a['jnt_type'],[0,1,2,3])):
            raise ValueError('unsupported original joint type')
        for b in range(1,n['nbody']):
            if not 0 <= a['body_parentid'][b] < b:
                raise ValueError('body tree must be original ordered tree')
            count,offset = int(a['body_jntnum'][b]),int(a['body_jntadr'][b])
            if count and not 0 <= offset < offset+count <= n['njnt']:
                raise ValueError('body joint range')
        for j,kind in enumerate(a['jnt_type']):
            offset = int(a['jnt_qposadr'][j])
            extent = {0:7,1:4,2:1,3:1}[int(kind)]
            if not 0 <= offset < offset+extent <= n['nq']:
                raise ValueError('joint qpos range')
from libero.gpu.reference.position.kinematics_frames.frames import FrameWorkspace, MODEL_FIELDS, FLOAT_FIELDS, _tail_static_sha256


def prepare_frame_arrays(arrays, metadata, spec, nworld):
    """Cold exact native byte validation and lossless integer ABI conversion."""
    dims = spec.dimensions
    b,g,s = dims.nbody,dims.ngeom,dims.nsite
    if min(b,g,s)<1:
        raise ValueError('original frame candidate requires nonempty body/geom/site')
    checked_product([nworld,max(b,g,s),9])
    shapes = {'body_ipos':(b,3),'body_iquat':(b,4),'body_sameframe':(b,),
              'geom_bodyid':(g,),'geom_pos':(g,3),'geom_quat':(g,4),'geom_sameframe':(g,),
              'site_bodyid':(s,),'site_pos':(s,3),'site_quat':(s,4),'site_sameframe':(s,)}
    prepared = {}
    for name in MODEL_FIELDS:
        raw = np.array(arrays[name],copy=True,order='C')
        dtype = np.dtype('float64' if name in FLOAT_FIELDS else 'int32')
        native_types = (np.dtype('uint8'),np.dtype('int32')) if name.endswith('_sameframe') else (dtype,)
        if raw.dtype not in native_types or raw.shape != shapes[name] or _native_field_digest(name,raw) != metadata['array_sha256'].get(name):
            raise ValueError('frame actual native readset differs: ' + name)
        # Original mjtByte sameframe -> original kernel int32 ABI. This is
        # an exact static integer extension after native bytes are pinned.
        a = raw.astype(dtype,copy=True)
        if name in FLOAT_FIELDS and not np.isfinite(a).all():
            raise ValueError('nonfinite original frame static: ' + name)
        a.flags.writeable = False
        prepared[name] = a
    if not np.all(np.isin(prepared['body_sameframe'],(0,1))):
        raise ValueError('body original sameframe range')
    for kind in ('geom','site'):
        if not np.all(np.isin(prepared[kind+'_sameframe'],(0,1,2))):
            raise ValueError('original sameframe range: '+kind)
        ids = prepared[kind+'_bodyid']
        if np.any(ids<0) or np.any(ids>=b):
            raise ValueError('original frame body mapping: '+kind)
    return prepared,b,g,s

class PreparedFrameWorkspace(FrameWorkspace):
    """Explicit generalized static frame candidate using original frame kernels.

    A separate factory, never a change to FrameWorkspace's fixed-model guard.
    Registry production requires exact new model numerical/write/source review.
    The diagnostic path only allocates a source-qualified original-kernel case.
    """
    def __init__(self, arrays, metadata, spec, nworld, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', device='cuda:0', body_inputs=None, base_spec=None):
        if type(nworld) is not int or nworld < 1:
            raise ValueError('positive actual world count required')
        records = tuple((str(Path(r['path']).resolve()),r['sha256']) for r in source_records)
        if not records:
            raise ValueError('actual frame readset source licence required')
        for path,digest in records:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('original frame/native decoder bytes drift: ' + path)
        _pin_native_metadata(metadata,spec,records,metadata_record,MODEL_FIELDS)
        digests = {digest for _,digest in records}
        if metadata['model_sha256'] != spec.model_sha256 or metadata['static_arrays_sha256'] != spec.static_arrays_sha256:
            raise ValueError('frame native model/static identity differs')
        if metadata['decoder_pin']['source_sha256'] not in digests or spec.model_sha256 not in digests:
            raise ValueError('actual frame model/decoder pin absent')
        self.base_spec = base_spec
        self._base_identity = None if base_spec is None else base_spec.identity_key
        self.admission = _model_admission(registry,candidate_id,spec,mode,base_spec)
        if not self.admission.execution_allowed or registry.candidates[candidate_id].source_sha256 not in digests:
            raise ValueError('frame candidate source/model admission refused')
        self.spec,self.registry,self.candidate_id,self.mode = spec,registry,candidate_id,mode
        self.source_records = records
        prepared,b,g,s = prepare_frame_arrays(arrays,metadata,spec,nworld)
        self._frame_configuration = (nworld,b,g,s,str(wp.get_device(device)),spec.identity_key)
        prepared={name:immutable_array(value) for name,value in prepared.items()}
        self._native_static = prepared
        self.tail_static_sha256 = _tail_static_sha256(prepared)
        self._readset_seal=StaticReadsetSeal(prepared,self._frame_configuration,self.tail_static_sha256)
        self.device = wp.get_device(device)
        self.nworld,self.nbody,self.ngeom,self.nsite = nworld,b,g,s
        if body_inputs is not None and set(body_inputs) != {'body_xpos','body_xquat','body_xmat','active_world'}:
            raise ValueError('all four actual body input owners required')
        self._body_inputs = dict(body_inputs) if body_inputs is not None else None
        self.static = {name:wp.array(a.reshape(-1),dtype=scalar.wp_real if name in FLOAT_FIELDS else wp.int32,device=self.device)
                       for name,a in prepared.items()}
        for name,count,width in (('body_xpos',b,3),('body_xquat',b,4),('body_xmat',b,9),
            ('xipos',b,3),('ximat',b,9),('geom_xpos',g,3),('geom_xmat',g,9),
            ('site_xpos',s,3),('site_xmat',s,9),('body_tmpquat',b,4),
            ('geom_tmpquat',g,4),('site_tmpquat',s,4)):
            if body_inputs is not None and name in body_inputs:
                a=body_inputs[name]
                if not isinstance(a,wp.array) or a.dtype!=scalar.wp_real or a.shape!=(nworld*count*width,) or not a.is_contiguous or a.device!=self.device:
                    raise ValueError('actual body frame input dtype/layout/device: '+name)
                _origin_record(a)
                setattr(self,name,a)
            else:
                setattr(self,name,wp.empty(nworld*count*width,dtype=scalar.wp_real,device=self.device))
        if body_inputs is not None:
            a=body_inputs['active_world']
            if not isinstance(a,wp.array) or a.dtype!=wp.int32 or a.shape!=(nworld,) or not a.is_contiguous or a.device!=self.device:
                raise ValueError('actual body active source dtype/layout/device')
            _origin_record(a)
            self.active_world=a
        else:
            self.active_world = wp.empty(nworld,dtype=wp.int32,device=self.device)
        self.status = wp.empty(nworld,dtype=wp.int32,device=self.device)
        self._prepared_bindings = tuple((name,a,array_token(a),_origin_record(a))
            for name,a in vars(self).items() if isinstance(a,wp.array))
        self._static_bindings = tuple((name,a,array_token(a),_origin_record(a)) for name,a in self.static.items())
        self.validate_bindings()

    def validate_bindings(self):
        if (self.nworld,self.nbody,self.ngeom,self.nsite,str(self.device),self.spec.identity_key) != self._frame_configuration or set(self.static) != set(MODEL_FIELDS):
            raise ValueError('prepared frame configuration changed')
        seal=getattr(self,'_readset_seal',None)
        digest=seal.current_digest(self._native_static,self._frame_configuration,lambda:_tail_static_sha256(self._native_static)) if seal is not None else _tail_static_sha256(self._native_static)
        if digest != self.tail_static_sha256:
            raise ValueError('frame native static source changed')
        if (None if self.base_spec is None else self.base_spec.identity_key) != self._base_identity:
            raise ValueError('immutable generation base replaced')
        if not _model_admission(self.registry,self.candidate_id,self.spec,self.mode,self.base_spec).execution_allowed:
            raise ValueError('frame exact model admission changed')
        for name,a,token,origin in self._prepared_bindings:
            if getattr(self,name) is not a or array_token(a) != token:
                raise ValueError('prepared frame field owner changed: '+name)
            _check_origin(a,origin)
        for name,a,token,origin in self._static_bindings:
            if self.static[name] is not a or array_token(a) != token:
                raise ValueError('prepared static frame owner changed: '+name)
            _check_origin(a,origin)

    def launch(self, *, stream=None):
        self.validate_bindings()
        super().launch(stream=stream)
from libero.gpu.reference.observations.camlight.workspace import FrozenCamlightModel, MODEL_FIELDS as CAMLIGHT_FIELDS, INT_FIELDS as CAMLIGHT_INT_FIELDS, _canonical as _camlight_digest


class PreparedCamlightModel(FrozenCamlightModel):
    """Actual native fixed-mode camera/light readset for the original kernel.

    Tracks new model identity explicitly; no impersonation of the old fixed MJB.
    Non-fixed native camera/light modes remain a declared structural limitation.
    """
    def __init__(self, arrays, metadata, spec, *, registry, candidate_id,
                 source_records, metadata_record, mode='production', base_spec=None):
        records = tuple((str(Path(r['path']).resolve()),r['sha256']) for r in source_records)
        if not records:raise ValueError('actual camlight source licence required')
        for path,digest in records:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('camlight/static source bytes drift: '+path)
        _pin_native_metadata(metadata,spec,records,metadata_record,CAMLIGHT_FIELDS)
        digests={digest for _,digest in records}
        if metadata['model_sha256'] != spec.model_sha256 or metadata['static_arrays_sha256'] != spec.static_arrays_sha256:
            raise ValueError('camlight actual model identity differs')
        if metadata['decoder_pin']['source_sha256'] not in digests or spec.model_sha256 not in digests:
            raise ValueError('camlight actual model/decoder pin absent')
        self.base_spec = base_spec
        self._base_identity = None if base_spec is None else base_spec.identity_key
        self.admission = _model_admission(registry,candidate_id,spec,mode,base_spec)
        if not self.admission.execution_allowed or registry.candidates[candidate_id].source_sha256 not in digests:
            raise ValueError('camlight source/model registry admission refused')
        self.spec,self.registry,self.candidate_id,self.mode=spec,registry,candidate_id,mode
        self.nbody=spec.dimensions.nbody
        self.ncam=len(arrays['cam_bodyid']);self.nlight=len(arrays['light_bodyid'])
        self.arrays={}
        for name in CAMLIGHT_FIELDS:
            a=np.array(arrays[name],copy=True,order='C')
            if _native_field_digest(name,a) != metadata['array_sha256'].get(name):
                raise ValueError('actual native camlight field differs: '+name)
            a=immutable_array(a);self.arrays[name]=a
        self.kind='runtime-native-camlight-candidate'
        self.source_records=records
        self.static_sha256=_camlight_digest(self.arrays)
        self._configuration=(self.nbody,self.ncam,self.nlight,spec.identity_key,self.static_sha256)
        self._readset_seal=StaticReadsetSeal(self.arrays,self._configuration[:-1],self.static_sha256)
        self._static_validated=False
        self.validate()
        self._static_validated=True

    def validate(self):
        configuration=(self.nbody,self.ncam,self.nlight,self.spec.identity_key)
        seal=getattr(self,'_readset_seal',None)
        digest=seal.current_digest(self.arrays,configuration,lambda:_camlight_digest(self.arrays)) if seal is not None else _camlight_digest(self.arrays)
        if configuration+(digest,) != self._configuration:
            raise ValueError('camlight prepared static identity changed')
        if (None if self.base_spec is None else self.base_spec.identity_key) != self._base_identity:
            raise ValueError('immutable generation base replaced')
        if not _model_admission(self.registry,self.candidate_id,self.spec,self.mode,self.base_spec).execution_allowed:
            raise ValueError('camlight model admission changed')
        if getattr(self,'_static_validated',False) and seal is not None and seal.valid(self.arrays,configuration):
            return
        shapes={'cam_bodyid':(self.ncam,),'cam_pos':(self.ncam,3),'cam_quat':(self.ncam,4),
                'cam_mode':(self.ncam,),'cam_targetbodyid':(self.ncam,),
                'light_bodyid':(self.nlight,),'light_pos':(self.nlight,3),'light_dir':(self.nlight,3),
                'light_mode':(self.nlight,),'light_targetbodyid':(self.nlight,)}
        for name in CAMLIGHT_FIELDS:
            a=self.arrays[name];dtype=np.int32 if name in CAMLIGHT_INT_FIELDS else np.float64
            if a.dtype != dtype or a.shape != shapes[name] or not a.flags.c_contiguous:
                raise ValueError('actual camlight native shape/dtype: '+name)
            if dtype==np.float64 and not np.isfinite(a).all():
                raise ValueError('nonfinite static camlight: '+name)
        if np.any(self.arrays['cam_mode']!=0) or np.any(self.arrays['light_mode']!=0):
            raise ValueError('original camlight kernel supports fixed mode only')
        for name in ('cam_bodyid','light_bodyid'):
            a=self.arrays[name]
            if np.any(a<0) or np.any(a>=self.nbody):raise ValueError('camlight body id range: '+name)

from libero.gpu.reference.observations.camlight.workspace import CamlightWorkspace

class PreparedCamlightWorkspace(CamlightWorkspace):
    """Cold bind actual BodyK owners before one immutable binding snapshot.

    Original fixed-mode camlight math is unchanged. No host pose upload or
    rebind is permitted after prepare. Static generation uses a new qualified
    model holder rather than mutating the old frozen model identity.
    """
    def __init__(self, model, capacity, *, body_inputs, device='cuda:0'):
        if not isinstance(model,PreparedCamlightModel):
            raise TypeError('exact prepared native camlight holder required')
        if set(body_inputs) != {'body_xpos','body_xquat','body_xmat','active_world'}:
            raise ValueError('all actual body camlight input owners required')
        super().__init__(model,capacity,device=device)
        self._model,self._data,self._static = model,self.data,self.static
        actual_device=wp.get_device(device)
        for name,width in (('body_xpos',3),('body_xquat',4),('body_xmat',9)):
            a=body_inputs[name]
            if not isinstance(a,wp.array) or a.dtype!=scalar.wp_real or a.shape!=(capacity,model.nbody*width) or not a.is_contiguous or a.device!=actual_device:
                raise ValueError('actual body camlight dtype/layout/device: '+name)
            _origin_record(a)
            setattr(self.data,name,a);setattr(self,name,a)
        a=body_inputs['active_world']
        if not isinstance(a,wp.array) or a.dtype!=wp.int32 or a.shape!=(capacity,) or not a.is_contiguous or a.device!=actual_device:
            raise ValueError('actual body camlight active source layout/device')
        self.data.active=a;self.active=a
        self._configuration=(capacity,model.spec.identity_key,model.nbody,model.ncam,model.nlight,str(actual_device))
        data_names=('active','body_xpos','body_xquat','body_xmat','cam_xpos','cam_xmat','light_xpos','light_xdir',
                    'cam_tmpquat','light_raw_dir','light_norm','cam_identity','light_identity','light_norm_branch','status')
        static_names=('cam_bodyid','cam_pos','cam_quat','light_bodyid','light_pos','light_dir')
        self._data_bindings=tuple((name,getattr(self.data,name),array_token(getattr(self.data,name)),_origin_record(getattr(self.data,name))) for name in data_names)
        self._static_bindings=tuple((name,getattr(self.static,name),array_token(getattr(self.static,name)),_origin_record(getattr(self.static,name))) for name in static_names)
        self.validate_bindings()

    def validate_bindings(self):
        if self.model is not self._model or self.data is not self._data or self.static is not self._static:
            raise ValueError('prepared camlight owner replaced')
        self.model.validate()
        if (self.capacity,self.model.spec.identity_key,self.model.nbody,self.model.ncam,self.model.nlight,str(wp.get_device(self.device))) != self._configuration:
            raise ValueError('prepared camlight configuration changed')
        if (self.static.nbody,self.static.ncam,self.static.nlight)!=(self.model.nbody,self.model.ncam,self.model.nlight):
            raise ValueError('prepared camlight static counts changed')
        for holder,bindings in ((self.data,self._data_bindings),(self.static,self._static_bindings)):
            for name,a,token,origin in bindings:
                if getattr(holder,name) is not a or array_token(a)!=token:
                    raise ValueError('prepared camlight actual owner changed: '+name)
                _check_origin(a,origin)
        for name in ('active','body_xpos','body_xquat','body_xmat','cam_xpos','cam_xmat','light_xpos','light_xdir','status'):
            if getattr(self,name) is not getattr(self.data,name):
                raise ValueError('prepared camlight public alias changed: '+name)

    def launch(self,stream=None):
        self.validate_bindings()
        if stream is not None and stream.device!=wp.get_device(self.device):
            raise ValueError('prepared camlight stream device mismatch')
        super().launch(stream=stream)
