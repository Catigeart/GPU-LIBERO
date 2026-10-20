"""Cold allocation and real same-stream consumers for RP03/RP06 new ABI."""
import libero.gpu.precision as scalar
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import warp as wp
from ..runtime.production.private_payload import PrivatePayloadOwner,SourceVersion


def _token(a):
    return (id(a),a.ptr,a.dtype,tuple(a.shape),tuple(a.strides),str(a.device))


def _payload(value):
    if not isinstance(value,np.ndarray) or value.dtype not in (np.float64,np.int32,np.int64):
        raise TypeError('typed numpy cold payload required')
    if not value.size:raise ValueError('nonempty private payload required')
    # Freeze actual bits independently; a retained caller numpy alias cannot
    # alter this asset after the cold transaction.
    raw=value.tobytes(order='C')
    return raw,value.dtype,tuple(value.shape)


class _WarpPayloadStorage:
    def __init__(self,dynamic,stream):
        self.stream=stream;self.names=tuple(dynamic)
        self._schema={n:_payload(v)[1:] for n,v in dynamic.items()}
        self._data={};self._saved={};self._closed=False
        types={np.dtype('float64'):scalar.wp_real,np.dtype('int32'):wp.int32,np.dtype('int64'):wp.int64}
        def allocate(value):
            raw,dtype,shape=_payload(value)
            return wp.array(np.frombuffer(raw,dtype=dtype).reshape(shape),dtype=types[dtype],device=stream.device)
        with wp.ScopedStream(stream):
            for name,value in dynamic.items():
                self._data[name]=allocate(value);self._saved[name]=wp.empty_like(self._data[name])
        self._seal()

    def _seal(self):
        self._tokens=tuple((group,name,_token(a)) for group in ('_data','_saved')
                           for name,a in getattr(self,group).items())
        self._stream_token=(id(self.stream),int(self.stream.cuda_stream),str(self.stream.device))
        self._config=(self.names,tuple(self._schema.items()))
        self._primitives=(wp.copy,wp.array,wp.empty_like,wp.synchronize_stream,wp.array.assign,wp.array.numpy,_payload,_token)

    def validate(self):
        if self._closed:raise ValueError('private CUDA payload released')
        if ((self.names,tuple(self._schema.items()))!=self._config
                or (wp.copy,wp.array,wp.empty_like,wp.synchronize_stream,wp.array.assign,wp.array.numpy,_payload,_token)!=self._primitives):
            raise ValueError('private CUDA payload source method/schema changed')
        if ((id(self.stream),int(self.stream.cuda_stream),str(self.stream.device))!=self._stream_token
                or tuple((g,n,_token(a)) for g in ('_data','_saved')
                    for n,a in getattr(self,g).items())!=self._tokens):
            raise ValueError('private CUDA payload owner/layout changed')
        if self.stream.is_capturing:raise ValueError('host request ABI requires eager controlled submission')

    def capture(self,name):
        wp.copy(self._saved[name],self._data[name],stream=self.stream)
        return name

    def validate_write(self,name,payload):
        raw,dtype,shape=_payload(payload)
        if (dtype,shape)!=self._schema[name]:raise ValueError('private write dtype/extent differs')

    def write(self,name,payload):
        self.validate_write(name,payload)
        raw,dtype,shape=_payload(payload)
        with wp.ScopedStream(self.stream):
            self._data[name].assign(np.frombuffer(raw,dtype=dtype).reshape(shape))

    def read_capture(self,name):
        wp.synchronize_stream(self.stream)
        return self._saved[name].numpy().tobytes(order='C')

    def replace(self,payloads):
        # Validate the whole transaction before replacing any device handle.
        for name,value in payloads.items():
            if _payload(value)[1:]!=self._schema[name]:raise ValueError('private rebind schema differs')
        replacements={}
        with wp.ScopedStream(self.stream):
            for name,value in payloads.items():
                replacements[name]=wp.array(np.array(value,copy=True,order='C'),dtype=self._data[name].dtype,device=self.stream.device)
            backups={name:wp.empty_like(a) for name,a in replacements.items()}
        self._data=replacements;self._saved=backups;self._seal()

    def release(self):
        wp.synchronize_stream(self.stream)
        self._closed=True
        self._data.clear();self._saved.clear()


def assemble_private_payloads(bundle,*,dynamic_fields,static_fields,source_records,
                              digest_pool=None,capture_policy='eager'):
    """Explicit cold install; caller receives no device arrays or native plans.

    Dynamic/static descriptors are host numpy snapshots, never aliases of old
    public resident roots. Baseline eager policy is for the same new ABI only.
    static_fields are immutable host model descriptors, with no device static
    allocation or consumer. RP06's GPU static consumer was withdrawn after its
    local negative result. Model/reset/Graph/release invalidate this controller.
    """
    owner=bundle.runtime;stream=bundle.stream;model=bundle.model_spec
    identity=model.model_sha256
    if owner.stream is not stream:raise ValueError('private payload runtime stream differs')
    records=list(source_records)
    for source in (Path(__file__),Path(__file__).parents[1]/'runtime/production/private_payload.py',
                   Path(__file__).parents[1]/'runtime/production/static_domain.py',
                   Path(__file__).parents[1]/'runtime/production/retained_witness.py'):
        records.append((source,hashlib.sha256(source.read_bytes()).hexdigest()))
    credential=SourceVersion(records,digest_pool=digest_pool)
    # Freeze a descriptor exactly once before both upload and lease creation.
    # Concurrent caller writes thereafter affect neither copy of the asset.
    def freeze(fields):
        result={}
        for name,value in fields.items():
            if type(name) is not str or not name:raise ValueError('private field name required')
            raw,dtype,shape=_payload(value)
            result[name]=np.frombuffer(raw,dtype=dtype).reshape(shape)
        return result
    dynamic_fields=freeze(dynamic_fields);static_fields=freeze(static_fields)
    storage=_WarpPayloadStorage(dynamic_fields,stream)
    def binding_guard():
        if (getattr(bundle,'closed',False) or bundle.runtime is not owner or bundle.stream is not stream
                or owner.stream is not stream or bundle.model_spec is not model
                or model.model_sha256!=identity):raise ValueError('private payload actual owner/model/stream changed')
    controller=PrivatePayloadOwner(storage,credential,binding_guard,identity,
        {n:_payload(v)[0] for n,v in static_fields.items()},capture_policy=capture_policy)
    def expose(name):
        bound=getattr(controller,name);validator=controller._validate
        def operation(*args,**kwargs):
            with controller._lock:
                if bound.__self__ is not controller:raise ValueError('private exported owner changed')
                # The cold original validator cannot be bypassed by replacing
                # a method. Terminal quarantine remains callable.
                if name not in ('invalidate','release'):validator()
                return bound(*args,**kwargs)
        return operation
    return SimpleNamespace(**{name:expose(name) for name in
        ('begin','write','materialize_error','discard','invalidate','reset','rebind','release')},
        manifest=dict(abi='private-payload-copy-on-error-v2',
            installed_default=False,raw_arrays_exposed=False,capture_policy=capture_policy,
            dynamic_fields=tuple(dynamic_fields),static_descriptor_fields=tuple(static_fields),
            static_device_consumer=False))
