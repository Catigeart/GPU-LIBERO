"""Prepared device-only input tape; diagnostic writes never feed physics."""
import libero.gpu.precision as scalar
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})


@wp.kernel(enable_backward=False)
def _copy_f64(source:wp.array(dtype=scalar.wp_real),target:wp.array(dtype=scalar.wp_real),
              index:wp.array(dtype=wp.int32),width:int,capacity:int):
    i=wp.tid()
    row=index[0]
    if row>=0 and row<capacity:
        target[row*width+i]=source[i]


@wp.kernel(enable_backward=False)
def _copy_i64(source:wp.array(dtype=wp.int64),target:wp.array(dtype=wp.int64),
              index:wp.array(dtype=wp.int32),width:int,capacity:int):
    i=wp.tid()
    row=index[0]
    if row>=0 and row<capacity:
        target[row*width+i]=source[i]


@wp.kernel(enable_backward=False)
def _copy_i32(source:wp.array(dtype=wp.int32),target:wp.array(dtype=wp.int32),
              index:wp.array(dtype=wp.int32),width:int,capacity:int):
    i=wp.tid()
    row=index[0]
    if row>=0 and row<capacity:
        target[row*width+i]=source[i]


@wp.kernel(enable_backward=False)
def _advance(index:wp.array(dtype=wp.int32),overflow:wp.array(dtype=wp.int32),capacity:int):
    if index[0]>=capacity:
        overflow[0]=1
    else:
        index[0]=index[0]+1


def _token(a):
    return (id(a),int(a.ptr),a.dtype,a.shape,a.strides,str(a.device),a.capacity)


class DeviceInputTape:
    """Each callback records actual operands before the named consumer.

    The independent counter counts callbacks, including masked invocations;
    recorded intent/epochs identify the actual producer scope. CUDA replay
    advances it on device. Exhaustion is diagnostic and cannot change physics.
    """
    def __init__(self,sources,*,capacity,stream,phase):
        if type(capacity) is not int or not 0<capacity<2147483647:
            raise ValueError('bounded tape capacity required')
        self.sources=dict(sources)
        self.capacity,self.stream,self.phase=capacity,stream,str(phase)
        self.tokens={k:_token(a) for k,a in self.sources.items()}
        self.flat={k:a.flatten() for k,a in self.sources.items()}
        self._flat_tokens={k:_token(a) for k,a in self.flat.items()}
        for a in self.sources.values():
            if a.dtype not in (scalar.wp_real,wp.int64,wp.int32) or a.device!=stream.device:
                raise TypeError('typed f64/i64/i32 CUDA operand tape')
        self.index=wp.zeros(1,dtype=wp.int32,device=stream.device)
        self.overflow=wp.zeros(1,dtype=wp.int32,device=stream.device)
        self.data={k:wp.empty(capacity*a.size,dtype=a.dtype,device=stream.device)
                   for k,a in self.sources.items()}
        self._owners={k:_token(a) for k,a in self.data.items()}
        self._counter_tokens=(_token(self.index),_token(self.overflow))
        self._config=(capacity,id(stream),int(stream.cuda_stream),int(stream.device.context),
                      id(self.sources),id(self.flat),id(self.data),tuple(self.sources))

    def capture(self):
        if (self.capacity,id(self.stream),int(self.stream.cuda_stream),int(self.stream.device.context),
                id(self.sources),id(self.flat),id(self.data),tuple(self.sources))!=self._config:
            raise ValueError('prepared tape configuration/keys changed')
        if (_token(self.index),_token(self.overflow))!=self._counter_tokens:
            raise ValueError('prepared tape counters changed')
        for k,a in self.sources.items():
            if (_token(a)!=self.tokens[k] or _token(self.data[k])!=self._owners[k]
                    or _token(self.flat[k])!=self._flat_tokens[k]):
                raise ValueError('prepared tape source/target changed: '+k)
            kernel={scalar.wp_real:_copy_f64,wp.int64:_copy_i64,wp.int32:_copy_i32}[a.dtype]
            wp.launch(kernel,dim=a.size,
                inputs=[self.flat[k],self.data[k],self.index,a.size,self.capacity],
                device=self.stream.device,stream=self.stream)
        wp.launch(_advance,dim=1,inputs=[self.index,self.overflow,self.capacity],
                  device=self.stream.device,stream=self.stream)

    def download(self):
        wp.synchronize_stream(self.stream)
        return {'index':self.index.numpy(),'overflow':self.overflow.numpy(),
                **{k:a.numpy().reshape((self.capacity,*self.sources[k].shape))
                   for k,a in self.data.items()}}
