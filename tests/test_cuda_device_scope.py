"""Resource boundaries restore thread-local CUDA device ownership."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from threading import local
import unittest
from libero.gpu.host.backends.cuda_fp32 import device as cuda


class FakeRuntime:
    def __init__(self):
        self.state=local();self.events=[]
    def call(self,name,*args):
        current=getattr(self.state,'ordinal',0)
        if name=='get_device':ct.cast(args[0],ct.POINTER(ct.c_int))[0]=current
        elif name=='set_device':self.state.ordinal=args[0]
        elif name=='device':
            ct.cast(args[1],ct.POINTER(ct.c_int))[0]=90
        elif name=='stream_create':ct.cast(args[0],ct.POINTER(ct.c_void_p))[0]=len(self.events)+1
        elif name=='allocate':ct.cast(args[0],ct.POINTER(ct.c_void_p))[0]=len(self.events)+1
        self.events.append((name,current))


class DeviceScopeTests(unittest.TestCase):
    def test_interleaved_resources_and_exception_restore(self):
        runtime=FakeRuntime()
        first,second=cuda.Device(runtime,0),cuda.Device(runtime,1)
        a,b=cuda.Stream(first),cuda.Stream(second)
        allocation=cuda.Allocation(b,4)
        self.assertEqual(runtime.state.ordinal,0)
        with a:
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with b:
                    self.assertEqual(runtime.state.ordinal,1)
                    self.assertIs(cuda.current_stream(),b)
                    raise RuntimeError('injected')
            self.assertEqual(runtime.state.ordinal,0)
            self.assertIs(cuda.current_stream(),a)
        with self.assertRaisesRegex(RuntimeError,'capture failed'):
            with cuda.Graph(b):
                self.assertEqual(runtime.state.ordinal,1)
                raise RuntimeError('capture failed')
        self.assertEqual(runtime.state.ordinal,0)
        allocation.close();b.close();a.close()
        for operation in ('allocate','free'):
            self.assertEqual([d for n,d in runtime.events if n==operation],[1])
        self.assertEqual([d for n,d in runtime.events if n=='stream_create'],[0,1])
        self.assertEqual(runtime.state.ordinal,0)

    def test_same_scope_object_has_separate_thread_stacks(self):
        runtime=FakeRuntime();stream=cuda.Stream(cuda.Device(runtime,1))
        def worker(original):
            runtime.state.ordinal=original
            with stream:
                with stream:self.assertEqual(runtime.state.ordinal,1)
                self.assertIs(cuda.current_stream(),stream)
            self.assertEqual(runtime.state.ordinal,original)
            self.assertIsNone(cuda._active_stream.get())
        with ThreadPoolExecutor(2) as pool:list(pool.map(worker,[0,2]))
        stream.close()
