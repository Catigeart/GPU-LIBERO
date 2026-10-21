"""Independent runtime ownership/copy/Graph checks with Warp imports forbidden."""
import importlib.abc
import json
import sys
from pathlib import Path


class ForbidWarp(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('warp','mujoco_warp'):
            raise ImportError('Warp-free validation rejected import: '+fullname)


def add_arguments(parser):
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)


def main(args):
    import numpy as np
    sys.meta_path.insert(0,ForbidWarp())
    from libero.gpu.host.backends.cuda_fp32 import device as cuda
    runtime=cuda.Runtime(args.build_root/'cmake-control/sm80-dev/cuda_runtime.so')
    device=runtime.device();stream=cuda.Stream(device)
    try:
        with stream:
            source=cuda.array(np.arange(42,dtype=np.float32).reshape(2,21))
            target=cuda.zeros(source.shape)
            with cuda.Graph(stream,owners=[source,target]) as graph:cuda.copy(target,source)
            graph.launch();assert np.array_equal(target.numpy(),source.numpy())
            source.assign(np.full(source.shape,7,dtype=np.float32));graph.launch()
            assert np.array_equal(target.numpy(),np.full(source.shape,7,dtype=np.float32))
            alias=target.reshape((42,));del target;assert np.all(alias.numpy()==7)
            graph.close()
            empty=cuda.zeros((2,0));assert empty.capacity==0 and empty.numpy().shape==(2,0)
            vector=cuda.array(np.ones((2,3),dtype=np.float32),dtype=cuda.vec3)
            assert vector.shape==(2,) and vector.numpy().shape==(2,3)
            counter=cuda.zeros(1,dtype=int)
            with cuda.Graph(stream,owners=[counter]) as loop:
                cuda.store_integer(counter,4)
                with cuda.While(counter):runtime.call('decrement_integer',counter.ptr,stream.cuda_stream)
            loop.launch();assert counter.numpy()[0]==0
            loop.launch();assert counter.numpy()[0]==0
            loop.close()
            assert not any(k.split('.')[0] in ('warp','mujoco_warp') for k in sys.modules)
        report=dict(passed=True,worlds=2,scope=['allocation','typed views','current-input Graph copy','zero-width fields','vector storage','device conditional while and repeated reset'],warp_imported=False,integrations=0,performance=None,arch=device.arch,
            runtime_source_id=runtime.lib.flash_runtime_source_id().decode())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    finally:stream.close()
    return 0
