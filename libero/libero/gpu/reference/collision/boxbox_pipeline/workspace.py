"""Preallocated Warp/native-CUDA BoxBox pipeline workspace."""
from __future__ import annotations
import libero.gpu.precision as scalar

import ctypes as ct
from pathlib import Path

import numpy as np
import warp as wp

from libero.gpu.reference.collision.box_cross.kernel import BoxCrossWorkspace
from libero.gpu.reference.collision.box_face_contacts.kernel import BoxFaceContactWorkspace
from libero.gpu.reference.collision.box_edge_contacts.workspace import BoxEdgeContactWorkspace
from .kernel import prefix31, map_cross, map_sat, merge_raw


class BoxBoxPipelineWorkspace:
    """N worlds, P ordered pairs; launch reads/writes only preallocated device buffers.

    All Warp kernels and G14's native CUDA kernel use the same CUDA stream.
    G14 owns its native output buffers, while the adapter passes direct Warp
    pointers for pair/SAT inputs to avoid an intermediate device copy.
    """

    def __init__(self, nworld: int, pairs: int, edge_library: str | Path,
                 device: str = "cuda:0"):
        if nworld < 1 or pairs < 1:
            raise ValueError("N and P must be positive")
        wp.init()
        self.N,self.P,self.device=nworld,pairs,device
        t=nworld*pairs
        self.stream=wp.Stream(device)
        self.pair_f=wp.empty(t*31,dtype=scalar.wp_real,device=device)
        self.active=wp.empty(nworld,dtype=wp.int32,device=device)
        self.counts=wp.empty(nworld,dtype=wp.int32,device=device)
        self.prefix55=wp.empty(t*55,dtype=scalar.wp_real,device=device)
        self.prefix_meta=wp.empty(t*5,dtype=wp.int32,device=device)
        self.prefix38=wp.empty(t*38,dtype=scalar.wp_real,device=device)
        self.prefix_state=wp.empty(t*3,dtype=wp.int32,device=device)
        self.cross=BoxCrossWorkspace(nworld,pairs,device=device)
        self.cross.prefix=self.prefix38
        self.cross.prefix_state=self.prefix_state
        self.cross.active=self.active
        self.cross.counts=self.counts
        self.sat_f=wp.empty(t*45,dtype=scalar.wp_real,device=device)
        self.sat_i=wp.empty(t*6,dtype=wp.int32,device=device)
        self.face=BoxFaceContactWorkspace(nworld,pairs,device=device)
        self.face.pair_f=self.pair_f
        self.face.sat_f=self.sat_f
        self.face.sat_i=self.sat_i
        self.face.active=self.active
        self.face.counts=self.counts
        self.edge=BoxEdgeContactWorkspace(nworld,pairs,edge_library,stream=self.stream.cuda_stream)
        self.edge_view={}
        for name,(ptr,length,code) in self.edge.buffers.items():
            if name in ("pair_f","sat_f","sat_i","active","counts"):
                continue
            self.edge_view[name]=wp.array(ptr=ptr.value,shape=(length,),
                                          dtype=scalar.wp_real if code=="d" else wp.int32,
                                          device=device)
        self.out_count=wp.empty(t,dtype=wp.int32,device=device)
        self.out_raw=wp.empty(t*500,dtype=scalar.wp_real,device=device)
        self.write_mask=wp.empty(t*500,dtype=wp.int32,device=device)
        self.state=wp.empty(t*4,dtype=wp.int32,device=device)
        self.frame_written=wp.empty(t,dtype=wp.int32,device=device)
        self.frame0=wp.empty(t*6,dtype=scalar.wp_real,device=device)
        b=self.edge.buffers
        self._edge_launch_args=tuple(
            [ct.c_void_p(a.ptr) for a in (self.pair_f,self.sat_f,self.sat_i,self.active,self.counts)]
            + [self.N,self.P]
            + [b[name][0] for name in ("count","raw","state","stage_i","stage_d",
                                          "trace_i","trace_d","frame_written","frame0")]
            + [0,ct.c_void_p(self.stream.cuda_stream)])
        self.graph=None

    @staticmethod
    def _host(value, dtype, shape, name):
        if not isinstance(value,np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name}: expected contiguous {shape} {dtype}")
        return value

    def upload(self,pair_f,active,counts,*,prefill=None):
        """Preparation boundary: validate before transfer; never called by launch."""
        pair=self._host(pair_f,np.float64,(self.N,self.P,31),"pair_f")
        active=self._host(active,np.int32,(self.N,),"active")
        counts=self._host(counts,np.int32,(self.N,),"counts")
        if np.any((active!=0)&(active!=1)):
            raise ValueError("active must be 0/1")
        if prefill is None:
            prefill=np.zeros((self.N,self.P,50,10),np.float64)
        prefill=self._host(prefill,np.float64,(self.N,self.P,50,10),"prefill")
        with wp.ScopedStream(self.stream):
            self.pair_f.assign(pair.ravel())
            self.active.assign(active)
            self.counts.assign(counts)
            self.out_raw.assign(prefill.ravel())

    def launch(self):
        """Production path: no allocation, transfer, CPU collision work or host route."""
        t=self.N*self.P
        wp.launch(prefix31,dim=t,
                  inputs=[self.pair_f,self.active,self.counts,self.prefix55,self.prefix_meta,self.P],
                  device=self.device,stream=self.stream)
        wp.launch(map_cross,dim=t,
                  inputs=[self.pair_f,self.prefix55,self.prefix_meta,self.prefix38,self.prefix_state],
                  device=self.device,stream=self.stream)
        self.cross.launch(stream=self.stream)
        wp.launch(map_sat,dim=t,
                  inputs=[self.prefix55,self.prefix_meta,self.cross.final_f,self.cross.final_state,
                          self.active,self.counts,self.sat_f,self.sat_i,self.P],
                  device=self.device,stream=self.stream)
        self.face.launch(stream=self.stream)
        rc=self.edge.lib.launch_box_edge(*self._edge_launch_args)
        self.edge._check(rc,"launch_box_edge")
        e=self.edge_view
        wp.launch(merge_raw,dim=t,
                  inputs=[self.sat_i,self.prefix_meta,self.cross.final_state,self.active,self.counts,
                          self.face.count,self.face.raw,self.face.state,self.face.frame_written,self.face.frame0,
                          e["count"],e["raw"],e["state"],e["frame_written"],e["frame0"],
                          self.out_count,self.out_raw,self.write_mask,self.state,self.frame_written,
                          self.frame0,self.P],device=self.device,stream=self.stream)

    def capture(self):
        if self.graph is not None:
            raise RuntimeError("graph already captured")
        self.launch()
        wp.synchronize_stream(self.stream)
        wp.capture_begin(device=self.device,stream=self.stream)
        self.launch()
        self.graph=wp.capture_end(device=self.device,stream=self.stream)

    def graph_launch(self):
        if self.graph is None:
            raise RuntimeError("capture first")
        wp.capture_launch(self.graph,stream=self.stream)

    def download(self,*,stages=False,tails=False):
        """Explicit diagnostic boundary outside production/captured ranges."""
        wp.synchronize_stream(self.stream)
        n,p=self.N,self.P
        result={"count":self.out_count.numpy().reshape(n,p),
                "raw":self.out_raw.numpy().reshape(n,p,50,10),
                "write_mask":self.write_mask.numpy().reshape(n,p,50,10),
                "state":self.state.numpy().reshape(n,p,4),
                "frame_written":self.frame_written.numpy().reshape(n,p),
                "frame0":self.frame0.numpy().reshape(n,p,6),
                "sat_f":self.sat_f.numpy().reshape(n,p,45),
                "sat_i":self.sat_i.numpy().reshape(n,p,6)}
        if stages:
            result.update(prefix55=self.prefix55.numpy().reshape(n,p,55),
                          prefix_meta=self.prefix_meta.numpy().reshape(n,p,5),
                          prefix38=self.prefix38.numpy().reshape(n,p,38),
                          prefix_state=self.prefix_state.numpy().reshape(n,p,3),
                          axis=self.cross.axis.numpy().reshape(n,p,54),
                          axis_mask=self.cross.axis_mask.numpy().reshape(n,p,9),
                          final_f=self.cross.final_f.numpy().reshape(n,p,4),
                          final_state=self.cross.final_state.numpy().reshape(n,p,8),
                          face_state=self.face.state.numpy().reshape(n,p,4),
                          edge_state=self.edge_view["state"].numpy().reshape(n,p))
        if tails:
            e=self.edge_view
            result.update(face_count=self.face.count.numpy().reshape(n,p),
                          face_raw=self.face.raw.numpy().reshape(n,p,50,10),
                          face_stage_i=self.face.stage_i.numpy().reshape(n,p,256),
                          face_stage_d=self.face.stage_d.numpy().reshape(n,p,512),
                          face_trace_i=self.face.trace_i.numpy().reshape(n,p,48,4),
                          face_trace_d=self.face.trace_d.numpy().reshape(n,p,48,4),
                          face_frame_written=self.face.frame_written.numpy().reshape(n,p),
                          face_frame0=self.face.frame0.numpy().reshape(n,p,6),
                          edge_count=e["count"].numpy().reshape(n,p),
                          edge_raw=e["raw"].numpy().reshape(n,p,50,10),
                          edge_stage_i=e["stage_i"].numpy().reshape(n,p,64),
                          edge_stage_d=e["stage_d"].numpy().reshape(n,p,320),
                          edge_trace_i=e["trace_i"].numpy().reshape(n,p,128),
                          edge_trace_d=e["trace_d"].numpy().reshape(n,p,256),
                          edge_frame_written=e["frame_written"].numpy().reshape(n,p),
                          edge_frame0=e["frame0"].numpy().reshape(n,p,6))
        return result

    def close(self):
        self.graph=None
        self.edge_view.clear()
        self.edge.close()
