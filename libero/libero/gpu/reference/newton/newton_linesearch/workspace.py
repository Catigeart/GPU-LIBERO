"""Strict host ABI and preallocated CUDA workspace for S07."""
from __future__ import annotations
import libero.gpu.precision as scalar

import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np
import warp as wp


FLOAT_INPUTS = ('search','Ma','Jaref','quad0','qfrc_smooth','qM','J',
                'D','R','loss','cone_mu','cone_friction')
INT_INPUTS = ('type','id','cone_dim','dof_Madr','dof_parentid','dof_simplenum',
              'nefc','ne','nf','active')
FLOAT_OUTPUTS = ('Mv','Jv','quadGauss','quad','alpha','slope','eval_trace')
INT_OUTPUTS = ('LSiter','LSresult','eval_tag','status')
_MAIN_TOPOLOGY_SHA = {
    'dof_Madr':'3e27466135207e6124bbf87bad323aa0795dd99f58cb135885cce724e1f7b27d',
    'dof_parentid':'a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a',
    'dof_simplenum':'58f218aeb259309eeb545345a8f5c715987967b0ec724806d448394da578d0b0',
}


class Args(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in (
        'search','Ma','Jaref','quad0','smooth','qM','J','D','R','loss',
        'cone_mu','cone_friction','efc_type','efc_id','cone_dim',
        'madr','parent','simplenum','nefc','ne','nf','active','model_scalars',
        'Mv','Jv','quadGauss','quad','alpha','slope','eval_trace',
        'LSiter','LSresult','eval_tag','status')]


class NewtonLineSearchWorkspace:
    def __init__(self, nworld: int, rows: int, bridge: Path, static: dict,
                 *, device='cuda:0', tracecap=52):
        if type(nworld) is not int or nworld < 1 or type(rows) is not int or rows < 1:
            raise ValueError('positive world and row capacity required')
        if tracecap != 52:
            raise ValueError('diagnostic allocation must be exactly 52 records')
        if not wp.get_device(device).is_cuda:
            raise ValueError('CUDA device required')
        self.nworld, self.rows, self.nv, self.nM = nworld, rows, 43, 153
        self.device, self.tracecap = device, tracecap
        self.stream = wp.Stream(device)
        self.arrays = {}
        shapes = {
            'search':(nworld,43), 'Ma':(nworld,43),
            'Jaref':(nworld,rows), 'quad0':(nworld,),
            'qfrc_smooth':(nworld,43), 'qM':(nworld,153),
            'J':(nworld,rows,43), 'D':(nworld,rows),
            'R':(nworld,rows), 'loss':(nworld,rows),
            'type':(nworld,rows), 'id':(nworld,rows),
            'cone_dim':(nworld,rows), 'cone_mu':(nworld,rows),
            'cone_friction':(nworld,rows,5),
            'nefc':(nworld,), 'ne':(nworld,), 'nf':(nworld,),
            'active':(nworld,), 'Mv':(nworld,43), 'Jv':(nworld,rows),
            'quadGauss':(nworld,3), 'quad':(nworld,rows,3),
            'alpha':(nworld,), 'slope':(nworld,),
            'eval_trace':(nworld,52,4), 'eval_tag':(nworld,52),
            'LSiter':(nworld,), 'LSresult':(nworld,), 'status':(nworld,)}
        for name, shape in shapes.items():
            dtype = wp.int32 if name in INT_INPUTS+INT_OUTPUTS else scalar.wp_real
            self.arrays[name] = wp.zeros(shape, dtype=dtype, device=device)
        for key in ('dof_Madr','dof_parentid','dof_simplenum'):
            value = np.asarray(static[key])
            if value.dtype != np.int32 or value.shape != (43,):
                raise TypeError((key, value.shape, value.dtype))
            self.arrays[key] = wp.array(value, dtype=wp.int32, device=device)
        scalars = np.asarray(static['model_scalars'])
        if scalars.dtype != np.float64 or scalars.shape != (2,):
            raise TypeError(('model_scalars', scalars.shape, scalars.dtype))
        self.arrays['model_scalars'] = wp.array(scalars, dtype=scalar.wp_real, device=device)
        self.static_sha256 = {key: hashlib.sha256(np.ascontiguousarray(
            static[key]).tobytes()).hexdigest() for key in static}
        for key,digest in _MAIN_TOPOLOGY_SHA.items():
            if self.static_sha256[key] != digest:
                raise ValueError(('unsupported main-model topology',key,self.static_sha256[key]))
        if not np.all(np.isfinite(scalars)) or scalars[0]<0 or scalars[1]<=0:
            raise ValueError('invalid model tolerance or meaninertia')
        self.lib = ct.CDLL(str(bridge.resolve()))
        self.lib.s07_launch.argtypes = [ct.POINTER(Args),ct.c_int,ct.c_int,
                                        ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.lib.s07_launch.restype = ct.c_int
        self._args = Args(**{
            'smooth':self.arrays['qfrc_smooth'].ptr,
            'efc_type':self.arrays['type'].ptr,
            'efc_id':self.arrays['id'].ptr,
            'madr':self.arrays['dof_Madr'].ptr,
            'parent':self.arrays['dof_parentid'].ptr,
            'simplenum':self.arrays['dof_simplenum'].ptr,
            **{key:self.arrays[key].ptr for key in self.arrays if key in dict(Args._fields_)}})

    def upload(self, batch: dict):
        """Preparation only; never called inside a production launch or capture."""
        required = set(FLOAT_INPUTS+INT_INPUTS)-{'dof_Madr','dof_parentid','dof_simplenum'}
        if set(batch) != required:
            raise ValueError(('input keys', sorted(set(batch)^required)))
        for key in required:
            value = np.asarray(batch[key])
            target = self.arrays[key]
            dtype = np.int32 if key in INT_INPUTS else np.float64
            if value.dtype != dtype or value.shape != target.shape:
                raise TypeError((key, value.shape, value.dtype,target.shape,dtype))
            if key in FLOAT_INPUTS and not np.all(np.isfinite(value)):
                raise ValueError((key, 'nonfinite input'))
            target.assign(np.ascontiguousarray(value))

    def launch(self, *, diagnostic_capacity=52):
        """One meaningful CUDA kernel; all payload and scratch remain device resident."""
        error = self.lib.s07_launch(ct.byref(self._args), self.nworld, self.nv,
                                    self.nM, self.rows, diagnostic_capacity,
                                    ct.c_void_p(int(self.stream.cuda_stream)))
        if error:
            raise RuntimeError(f'CUDA launch error {error}')

    def outputs(self):
        """Diagnostic download, outside the production launch."""
        wp.synchronize_stream(self.stream)
        return {name:self.arrays[name].numpy().copy() for name in FLOAT_OUTPUTS+INT_OUTPUTS}
