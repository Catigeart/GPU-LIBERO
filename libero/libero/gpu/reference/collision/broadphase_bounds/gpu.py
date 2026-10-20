"""Preparation, resident launch and Graph binding for C02.

All numpy use is preparation/diagnostic only.  `run` and `graph` call only a
precompiled CUDA launch; their device inputs and outputs are already resident.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import ctypes as ct
import os
import subprocess
from pathlib import Path

import numpy as np


class Output(ct.Structure):
    _fields_ = [('status', ct.c_int), ('entered', ct.c_int), ('count', ct.c_int),
                ('iter', ct.c_int), ('reason', ct.c_int),
                ('cen', scalar.ct_real * 3), ('cov', scalar.ct_real * 9),
                ('eigval', scalar.ct_real * 3), ('frame', scalar.ct_real * 9),
                ('quat', scalar.ct_real * 4), ('aabb', scalar.ct_real * (6*38)),
                ('valid', ct.c_int * 38), ('axis', ct.c_int * 500),
                ('stop', ct.c_int * 500), ('offdiag', scalar.ct_real * 500),
                ('cosine', scalar.ct_real * 500)]


def build_library(cache: Path) -> Path:
    """Compile once in the private result cache, under the caller's outer lock."""
    cache.mkdir(parents=True, exist_ok=False)
    source = (GPU_ROOT / 'reference/collision/broadphase_bounds/bounds.cu')
    library = cache / 'libc02_cuda.so'
    cmd = ['nvcc', '-std=c++17', '-shared', '-Xcompiler', '-fPIC', '-O2',
           '-arch=sm_80', '-fmad=true', '--prec-div=true', '--prec-sqrt=true',
           '--ftz=false', '-lineinfo', '-keep', '-keep-dir', str(cache),
           str(source), '-o', str(library)]
    env = os.environ.copy(); env['TMPDIR'] = str(cache)
    p = subprocess.run(cmd, cwd=cache, env=env, capture_output=True, text=True)
    (cache/'build.log').write_text(repr(cmd)+'\nexit_code='+str(p.returncode)+'\n'+p.stdout+p.stderr)
    if p.returncode:
        raise RuntimeError(f'nvcc failed ({p.returncode}): {p.stderr[-2000:]}')
    return library


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ct.POINTER(ctype))


def validate_static(static: dict[str, np.ndarray], n: int) -> tuple[int, int]:
    if type(n) is not int or n < 1 or n > 128:
        raise ValueError('world capacity must be int in [1,128]')
    int_g = ('geom_bodyid','geom_contype','geom_conaffinity')
    int_b = ('body_geomadr','body_geomnum')
    double_g = ('geom_rbound','geom_margin')
    g = len(static['geom_bodyid']); b = len(static['body_geomadr'])
    if not 1 <= g <= 1024 or not 1 <= b <= 38:
        raise ValueError('model capacity')
    for key in int_g + int_b:
        a=static[key]; expected = g if key in int_g else b
        if not isinstance(a,np.ndarray) or a.dtype!=np.int32 or a.shape!=(expected,) or not a.flags.c_contiguous:
            raise ValueError(f'{key}: expected contiguous int32 [{expected}]')
    for key in double_g:
        a=static[key]
        if not isinstance(a,np.ndarray) or a.dtype!=np.float64 or a.shape!=(g,) or not a.flags.c_contiguous or not np.all(np.isfinite(a)):
            raise ValueError(f'{key}: expected finite contiguous float64 [{g}]')
    if np.any(static['geom_bodyid']<0) or np.any(static['geom_bodyid']>=b):
        raise ValueError('geom body range')
    covered=np.zeros(g,dtype=np.int8)
    for body in range(b):
        start=int(static['body_geomadr'][body]); count=int(static['body_geomnum'][body])
        if count<0 or (count>0 and (start<0 or start+count>g)):
            raise ValueError('body geom span')
        for geom in range(start,start+count):
            if covered[geom] or static['geom_bodyid'][geom]!=body:
                raise ValueError('body geom order')
            covered[geom]=1
    if not np.all(covered):
        raise ValueError('body geom spans do not cover all geoms')
    return g,b


class Workspace:
    def __init__(self, library: Path, static: dict[str,np.ndarray], n: int):
        self.g,self.b=validate_static(static,n)
        self.n=n
        self.lib=ct.CDLL(str(library))
        l=self.lib
        l.c02_output_size.restype=ct.c_size_t
        if l.c02_output_size()!=ct.sizeof(Output):
            raise AssertionError('host/device Output ABI mismatch')
        l.c02_create.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),ct.POINTER(scalar.ct_real),ct.POINTER(scalar.ct_real),
            ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
        l.c02_create.restype=ct.c_void_p
        l.c02_upload.argtypes=[ct.c_void_p,ct.POINTER(scalar.ct_real),ct.POINTER(ct.c_int)]
        l.c02_seed.argtypes=[ct.c_void_p,ct.POINTER(Output)]
        l.c02_upload_cov.argtypes=[ct.c_void_p,ct.POINTER(scalar.ct_real),ct.POINTER(ct.c_int)]
        l.c02_run_eig.argtypes=[ct.c_void_p]
        l.c02_run.argtypes=[ct.c_void_p]
        l.c02_capture_graph.argtypes=[ct.c_void_p]
        l.c02_graph.argtypes=[ct.c_void_p]
        l.c02_sync.argtypes=[ct.c_void_p]
        l.c02_download.argtypes=[ct.c_void_p,ct.POINTER(Output)]
        l.c02_destroy.argtypes=[ct.c_void_p]
        self.static=static
        self.handle=l.c02_create(n,self.g,self.b,*[_ptr(static[k], ct.c_int) for k in
            ('geom_bodyid','body_geomadr','body_geomnum')],
            *[_ptr(static[k], scalar.ct_real) for k in ('geom_rbound','geom_margin')],
            *[_ptr(static[k], ct.c_int) for k in ('geom_contype','geom_conaffinity')])
        if not self.handle:
            raise RuntimeError('cuda workspace allocation or static upload failed')
        self._held=[]

    def upload(self, xpos: np.ndarray, active: np.ndarray) -> None:
        """Completed prior launches are synchronized in the C wrapper before upload."""
        if not isinstance(xpos,np.ndarray) or xpos.dtype!=np.float64 or xpos.shape!=(self.n,self.g,3) or not xpos.flags.c_contiguous:
            raise ValueError('geom_xpos must be contiguous float64 [N,G,3]')
        if not isinstance(active,np.ndarray) or active.dtype!=np.int32 or active.shape!=(self.n,) or not active.flags.c_contiguous:
            raise ValueError('active must be contiguous int32 [N]')
        if np.any((active!=0)&(active!=1)):
            raise ValueError('active must contain only 0/1')
        self._held=[xpos,active]
        if self.lib.c02_upload(self.handle,_ptr(xpos,scalar.ct_real),_ptr(active,ct.c_int)):
            raise RuntimeError('CUDA upload failed')

    def seed(self, byte: int=0x55) -> None:
        host=(Output*self.n)()
        ct.memset(ct.byref(host),byte,ct.sizeof(host))
        if self.lib.c02_seed(self.handle,host):
            raise RuntimeError('CUDA seed upload failed')
        self._seed=host

    def upload_cov(self, cov: np.ndarray, active: np.ndarray) -> None:
        if not isinstance(cov,np.ndarray) or cov.dtype!=np.float64 or cov.shape!=(self.n,9) or not cov.flags.c_contiguous:
            raise ValueError('local cov must be contiguous float64 [N,9]')
        if not isinstance(active,np.ndarray) or active.dtype!=np.int32 or active.shape!=(self.n,) or not active.flags.c_contiguous or np.any((active!=0)&(active!=1)):
            raise ValueError('local active must be int32 0/1 [N]')
        self._held=[cov,active]
        if self.lib.c02_upload_cov(self.handle,_ptr(cov,scalar.ct_real),_ptr(active,ct.c_int)):
            raise RuntimeError('CUDA local covariance upload failed')

    def run_eig(self) -> None:
        if self.lib.c02_run_eig(self.handle): raise RuntimeError('CUDA local eig launch failed')

    def run(self) -> None:
        if self.lib.c02_run(self.handle): raise RuntimeError('CUDA production launch failed')

    def capture_graph(self) -> None:
        if self.lib.c02_capture_graph(self.handle): raise RuntimeError('CUDA graph capture failed')

    def graph(self) -> None:
        if self.lib.c02_graph(self.handle): raise RuntimeError('CUDA graph launch failed')

    def sync(self) -> None:
        if self.lib.c02_sync(self.handle): raise RuntimeError('CUDA stream sync failed')

    def download(self) -> list[Output]:
        host=(Output*self.n)()
        if self.lib.c02_download(self.handle,host): raise RuntimeError('CUDA diagnostic download failed')
        return list(host)

    def close(self) -> None:
        if self.handle:
            self.lib.c02_destroy(self.handle);self.handle=None

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
