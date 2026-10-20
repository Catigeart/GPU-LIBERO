"""D01 prepare-time model rejection and device-resident camlight workspace."""
from __future__ import annotations
import libero.gpu.precision as scalar

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from .kernel import Data, Model, compute

MODEL_FIELDS=('cam_bodyid','cam_pos','cam_quat','cam_mode','cam_targetbodyid',
              'light_bodyid','light_pos','light_dir','light_mode','light_targetbodyid')
INT_FIELDS={'cam_bodyid','cam_mode','cam_targetbodyid','light_bodyid','light_mode','light_targetbodyid'}
# Filled from the pre-GPU CPU census.  Both original MJB identities are checked.
FIXED={
    'real':{'file_sha256':'3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42',
            'static_sha256':'b14642399727822b5fc70202cc36c4d841ebc1dc971e3c9399a2b511fe8ce7a5'},
    'empty':{'file_sha256':'8025007f27143c14a048993c6faac2f1224b5190ab548848c8627f54cc3ae632',
             'static_sha256':'9e8e0d3fea4517765f6d415a1532e61c3220fe70f706a9f9c96fcd8edec9d966'},
}


def _canonical(fields):
    h=hashlib.sha256()
    for name in sorted(fields):
        a=np.asarray(fields[name])
        a=np.ascontiguousarray(a.astype('<i8' if a.dtype.kind in 'iu' else '<f8',copy=False))
        h.update(json.dumps([name,a.shape,a.dtype.str],separators=(',',':')).encode())
        h.update(b'\0');h.update(a.tobytes())
    return h.hexdigest()


class FrozenCamlightModel:
    def __init__(self,spec,requirements,kind):
        if kind not in FIXED:raise ValueError('unsupported model kind')
        spec=Path(spec);item=requirements['model'][kind]
        if hashlib.sha256(spec.read_bytes()).hexdigest()!=item['spec_sha256']:
            raise ValueError('model spec SHA')
        if item['file_sha256']!=FIXED[kind]['file_sha256'] or item['static_sha256']!=FIXED[kind]['static_sha256']:
            raise ValueError('frozen model identity')
        with np.load(spec,allow_pickle=False) as z:
            if set(z.files)!=set(MODEL_FIELDS):raise ValueError('model field set')
            self.arrays={k:np.ascontiguousarray(z[k]) for k in MODEL_FIELDS}
        self.kind=kind;self.nbody=int(item['nbody']);self.ncam=int(item['ncam']);self.nlight=int(item['nlight'])
        self.validate()

    def validate(self):
        a=self.arrays
        if (self.nbody,self.ncam,self.nlight) not in ((38,7,2),(6,0,0)):
            raise ValueError('unsupported topology')
        shapes={'cam_bodyid':(self.ncam,),'cam_pos':(self.ncam,3),'cam_quat':(self.ncam,4),
          'cam_mode':(self.ncam,),'cam_targetbodyid':(self.ncam,),
          'light_bodyid':(self.nlight,),'light_pos':(self.nlight,3),
          'light_dir':(self.nlight,3),'light_mode':(self.nlight,),
          'light_targetbodyid':(self.nlight,)}
        for name in MODEL_FIELDS:
            dtype=np.int32 if name in INT_FIELDS else np.float64
            if a[name].dtype!=dtype or a[name].shape!=shapes[name] or not a[name].flags.c_contiguous:
                raise ValueError('model schema '+name)
            if dtype==np.float64 and not np.isfinite(a[name]).all():
                raise ValueError('nonfinite static '+name)
        if np.any(a['cam_mode']!=0) or np.any(a['light_mode']!=0):
            raise ValueError('only fixed mode')
        for k in ('cam_bodyid','light_bodyid'):
            if np.any(a[k]<0) or np.any(a[k]>=self.nbody):raise ValueError('body id '+k)
        if _canonical(a)!=FIXED[self.kind]['static_sha256']:
            raise ValueError('frozen read-set SHA')


class CamlightWorkspace:
    """All production arrays are resident; launch enqueues one real CUDA kernel."""
    def __init__(self,model:FrozenCamlightModel,capacity:int,device='cuda:0'):
        if not isinstance(model,FrozenCamlightModel):raise TypeError('frozen model required')
        if not isinstance(capacity,int) or capacity<1:raise ValueError('capacity')
        model.validate()
        self.model=model;self.capacity=capacity;self.device=device
        m=Model();n=model
        for name in ('cam_bodyid','cam_pos','cam_quat','light_bodyid','light_pos','light_dir'):
            arr=model.arrays[name].reshape(-1)
            if arr.size==0:
                arr=np.zeros(1,dtype=arr.dtype)  # inaccessible static padding for empty model
            setattr(m,name,wp.array(arr,dtype=wp.int32 if name.endswith('id') else scalar.wp_real,device=device))
        m.nbody=n.nbody;m.ncam=n.ncam;m.nlight=n.nlight;self.static=m
        d=Data()
        d.active=wp.zeros(capacity,dtype=wp.int32,device=device)
        for name,cols in (('body_xpos',3*n.nbody),('body_xquat',4*n.nbody),('body_xmat',9*n.nbody),
                          ('cam_xpos',3*n.ncam),('cam_xmat',9*n.ncam),('light_xpos',3*n.nlight),
                          ('light_xdir',3*n.nlight),('cam_tmpquat',4*n.ncam),('light_raw_dir',3*n.nlight),('light_norm',n.nlight)):
            setattr(d,name,wp.zeros((capacity,max(1,cols)),dtype=scalar.wp_real,device=device))
        for name,cols in (('cam_identity',n.ncam),('light_identity',n.nlight),('light_norm_branch',n.nlight)):
            setattr(d,name,wp.zeros((capacity,max(1,cols)),dtype=wp.int32,device=device))
        d.status=wp.zeros(capacity,dtype=wp.int32,device=device)
        self.data=d
        for name in ('active','body_xpos','body_xquat','body_xmat','cam_xpos','cam_xmat','light_xpos','light_xdir','status'):
            setattr(self,name,getattr(d,name))

    def launch(self,stream=None):
        wp.launch(compute,dim=self.capacity,inputs=[self.static,self.data],device=self.device,stream=stream)

    def upload(self,active,xpos,xquat,xmat):
        """Host convenience, exclusively outside production or Graph replay."""
        expected={'active':((self.capacity,),np.int32),
                  'xpos':((self.capacity,self.model.nbody,3),np.float64),
                  'xquat':((self.capacity,self.model.nbody,4),np.float64),
                  'xmat':((self.capacity,self.model.nbody,9),np.float64)}
        for name,value in (('active',active),('xpos',xpos),('xquat',xquat),('xmat',xmat)):
            shape,dtype=expected[name]
            if not isinstance(value,np.ndarray) or value.dtype!=dtype or value.shape!=shape:
                raise ValueError('input schema '+name)
            dst=self.active if name=='active' else getattr(self,'body_'+name)
            payload=np.ascontiguousarray(value.reshape((self.capacity,-1)) if name!='active' else value)
            wp.copy(dst,wp.array(payload,dtype=wp.int32 if name=='active' else scalar.wp_real,device=self.device))

    def download(self):
        """Diagnostic transfer outside the production interval."""
        n=self.model;N=self.capacity;d=self.data
        shapes={'cam_xpos':(N,n.ncam,3),'cam_xmat':(N,n.ncam,9),
                'light_xpos':(N,n.nlight,3),'light_xdir':(N,n.nlight,3),
                'cam_tmpquat':(N,n.ncam,4),'light_raw_dir':(N,n.nlight,3),'light_norm':(N,n.nlight),
                'cam_identity':(N,n.ncam),'light_identity':(N,n.nlight),
                'light_norm_branch':(N,n.nlight)}
        out={name:getattr(d,name).numpy()[:,:int(np.prod(shape[1:]))].reshape(shape).copy()
             for name,shape in shapes.items()}
        out['status']=d.status.numpy().copy()
        return out
