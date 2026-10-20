"""Original MuJoCo 2.3.7 contact parameter rules after raw-contact dedup.

Source engine_collision_driver.c:900-927,990-1096 and
engine_core_constraint.c:92-120, Apache-2.0; hashes in source_oracle.py.
Frame completion and global contact append are outside this kernel.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})


def validate_static(static):
    """Validate the frozen static-table schema and the executed solmix domain.

    This is a one-time host admission check. The device still computes every
    accepted contact; the per-route arithmetic envelope is checked by budget.audit.
    """
    required=('geom_priority','geom_condim','geom_margin','geom_gap',
              'geom_friction','geom_solmix','geom_solref','geom_solimp',
              'pair_geom1','pair_geom2','pair_margin','pair_gap','pair_dim',
              'pair_friction','pair_solref','pair_solreffriction','pair_solimp',
              'o_margin','o_solref','o_solimp','override')
    if any(name not in static for name in required):
        raise ValueError('static schema: missing field')
    g=len(static['geom_priority']);q=len(static['pair_margin'])
    if g<1:
        raise ValueError('static schema: no geoms')
    specs={
        'geom_priority':((g,),np.int32),'geom_condim':((g,),np.int32),
        'geom_margin':((g,),np.float64),'geom_gap':((g,),np.float64),
        'geom_friction':((g,3),np.float64),'geom_solmix':((g,),np.float64),
        'geom_solref':((g,2),np.float64),'geom_solimp':((g,5),np.float64),
        'pair_geom1':((q,),np.int32),'pair_geom2':((q,),np.int32),
        'pair_margin':((q,),np.float64),'pair_gap':((q,),np.float64),
        'pair_dim':((q,),np.int32),'pair_friction':((q,5),np.float64),
        'pair_solref':((q,2),np.float64),
        'pair_solreffriction':((q,2),np.float64),
        'pair_solimp':((q,5),np.float64),
        'o_margin':((1,),np.float64),'o_solref':((2,),np.float64),
        'o_solimp':((5,),np.float64),'override':((1,),np.int32),
    }
    for name,(shape,dtype) in specs.items():
        arr=np.asarray(static[name])
        if arr.shape!=shape or arr.dtype!=dtype:
            raise ValueError(f'static schema: {name} shape/dtype {arr.shape}/{arr.dtype}')
        if dtype==np.float64 and not np.all(np.isfinite(arr)):
            raise ValueError(f'unsupported_numeric_range: {name} must be finite')
    priority=np.asarray(static['geom_priority'])
    solmix=np.asarray(static['geom_solmix'])
    eligible=(priority[:,None]==priority[None,:]) & (solmix[:,None]>=1e-15) & (solmix[None,:]>=1e-15)
    g1,g2=np.nonzero(eligible)
    with np.errstate(over='ignore',invalid='ignore'):
        denominator=solmix[g1]+solmix[g2]
        upper=np.nextafter(denominator,np.inf)
    if not np.all(np.isfinite(upper)):
        raise ValueError('unsupported_numeric_range: geom_solmix denominator')


@wp.func
def _max(a:scalar.wp_real,b:scalar.wp_real)->scalar.wp_real:
    if a>=b:
        return a
    return b


@wp.func
def _min(a:scalar.wp_real,b:scalar.wp_real)->scalar.wp_real:
    if a<=b:
        return a
    return b


@wp.kernel(enable_backward=False)
def _complete(route:wp.array(dtype=wp.int32), raw:wp.array(dtype=scalar.wp_real),
              usermargin:wp.array(dtype=scalar.wp_real), active:wp.array(dtype=wp.int32),
              counts:wp.array(dtype=wp.int32),
              priority:wp.array(dtype=wp.int32),condim:wp.array(dtype=wp.int32),
              geom_margin:wp.array(dtype=scalar.wp_real),geom_gap:wp.array(dtype=scalar.wp_real),
              geom_friction:wp.array(dtype=scalar.wp_real),geom_solmix:wp.array(dtype=scalar.wp_real),
              geom_solref:wp.array(dtype=scalar.wp_real),geom_solimp:wp.array(dtype=scalar.wp_real),
              pair_margin:wp.array(dtype=scalar.wp_real),pair_gap:wp.array(dtype=scalar.wp_real),
              pair_dim:wp.array(dtype=wp.int32),pair_friction:wp.array(dtype=scalar.wp_real),
              pair_solref:wp.array(dtype=scalar.wp_real),pair_solreffriction:wp.array(dtype=scalar.wp_real),
              pair_solimp:wp.array(dtype=scalar.wp_real),
              o_margin:wp.array(dtype=scalar.wp_real),o_solref:wp.array(dtype=scalar.wp_real),
              o_solimp:wp.array(dtype=scalar.wp_real),override_enabled:wp.array(dtype=wp.int32),
              out_f:wp.array(dtype=scalar.wp_real),out_i:wp.array(dtype=wp.int32),
              slots:int,ngeom:int,npair:int):
    tid=wp.tid()
    w=tid//slots
    q=tid-w*slots
    for k in range(26):
        out_f[tid*26+k]=scalar.wp_real(0.0)
    for k in range(4):
        out_i[tid*5+k]=-1
    out_i[tid*5+4]=0
    if active[w]==0:
        return
    if counts[w]<0 or counts[w]>slots:
        out_i[tid*5+4]=1
        return
    if q>=counts[w]:
        return
    g1=route[tid*4]
    g2=route[tid*4+1]
    ipair=route[tid*4+2]
    user=route[tid*4+3]
    if g1<0 or g2<0 or g1>=ngeom or g2>=ngeom:
        out_i[tid*5+4]=2
        return
    if ipair>=npair or ipair< -1 or (user!=0 and user!=1):
        out_i[tid*5+4]=3
        return
    for k in range(10):
        out_f[tid*26+k]=raw[tid*10+k]
    margin=scalar.wp_real(0.0)
    gap=scalar.wp_real(0.0)
    dim=wp.int32(0)
    if ipair<0:
        margin=_max(geom_margin[g1],geom_margin[g2])
        gap=_max(geom_gap[g1],geom_gap[g2])
        if priority[g1]!=priority[g2]:
            gp=g2
            if priority[g1]>priority[g2]:
                gp=g1
            dim=condim[gp]
        else:
            dim=wp.max(condim[g1],condim[g2])
    else:
        margin=pair_margin[ipair]
        gap=pair_gap[ipair]
        dim=pair_dim[ipair]
    if user!=0:
        margin=usermargin[tid]
    elif override_enabled[0]!=0:
        margin=o_margin[0]
    out_f[tid*26+10]=margin
    out_f[tid*26+11]=margin-gap
    out_i[tid*5]=dim
    out_i[tid*5+1]=g1
    out_i[tid*5+2]=g2
    if dim<1 or dim>6:
        out_i[tid*5+4]=4
        return
    mix=scalar.wp_real(0.0)
    if ipair<0:
        if priority[g1]!=priority[g2]:
            gp=g2
            if priority[g1]>priority[g2]:
                gp=g1
            out_f[tid*26+12]=geom_friction[gp*3]
            out_f[tid*26+14]=geom_friction[gp*3+1]
            out_f[tid*26+16]=geom_friction[gp*3+2]
            for k in range(2):
                out_f[tid*26+17+k]=geom_solref[gp*2+k]
            for k in range(5):
                out_f[tid*26+21+k]=geom_solimp[gp*5+k]
        else:
            for k in range(3):
                out_f[tid*26+12+2*k]=_max(geom_friction[g1*3+k],geom_friction[g2*3+k])
            if geom_solmix[g1]>=scalar.wp_real(1e-15) and geom_solmix[g2]>=scalar.wp_real(1e-15):
                mix=geom_solmix[g1]/(geom_solmix[g1]+geom_solmix[g2])
            elif geom_solmix[g1]<scalar.wp_real(1e-15) and geom_solmix[g2]<scalar.wp_real(1e-15):
                mix=scalar.wp_real(0.5)
            elif geom_solmix[g1]<scalar.wp_real(1e-15):
                mix=scalar.wp_real(0.0)
            else:
                mix=scalar.wp_real(1.0)
            if geom_solref[g1*2]>scalar.wp_real(0.0) and geom_solref[g2*2]>scalar.wp_real(0.0):
                for k in range(2):
                    out_f[tid*26+17+k]=mix*geom_solref[g1*2+k]+(scalar.wp_real(1.0)-mix)*geom_solref[g2*2+k]
            else:
                for k in range(2):
                    out_f[tid*26+17+k]=_min(geom_solref[g1*2+k],geom_solref[g2*2+k])
            for k in range(5):
                out_f[tid*26+21+k]=geom_solimp[g1*5+k]*mix
            for k in range(5):
                out_f[tid*26+21+k]=out_f[tid*26+21+k]+geom_solimp[g2*5+k]*(scalar.wp_real(1.0)-mix)
        out_f[tid*26+13]=out_f[tid*26+12]
        out_f[tid*26+15]=out_f[tid*26+16]
    else:
        for k in range(5):
            out_f[tid*26+12+k]=pair_friction[ipair*5+k]
            out_f[tid*26+21+k]=pair_solimp[ipair*5+k]
        for k in range(2):
            out_f[tid*26+17+k]=pair_solref[ipair*2+k]
        if pair_solreffriction[ipair*2]!=scalar.wp_real(0.0) or pair_solreffriction[ipair*2+1]!=scalar.wp_real(0.0):
            for k in range(2):
                out_f[tid*26+19+k]=pair_solreffriction[ipair*2+k]
    for k in range(5):
        out_f[tid*26+12+k]=_max(scalar.wp_real(1.0e-5),out_f[tid*26+12+k])
    if override_enabled[0]!=0:
        for k in range(2):
            out_f[tid*26+17+k]=o_solref[k]
            out_f[tid*26+19+k]=o_solref[k]
        for k in range(5):
            out_f[tid*26+21+k]=o_solimp[k]
    if raw[tid*10]<out_f[tid*26+11]:
        out_i[tid*5+3]=0
    else:
        out_i[tid*5+3]=1


class ContactParamsWorkspace:
    def __init__(self,nworld,slots,static,device='cuda:0'):
        if nworld<1 or slots<1:raise ValueError('positive N and slots required')
        validate_static(static)
        self.nworld,self.slots,self.device=nworld,slots,device
        self.ngeom=len(static['geom_priority']);self.npair=len(static['pair_margin'])
        self.static={}
        for name,value in static.items():
            arr=np.asarray(value)
            if arr.dtype not in (np.dtype('float64'),np.dtype('int32')):
                raise ValueError(f'{name} dtype {arr.dtype}')
            dtype=scalar.wp_real if arr.dtype==np.float64 else wp.int32
            self.static[name]=wp.array(np.ascontiguousarray(arr).ravel(),dtype=dtype,device=device)
        self.route=wp.zeros(nworld*slots*4,dtype=wp.int32,device=device)
        self.raw=wp.zeros(nworld*slots*10,dtype=scalar.wp_real,device=device)
        self.usermargin=wp.zeros(nworld*slots,dtype=scalar.wp_real,device=device)
        self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.counts=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.out_f=wp.zeros(nworld*slots*26,dtype=scalar.wp_real,device=device)
        self.out_i=wp.zeros(nworld*slots*5,dtype=wp.int32,device=device)

    def upload(self,route,raw,usermargin,active,counts):
        specs=((route,(self.nworld,self.slots,4),np.int32,self.route),
               (raw,(self.nworld,self.slots,10),np.float64,self.raw),
               (usermargin,(self.nworld,self.slots),np.float64,self.usermargin),
               (active,(self.nworld,),np.int32,self.active),
               (counts,(self.nworld,),np.int32,self.counts))
        for value,shape,dtype,dst in specs:
            a=np.asarray(value)
            if a.shape!=shape or a.dtype!=dtype:
                raise ValueError(f'input shape/dtype {a.shape}/{a.dtype} != {shape}/{dtype}')
            if value is active and not np.all((a==0)|(a==1)):
                raise ValueError('active mask must contain only 0/1')
            dst.assign(np.ascontiguousarray(a).ravel())

    def launch(self,stream=None):
        s=self.static
        wp.launch(_complete,dim=self.nworld*self.slots,inputs=[
            self.route,self.raw,self.usermargin,self.active,self.counts,
            s['geom_priority'],s['geom_condim'],s['geom_margin'],s['geom_gap'],
            s['geom_friction'],s['geom_solmix'],s['geom_solref'],s['geom_solimp'],
            s['pair_margin'],s['pair_gap'],s['pair_dim'],s['pair_friction'],
            s['pair_solref'],s['pair_solreffriction'],s['pair_solimp'],
            s['o_margin'],s['o_solref'],s['o_solimp'],s['override'],
            self.out_f,self.out_i,self.slots,self.ngeom,self.npair],
            device=self.device,stream=stream)
