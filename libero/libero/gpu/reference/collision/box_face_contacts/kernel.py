"""MuJoCo 2.3.7 BoxBox face tail, original C lines 733--975.

Apache-2.0 source: engine_collision_box.c SHA256
fd038c185abfae66213bcabd0db34450a5fb596481fa5d1bb567fdff62a587b2.
The SAT tuple is the input boundary. No clipped geometry or contact is uploaded.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@wp.func
def _clear_error(stage_i: wp.array(dtype=wp.int32), stage_d: wp.array(dtype=scalar.wp_real),
                 trace_i: wp.array(dtype=wp.int32), trace_d: wp.array(dtype=scalar.wp_real),
                 raw: wp.array(dtype=scalar.wp_real), out_count: wp.array(dtype=wp.int32),
                 state: wp.array(dtype=wp.int32), frame_written: wp.array(dtype=wp.int32),
                 frame0: wp.array(dtype=scalar.wp_real), t: int, status: int, phase: int):
    for z in range(256):stage_i[t*256+z]=0
    for z in range(512):stage_d[t*512+z]=scalar.wp_real(0.0)
    for z in range(192):
        trace_i[t*192+z]=0
        trace_d[t*192+z]=scalar.wp_real(0.0)
    for z in range(500):raw[t*500+z]=scalar.wp_real(0.0)
    for z in range(6):frame0[t*6+z]=scalar.wp_real(0.0)
    out_count[t]=0
    frame_written[t]=0
    state[t*4]=0
    state[t*4+2]=status
    state[t*4+3]=phase


@wp.func
def _dot3(a: wp.array(dtype=scalar.wp_real), ao: int,
          b: wp.array(dtype=scalar.wp_real), bo: int) -> scalar.wp_real:
    return (a[ao]*b[bo] + a[ao+1]*b[bo+1]) + a[ao+2]*b[bo+2]


@wp.func
def _add_scaled(dst: wp.array(dtype=scalar.wp_real), do: int,
                src: wp.array(dtype=scalar.wp_real), so: int, scale: scalar.wp_real):
    for c in range(3):
        dst[do+c] = dst[do+c] + src[so+c]*scale


@wp.kernel(enable_backward=False)
def _face(pair: wp.array(dtype=scalar.wp_real), sat: wp.array(dtype=scalar.wp_real),
          sat_i: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
          counts: wp.array(dtype=wp.int32), point_scratch: wp.array(dtype=scalar.wp_real),
          post_scratch: wp.array(dtype=scalar.wp_real),
          stage_i: wp.array(dtype=wp.int32), stage_d: wp.array(dtype=scalar.wp_real),
          trace_i: wp.array(dtype=wp.int32), trace_d: wp.array(dtype=scalar.wp_real),
          raw: wp.array(dtype=scalar.wp_real), out_count: wp.array(dtype=wp.int32),
          state: wp.array(dtype=wp.int32), frame_written: wp.array(dtype=wp.int32),
          frame0: wp.array(dtype=scalar.wp_real), pairs: int):
    t = wp.tid()
    w = t // pairs
    q = t - w*pairs
    pb = t*31
    sb = t*45
    ib = t*6
    di = t*512
    ii = t*256
    rb = t*500
    ps = t*18
    for z in range(512):
        stage_d[di+z] = scalar.wp_real(0.0)
    for z in range(256):
        stage_i[ii+z] = 0
    for z in range(500):
        raw[rb+z] = scalar.wp_real(0.0)
    for z in range(192):
        trace_i[t*192+z] = 0
        trace_d[t*192+z] = scalar.wp_real(0.0)
    for z in range(6):
        frame0[t*6+z] = scalar.wp_real(0.0)
    for z in range(18):
        point_scratch[ps+z] = scalar.wp_real(0.0)
    out_count[t] = 0
    frame_written[t] = 0
    state[t*4] = 0
    state[t*4+1] = 0
    state[t*4+2] = 0
    state[t*4+3] = 0
    if active[w] == 0:
        return
    if active[w] != 1 or counts[w] < 0 or counts[w] > pairs:
        state[t*4+2] = 1
        return
    if q >= counts[w]:
        state[t*4+1] = 3
        return
    route = sat_i[ib+4]
    code = sat_i[ib]
    state[t*4+1] = route
    if (route != 0 and route != 1 and route != 2) or sat_i[ib+5] != 0:
        state[t*4+2] = 2
        return
    if (route == 0 and (code != -1 or sat_i[ib+1] != -1 or sat_i[ib+2] != -1 or sat_i[ib+3] != -1)) or (route == 1 and (code < 0 or code > 11 or sat_i[ib+1] != -1 or sat_i[ib+2] != -1 or sat_i[ib+3] != -1)) or (route == 2 and (code < 12 or code > 20 or sat_i[ib+1] < 0 or sat_i[ib+1] > 7 or sat_i[ib+2] < 0 or sat_i[ib+2] > 7 or sat_i[ib+3] < 0 or sat_i[ib+3] > 1)):
        state[t*4+2] = 2
        return
    if route != 1:
        return
    for z in range(31):
        if not wp.isfinite(pair[pb+z]):
            state[t*4+2]=2
            state[t*4+3]=1
            return
    for z in range(45):
        if not wp.isfinite(sat[sb+z]):
            state[t*4+2]=2
            state[t*4+3]=1
            return
    state[t*4] = 1
    stage_i[ii] = code
    q1 = code % 6
    q2 = code // 6
    stage_i[ii+1] = q1
    stage_i[ii+2] = q2
    # rotmore is a signed permutation, including source +0 initialization.
    if q1 == 0:
        stage_d[di+2]=-scalar.wp_real(1.0); stage_d[di+4]=scalar.wp_real(1.0); stage_d[di+6]=scalar.wp_real(1.0)
    elif q1 == 1:
        stage_d[di]=scalar.wp_real(1.0); stage_d[di+5]=-scalar.wp_real(1.0); stage_d[di+7]=scalar.wp_real(1.0)
    elif q1 == 2:
        stage_d[di]=scalar.wp_real(1.0); stage_d[di+4]=scalar.wp_real(1.0); stage_d[di+8]=scalar.wp_real(1.0)
    elif q1 == 3:
        stage_d[di+2]=scalar.wp_real(1.0); stage_d[di+4]=scalar.wp_real(1.0); stage_d[di+6]=-scalar.wp_real(1.0)
    elif q1 == 4:
        stage_d[di]=scalar.wp_real(1.0); stage_d[di+5]=scalar.wp_real(1.0); stage_d[di+7]=-scalar.wp_real(1.0)
    else:
        stage_d[di]=-scalar.wp_real(1.0); stage_d[di+4]=scalar.wp_real(1.0); stage_d[di+8]=-scalar.wp_real(1.0)
    i0=0; i1=1; i2=2
    f0=scalar.wp_real(1.0); f1=scalar.wp_real(1.0); f2=scalar.wp_real(1.0)
    if q1 == 0:
        i0=2; f0=-scalar.wp_real(1.0); i2=0
    elif q1 == 1:
        i1=2; f1=-scalar.wp_real(1.0); i2=1
    elif q1 == 3:
        i0=2; i2=0; f2=-scalar.wp_real(1.0)
    elif q1 == 4:
        i1=2; i2=1; f2=-scalar.wp_real(1.0)
    elif q1 == 5:
        f0=-scalar.wp_real(1.0); f2=-scalar.wp_real(1.0)
    stage_i[ii+12]=i0; stage_i[ii+13]=i1; stage_i[ii+14]=i2
    # sat layout: pos21, pos12, rot, rott, rotabs, rottabs, clnorm.
    # q2=1 calls mju_mulMatMatT (mju_dot starts with an add of +0).
    if q2 != 0:
        for i in range(3):
            for j in range(3):
                x=((stage_d[di+3*i]*sat[sb+6+3*j] +
                    stage_d[di+3*i+1]*sat[sb+6+3*j+1]) +
                    stage_d[di+3*i+2]*sat[sb+6+3*j+2])
                stage_d[di+12+3*i+j]=scalar.wp_real(0.0)+x
        stage_d[di+9]=sat[sb+3+i0]*f0
        stage_d[di+10]=sat[sb+3+i1]*f1
        stage_d[di+11]=sat[sb+3+i2]*f2
        stage_d[di+30]=pair[pb+1]
        stage_d[di+31]=pair[pb+2]
        stage_d[di+32]=pair[pb+3]
        stage_d[di+33]=wp.abs(pair[pb+16+i0]*f0)
        stage_d[di+34]=wp.abs(pair[pb+16+i1]*f1)
        stage_d[di+35]=wp.abs(pair[pb+16+i2]*f2)
    else:
        for j in range(3):
            stage_d[di+12+j]=sat[sb+6+3*i0+j]*f0
            stage_d[di+15+j]=sat[sb+6+3*i1+j]*f1
            stage_d[di+18+j]=sat[sb+6+3*i2+j]*f2
        stage_d[di+9]=sat[sb+i0]*f0
        stage_d[di+10]=sat[sb+i1]*f1
        stage_d[di+11]=sat[sb+i2]*f2
        for j in range(3):
            stage_d[di+30+j]=pair[pb+16+j]
        stage_d[di+33]=wp.abs(pair[pb+1+i0]*f0)
        stage_d[di+34]=wp.abs(pair[pb+1+i1]*f1)
        stage_d[di+35]=wp.abs(pair[pb+1+i2]*f2)
    for i in range(3):
        for j in range(3):
            stage_d[di+21+3*i+j]=stage_d[di+12+3*j+i]
    lx=stage_d[di+33]; ly=stage_d[di+34]; hz=stage_d[di+35]
    stage_d[di+36]=lx; stage_d[di+37]=ly; stage_d[di+38]=hz
    stage_d[di+11]=stage_d[di+11]-hz
    clcorner=wp.int32(0)
    for i in range(3):
        if stage_d[di+18+i] < scalar.wp_real(0.0):
            clcorner=clcorner+(1<<i)
    stage_i[ii+3]=clcorner
    for j in range(3):
        point_scratch[ps+j]=stage_d[di+9+j]
    for i in range(3):
        sign=scalar.wp_real(-1.0)
        if (clcorner & (1<<i)) != 0:
            sign=scalar.wp_real(1.0)
        scale=stage_d[di+30+i]*sign
        for j in range(3):
            point_scratch[ps+j]=point_scratch[ps+j]+stage_d[di+21+3*i+j]*scale
    m=wp.int32(1)
    for i in range(3):
        if wp.abs(stage_d[di+18+i]) < scalar.wp_real(0.5):
            sign=scalar.wp_real(2.0)
            if (clcorner & (1<<i)) != 0:
                sign=-scalar.wp_real(2.0)
            scale=stage_d[di+30+i]*sign
            for j in range(3):
                point_scratch[ps+3*m+j]=stage_d[di+21+3*i+j]*scale
            m=m+1
    if m>3:
        _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,2,2)
        return
    for j in range(3):
        point_scratch[ps+9+j]=point_scratch[ps+j]+point_scratch[ps+3+j]
        point_scratch[ps+12+j]=point_scratch[ps+j]+point_scratch[ps+6+j]
        point_scratch[ps+15+j]=point_scratch[ps+9+j]+point_scratch[ps+6+j]
    stage_i[ii+4]=m
    for j in range(3*m):
        stage_d[di+39+j]=point_scratch[ps+j]
    k=wp.int32(0)
    if m>1:
        for j in range(3):
            stage_d[di+57+j]=point_scratch[ps+j]
            stage_d[di+60+j]=point_scratch[ps+3+j]
        k=1
    if m>2:
        for z in range(3):
            start=0
            direction=2
            if z==1:
                start=3
            if z==2:
                start=4; direction=1
            for j in range(3):
                stage_d[di+57+6*(k+z)+j]=point_scratch[ps+3*start+j]
                stage_d[di+60+6*(k+z)+j]=point_scratch[ps+3*direction+j]
        k=4
    stage_i[ii+5]=k
    n=wp.int32(0)
    for i in range(k):
        for axis in range(2):
            a=stage_d[di+57+6*i+axis]
            b=stage_d[di+60+6*i+axis]
            c=stage_d[di+57+6*i+1-axis]
            d=stage_d[di+60+6*i+1-axis]
            for j in range(2):
                trace_d[t*192+4*(4*i+2*axis+j)]=a
                trace_d[t*192+4*(4*i+2*axis+j)+1]=b
            if wp.abs(b)>scalar.wp_real(1.0e-15):
                if not wp.isfinite(b):
                    _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,3)
                    return
                for j in range(2):
                    tr=t*192+4*(4*i+2*axis+j)
                    trace_i[tr]=1
                    extent=lx
                    other=ly
                    if axis==1:
                        extent=ly; other=lx
                    side=-scalar.wp_real(1.0)
                    if j==1:
                        side=scalar.wp_real(1.0)
                    l=extent*side
                    c1=(l-a)*(scalar.wp_real(1.0)/b)
                    trace_d[tr+2]=c1
                    if c1<scalar.wp_real(0.0) or c1>scalar.wp_real(1.0):
                        trace_i[tr+1]=1
                        stage_i[ii+16]=stage_i[ii+16]+1
                        continue
                    c2=c+d*c1
                    trace_d[tr+3]=c2
                    if wp.abs(c2)>other:
                        trace_i[tr+1]=2
                        stage_i[ii+17]=stage_i[ii+17]+1
                        continue
                    if n>=50:
                        _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,3,3)
                        return
                    for h in range(3):
                        stage_d[di+81+3*n+h]=stage_d[di+57+6*i+h]
                    for h in range(3):
                        at=di+81+3*n+h
                        stage_d[at]=stage_d[at]+stage_d[di+60+6*i+h]*c1
                    trace_i[tr+2]=n
                    n=n+1
    a=point_scratch[ps+3]; b=point_scratch[ps+6]
    c=point_scratch[ps+4]; d=point_scratch[ps+7]
    det=a*d-b*c
    if m>2:
        if not wp.isfinite(det) or det==scalar.wp_real(0.0):
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,4)
            return
        for i in range(4):
            llx=-lx; lly=-ly
            if i//2!=0:
                llx=lx
            if i%2!=0:
                lly=ly
            x=llx-point_scratch[ps]
            y=lly-point_scratch[ps+1]
            u=(x*d-y*b)*(scalar.wp_real(1.0)/det)
            v=(y*a-x*c)*(scalar.wp_real(1.0)/det)
            tr=t*192+4*(16+i)
            trace_i[tr]=1
            trace_d[tr]=x;trace_d[tr+1]=y;trace_d[tr+2]=u;trace_d[tr+3]=v
            if u<=scalar.wp_real(0.0) or v<=scalar.wp_real(0.0) or u>=scalar.wp_real(1.0) or v>=scalar.wp_real(1.0):
                trace_i[tr+1]=1
                stage_i[ii+18]=stage_i[ii+18]+1
                continue
            if n>=50:
                _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,3,4)
                return
            stage_d[di+81+3*n]=llx
            stage_d[di+82+3*n]=lly
            stage_d[di+83+3*n]=(point_scratch[ps+2]+u*point_scratch[ps+5])+v*point_scratch[ps+8]
            trace_i[tr+2]=n
            n=n+1
    for i in range(1<<(m-1)):
        src=0
        if i!=0:
            src=i+2
        x=point_scratch[ps+3*src]
        y=point_scratch[ps+3*src+1]
        tr=t*192+4*(20+i)
        trace_i[tr]=1
        trace_d[tr]=x;trace_d[tr+1]=y;trace_d[tr+2]=point_scratch[ps+3*src+2]
        # Vertex zero retains boundary ownership, but must also be clipped.
        if x < -lx or x > lx or (i!=0 and (x==-lx or x==lx)):
            trace_i[tr+1]=1
            stage_i[ii+19]=stage_i[ii+19]+1
            continue
        if y < -ly or y > ly or (i!=0 and (y==-ly or y==ly)):
            trace_i[tr+1]=2
            stage_i[ii+20]=stage_i[ii+20]+1
            continue
        if n>=50:
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,3,5)
            return
        for j in range(3):
            stage_d[di+81+3*n+j]=point_scratch[ps+3*src+j]
        trace_i[tr+2]=n
        n=n+1
    stage_i[ii+6]=n
    accepted=wp.int32(0)
    for i in range(n):
        tr=t*192+4*(24+i)
        trace_i[tr]=1
        trace_d[tr]=stage_d[di+81+3*i+2]
        trace_d[tr+1]=pair[pb]
        if stage_d[di+81+3*i+2]>pair[pb]:
            trace_i[tr+1]=1
            stage_i[ii+21]=stage_i[ii+21]+1
            continue
        for j in range(3):
            stage_d[di+231+3*accepted+j]=stage_d[di+81+3*i+j]
        stage_d[di+233+3*accepted]=stage_d[di+233+3*accepted]*scalar.wp_real(0.5)
        trace_d[tr+2]=stage_d[di+233+3*accepted]
        trace_i[tr+2]=accepted
        for j in range(3):
            post_scratch[t*150+3*accepted+j]=stage_d[di+231+3*accepted+j]
        accepted=accepted+1
    stage_i[ii+7]=accepted
    out_count[t]=accepted
    # final world transform is independent of clipping p/r.
    mat=pb+7
    pos=pb+4
    if q2!=0:
        mat=pb+22; pos=pb+19
    for i in range(3):
        for j in range(3):
            x=((pair[mat+3*i]*stage_d[di+3*j] +
                pair[mat+3*i+1]*stage_d[di+3*j+1]) +
                pair[mat+3*i+2]*stage_d[di+3*j+2])
            stage_d[di+403+3*i+j]=scalar.wp_real(0.0)+x
        stage_d[di+400+i]=pair[pos+i]
    sign=scalar.wp_real(1.0)
    if q2!=0:
        sign=-scalar.wp_real(1.0)
    for i in range(3):
        frame0[t*6+i]=sign*stage_d[di+403+3*i+2]
    frame_written[t]=1
    for i in range(accepted):
        depth_half=post_scratch[t*150+3*i+2]
        raw[rb+10*i]=depth_half
        post_scratch[t*150+3*i+2]=depth_half+hz
        for j in range(3):
            # mju_rotVecMat's three multiply / two add association.
            x=((stage_d[di+403+3*j]*post_scratch[t*150+3*i] +
                stage_d[di+403+3*j+1]*post_scratch[t*150+3*i+1]) +
                stage_d[di+403+3*j+2]*post_scratch[t*150+3*i+2])
            raw[rb+10*i+1+j]=x+stage_d[di+400+j]
        for j in range(6):
            raw[rb+10*i+4+j]=frame0[t*6+j]
    for z in range(512):
        if not wp.isfinite(stage_d[di+z]):
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,9)
            return
    for z in range(192):
        if not wp.isfinite(trace_d[t*192+z]):
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,9)
            return
    for z in range(500):
        if not wp.isfinite(raw[rb+z]):
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,9)
            return
    for z in range(6):
        if not wp.isfinite(frame0[t*6+z]):
            _clear_error(stage_i,stage_d,trace_i,trace_d,raw,out_count,state,frame_written,frame0,t,4,9)
            return


class BoxFaceContactWorkspace:
    """All device buffers preallocated. launch reads only device arrays."""

    def __init__(self, nworld: int, pairs: int, device: str = "cuda:0"):
        if nworld<1 or pairs<1:
            raise ValueError("positive N and P required")
        self.nworld=nworld; self.pairs=pairs; self.device=device
        npair=nworld*pairs
        self.pair_f=wp.zeros(npair*31,dtype=scalar.wp_real,device=device)
        self.sat_f=wp.zeros(npair*45,dtype=scalar.wp_real,device=device)
        self.sat_i=wp.zeros(npair*6,dtype=wp.int32,device=device)
        self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.counts=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.point_scratch=wp.zeros(npair*18,dtype=scalar.wp_real,device=device)
        self.post_scratch=wp.zeros(npair*150,dtype=scalar.wp_real,device=device)
        self.stage_i=wp.zeros(npair*256,dtype=wp.int32,device=device)
        self.stage_d=wp.zeros(npair*512,dtype=scalar.wp_real,device=device)
        self.trace_i=wp.zeros(npair*192,dtype=wp.int32,device=device)
        self.trace_d=wp.zeros(npair*192,dtype=scalar.wp_real,device=device)
        self.raw=wp.zeros(npair*500,dtype=scalar.wp_real,device=device)
        self.count=wp.zeros(npair,dtype=wp.int32,device=device)
        self.state=wp.zeros(npair*4,dtype=wp.int32,device=device)
        self.frame_written=wp.zeros(npair,dtype=wp.int32,device=device)
        self.frame0=wp.zeros(npair*6,dtype=scalar.wp_real,device=device)

    def upload(self,pair_f,sat_f,sat_i,active,counts):
        expected=((pair_f,(self.nworld,self.pairs,31),np.dtype('float64'),'pair_f',self.pair_f),
                  (sat_f,(self.nworld,self.pairs,45),np.dtype('float64'),'sat_f',self.sat_f),
                  (sat_i,(self.nworld,self.pairs,6),np.dtype('int32'),'sat_i',self.sat_i),
                  (active,(self.nworld,),np.dtype('int32'),'active',self.active),
                  (counts,(self.nworld,),np.dtype('int32'),'counts',self.counts))
        vals={}
        for val,shape,dtype,name,dst in expected:
            arr=np.asarray(val)
            if arr.shape!=shape or arr.dtype!=dtype or not arr.flags.c_contiguous:
                raise ValueError(f'{name}: expected contiguous {shape} {dtype}, got {arr.shape} {arr.dtype}')
            vals[name]=arr
        if np.any((vals['active']!=0)&(vals['active']!=1)):
            raise ValueError('active must be 0/1')
        for w in range(self.nworld):
            if vals['active'][w]==0:
                continue
            count=int(vals['counts'][w])
            if not 0<=count<=self.pairs:
                continue  # device emits status 1 for every pair
            if not np.isfinite(vals['pair_f'][w,:count]).all() or not np.isfinite(vals['sat_f'][w,:count]).all():
                raise ValueError('nonfinite active input')
            for j in range(count):
                code,cle1,cle2,inside,route,status=(int(x) for x in vals['sat_i'][w,j])
                if status!=0 or route not in (0,1,2) or (route==0 and (code,cle1,cle2,inside)!=(-1,-1,-1,-1)) or (route==1 and (not 0<=code<=11 or (cle1,cle2,inside)!=(-1,-1,-1))) or (route==2 and (not 12<=code<=20 or not 0<=cle1<=7 or not 0<=cle2<=7 or inside not in (0,1))):
                    raise ValueError(f'sat_i domain world={w} pair={j}')
        for _,_,_,name,dst in expected:
            dst.assign(vals[name].ravel())

    def launch(self,stream=None):
        wp.launch(_face,dim=self.nworld*self.pairs,
                  inputs=[self.pair_f,self.sat_f,self.sat_i,self.active,self.counts,
                          self.point_scratch,self.post_scratch,self.stage_i,self.stage_d,
                          self.trace_i,self.trace_d,self.raw,
                          self.count,self.state,self.frame_written,self.frame0,self.pairs],
                  device=self.device,stream=stream)

    def download(self):
        n,p=self.nworld,self.pairs
        return dict(count=self.count.numpy().reshape(n,p),
                    raw=self.raw.numpy().reshape(n,p,50,10),
                    stage_i=self.stage_i.numpy().reshape(n,p,256),
                    stage_d=self.stage_d.numpy().reshape(n,p,512),
                    trace_i=self.trace_i.numpy().reshape(n,p,48,4),
                    trace_d=self.trace_d.numpy().reshape(n,p,48,4),
                    state=self.state.numpy().reshape(n,p,4),
                    frame_written=self.frame_written.numpy().reshape(n,p),
                    frame0=self.frame0.numpy().reshape(n,p,6))
