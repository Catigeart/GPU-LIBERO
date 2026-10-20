"""Device migration of engine_collision_driver.c:855-938 through narrowphase call.

One thread handles one world in source request order.  The actual narrowphase
belongs to a downstream stage; a call is emitted even when it returns 0 later.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

# Default MuJoCo 2.3.7 mjCOLLISIONFUNC table, before any user modification.
DEFAULT_PRESENT=np.array([
    [0,0,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1],
    [0,0,0,1,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,1]],dtype=np.int32)


def validate_static(static:dict, *, callback=None):
    if callback is not None:
        raise ValueError('unsupported_callback: mjcb_contactfilter must be NULL')
    fields={'geom_type':np.int32,'geom_contype':np.int32,'geom_conaffinity':np.int32,
        'geom_priority':np.int32,'geom_condim':np.int32,'geom_margin':np.float64,
        'geom_gap':np.float64,'geom_rbound':np.float64,
        'pair_geom1':np.int32,'pair_geom2':np.int32,'pair_dim':np.int32,
        'pair_margin':np.float64,'pair_gap':np.float64,
        'override':np.int32,'o_margin':np.float64,'dispatch_present':np.int32}
    if set(static)!=set(fields):
        raise ValueError('static schema fields')
    g=len(static['geom_type']); q=len(static['pair_geom1'])
    if g<1:raise ValueError('no geoms')
    for name,dtype in fields.items():
        a=np.asarray(static[name])
        shape=(8,8) if name=='dispatch_present' else (1,) if name in ('override','o_margin') else (q,) if name.startswith('pair_') else (g,)
        if a.shape!=shape or a.dtype!=np.dtype(dtype):
            raise ValueError(f'static {name} shape/dtype: {a.shape}/{a.dtype}')
        if dtype==np.float64 and (not np.all(np.isfinite(a)) or np.any(np.abs(a)>1e100)):
            raise ValueError('unsupported_numeric_range: '+name)
    if not np.array_equal(static['dispatch_present'],DEFAULT_PRESENT):
        raise ValueError('unsupported_dispatch_table: nondefault function table')
    if np.any((static['geom_type']<0)|(static['geom_type']>7)):
        raise ValueError('unsupported geom type')
    if np.any(static['pair_geom1']<0) or np.any(static['pair_geom1']>=g) or np.any(static['pair_geom2']<0) or np.any(static['pair_geom2']>=g):
        raise ValueError('pair geom ID outside model')
    if static['override'][0] not in (0,1):raise ValueError('override must be 0/1')


@wp.func
def _max_source(a:scalar.wp_real,b:scalar.wp_real)->scalar.wp_real:
    if a>=b:return a
    return b


@wp.func
def _admitted(a:scalar.wp_real)->bool:
    return a==a and wp.abs(a)<=scalar.wp_real(scalar.safe_magnitude)


@wp.kernel(enable_backward=False)
def _dispatch(raw:wp.array(dtype=wp.int32), usermargin:wp.array(dtype=scalar.wp_real),
              xpos:wp.array(dtype=scalar.wp_real),xmat:wp.array(dtype=scalar.wp_real),
              active:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),
              geom_type:wp.array(dtype=wp.int32),geom_contype:wp.array(dtype=wp.int32),
              geom_conaffinity:wp.array(dtype=wp.int32),geom_priority:wp.array(dtype=wp.int32),
              geom_condim:wp.array(dtype=wp.int32),geom_margin:wp.array(dtype=scalar.wp_real),
              geom_gap:wp.array(dtype=scalar.wp_real),geom_rbound:wp.array(dtype=scalar.wp_real),
              pair_geom1:wp.array(dtype=wp.int32),pair_geom2:wp.array(dtype=wp.int32),
              pair_dim:wp.array(dtype=wp.int32),pair_margin:wp.array(dtype=scalar.wp_real),
              pair_gap:wp.array(dtype=scalar.wp_real),override:wp.array(dtype=wp.int32),
              o_margin:wp.array(dtype=scalar.wp_real),present:wp.array(dtype=wp.int32),
              out_i:wp.array(dtype=wp.int32),out_f:wp.array(dtype=scalar.wp_real),
              call_index:wp.array(dtype=wp.int32),call_count:wp.array(dtype=wp.int32),
              mid_count:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32),
              error_index:wp.array(dtype=wp.int32),slots:int,capacity:int,ngeom:int,npair:int):
    w=wp.tid()
    n=counts[w]
    status[w]=0
    error_index[w]=-1
    call_count[w]=0
    mid_count[w]=0
    if n<0 or n>slots:
        status[w]=1
        return
    if active[w]!=0 and active[w]!=1:
        status[w]=6
        return
    if active[w]==0:
        return
    for q in range(n):
        t=w*slots+q
        g1=raw[t*3]
        g2=raw[t*3+1]
        user=raw[t*3+2]
        ipair=-1
        if g2<0:
            ipair=g1
            if ipair<0 or ipair>=npair:
                status[w]=2;error_index[w]=q;return
            g1=pair_geom1[ipair]
            g2=pair_geom2[ipair]
        if g1<0 or g1>=ngeom or g2<0 or g2>=ngeom:
            status[w]=2;error_index[w]=q;return
        if user!=0 and user!=1:
            status[w]=3;error_index[w]=q;return
        if user!=0 and not _admitted(usermargin[t]):
            status[w]=5;error_index[w]=q;return
        type1=geom_type[g1]
        type2=geom_type[g2]
        if type1>type2:
            old=g1;g1=g2;g2=old
            old=type1;type1=type2;type2=old
        b=t*12
        out_i[b]=0          # reason: 0 call, 1 empty slot, 2 mask, 3 sphere
        out_i[b+1]=0        # called
        out_i[b+2]=g1
        out_i[b+3]=g2
        out_i[b+4]=type1
        out_i[b+5]=type2
        out_i[b+7]=ipair
        out_i[b+8]=mid_count[w]
        out_i[b+9]=mid_count[w]
        out_i[b+10]=0       # parameter write mask
        out_i[b+11]=-1      # compact index
        if present[type1*8+type2]==0:
            out_i[b]=1
            continue
        if ipair<0 and user==0:
            if (geom_contype[g1]&geom_conaffinity[g2])==0 and (geom_contype[g2]&geom_conaffinity[g1])==0:
                out_i[b]=2
                continue
        margin=scalar.wp_real(0.0)
        gap=scalar.wp_real(0.0)
        dim=wp.int32(0)
        if ipair<0:
            margin=_max_source(geom_margin[g1],geom_margin[g2])
            gap=_max_source(geom_gap[g1],geom_gap[g2])
            if geom_priority[g1]!=geom_priority[g2]:
                gp=g2
                if geom_priority[g1]>geom_priority[g2]:gp=g1
                dim=geom_condim[gp]
            else:
                dim=geom_condim[g2]
                if geom_condim[g1]>geom_condim[g2]:dim=geom_condim[g1]
        else:
            margin=pair_margin[ipair]
            gap=pair_gap[ipair]
            dim=pair_dim[ipair]
        if user!=0:margin=usermargin[t]
        elif override[0]!=0:margin=o_margin[0]
        out_i[b+6]=dim
        out_i[b+10]=1
        out_f[t*2]=margin
        out_f[t*2+1]=gap
        r1=geom_rbound[g1]
        r2=geom_rbound[g2]
        rejected=0
        if r1>scalar.wp_real(0.0) and r2>scalar.wp_real(0.0):
            bound=r1+r2+margin
            p1=(w*ngeom+g1)*3
            p2=(w*ngeom+g2)*3
            for k in range(3):
                if not _admitted(xpos[p1+k]) or not _admitted(xpos[p2+k]):
                    status[w]=5;error_index[w]=q;return
            dx=xpos[p1]-xpos[p2]
            dy=xpos[p1+1]-xpos[p2+1]
            dz=xpos[p1+2]-xpos[p2+2]
            dist2=dx*dx+dy*dy+dz*dz
            if dist2>bound*bound:rejected=1
        if rejected==0 and type1==0 and r2>scalar.wp_real(0.0):
            p1=(w*ngeom+g1)*3;p2=(w*ngeom+g2)*3;m1=(w*ngeom+g1)*9
            for k in range(3):
                if not _admitted(xpos[p1+k]) or not _admitted(xpos[p2+k]):
                    status[w]=5;error_index[w]=q;return
                if not _admitted(xmat[m1+2+3*k]):
                    status[w]=5;error_index[w]=q;return
            dx=xpos[p2]-xpos[p1]
            dy=xpos[p2+1]-xpos[p1+1]
            dz=xpos[p2+2]-xpos[p1+2]
            plane=dx*xmat[m1+2]+dy*xmat[m1+5]+dz*xmat[m1+8]
            if plane>margin+r2:rejected=1
        if rejected==0 and type2==0 and r1>scalar.wp_real(0.0):
            p1=(w*ngeom+g2)*3;p2=(w*ngeom+g1)*3;m1=(w*ngeom+g2)*9
            for k in range(3):
                if not _admitted(xpos[p1+k]) or not _admitted(xpos[p2+k]):
                    status[w]=5;error_index[w]=q;return
                if not _admitted(xmat[m1+2+3*k]):
                    status[w]=5;error_index[w]=q;return
            dx=xpos[p2]-xpos[p1]
            dy=xpos[p2+1]-xpos[p1+1]
            dz=xpos[p2+2]-xpos[p1+2]
            plane=dx*xmat[m1+2]+dy*xmat[m1+5]+dz*xmat[m1+8]
            if plane>margin+r1:rejected=1
        if rejected!=0:
            out_i[b]=3
            continue
        index=call_count[w]
        if index>=capacity:
            status[w]=4;error_index[w]=q;call_count[w]=-1;mid_count[w]=-1;return
        call_index[w*capacity+index]=q
        call_count[w]=index+1
        mid_count[w]=mid_count[w]+1
        out_i[b+1]=1
        out_i[b+9]=mid_count[w]
        out_i[b+11]=index


class CollisionDispatchWorkspace:
    def __init__(self,nworld:int,slots:int,capacity:int,static:dict,*,device='cuda:0',callback=None):
        if nworld<1 or slots<1 or capacity<1 or capacity>slots:
            raise ValueError('positive N, slots and 1<=capacity<=slots required')
        validate_static(static,callback=callback)
        self.nworld=nworld;self.slots=slots;self.capacity=capacity;self.device=device
        self.ngeom=len(static['geom_type']);self.npair=len(static['pair_geom1'])
        self.static={}
        for name,a in static.items():
            dtype=scalar.wp_real if a.dtype==np.float64 else wp.int32
            self.static[name]=(wp.array(np.ascontiguousarray(a).ravel(),dtype=dtype,device=device)
                               if a.size else wp.zeros(1,dtype=dtype,device=device))
        self.raw=wp.zeros(nworld*slots*3,dtype=wp.int32,device=device)
        self.usermargin=wp.zeros(nworld*slots,dtype=scalar.wp_real,device=device)
        self.xpos=wp.zeros(nworld*self.ngeom*3,dtype=scalar.wp_real,device=device)
        self.xmat=wp.zeros(nworld*self.ngeom*9,dtype=scalar.wp_real,device=device)
        self.active=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.counts=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.out_i=wp.zeros(nworld*slots*12,dtype=wp.int32,device=device)
        self.out_f=wp.zeros(nworld*slots*2,dtype=scalar.wp_real,device=device)
        self.call_index=wp.zeros(nworld*capacity,dtype=wp.int32,device=device)
        self.call_count=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.mid_count=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.status=wp.zeros(nworld,dtype=wp.int32,device=device)
        self.error_index=wp.zeros(nworld,dtype=wp.int32,device=device)

    def upload(self,raw,usermargin,xpos,xmat,active,counts):
        specs=((raw,(self.nworld,self.slots,3),np.int32,self.raw),
               (usermargin,(self.nworld,self.slots),np.float64,self.usermargin),
               (xpos,(self.nworld,self.ngeom,3),np.float64,self.xpos),
               (xmat,(self.nworld,self.ngeom,9),np.float64,self.xmat),
               (active,(self.nworld,),np.int32,self.active),
               (counts,(self.nworld,),np.int32,self.counts))
        checked=[]
        for value,shape,dtype,dst in specs:
            a=np.asarray(value)
            if a.shape!=shape or a.dtype!=np.dtype(dtype):
                raise ValueError(f'input dtype/shape {a.shape}/{a.dtype} != {shape}/{dtype}')
            if dtype==np.float64 and (not np.all(np.isfinite(a)) or np.any(np.abs(a)>1e100)):
                raise ValueError('unsupported_numeric_range: nonfinite input')
            checked.append((np.ascontiguousarray(a).ravel(),dst))
        if not np.all((active==0)|(active==1)):
            raise ValueError('active must be int32 0/1')
        for value,dst in checked:
            dst.assign(value)

    def launch(self,stream=None):
        s=self.static
        wp.launch(_dispatch,dim=self.nworld,inputs=[
            self.raw,self.usermargin,self.xpos,self.xmat,self.active,self.counts,
            s['geom_type'],s['geom_contype'],s['geom_conaffinity'],s['geom_priority'],
            s['geom_condim'],s['geom_margin'],s['geom_gap'],s['geom_rbound'],
            s['pair_geom1'],s['pair_geom2'],s['pair_dim'],s['pair_margin'],s['pair_gap'],
            s['override'],s['o_margin'],s['dispatch_present'],
            self.out_i,self.out_f,self.call_index,self.call_count,self.mid_count,
            self.status,self.error_index,self.slots,self.capacity,self.ngeom,self.npair],
            device=self.device,stream=stream)
