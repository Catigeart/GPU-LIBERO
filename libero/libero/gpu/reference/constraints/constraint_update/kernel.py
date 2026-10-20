"""Device implementation of MuJoCo 2.3.7 mj_constraintUpdate, dense J route.

One owner thread per world updates each cone block, its state and its Hessian.
The second kernel reduces J-transpose in original row order, skipping exact
zero force. Neither kernel allocates or transfers during production launch.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.func
def _flip_sign(x:scalar.wp_real)->scalar.wp_real:
    # Warp's unary neg lowers to +0-x, which loses the sign of a zero input.
    # Multiplication by exactly -1 flips the IEEE binary64 sign bit for every
    # finite x, including either zero and subnormal values.
    return scalar.wp_real(-1.0)*x

@wp.func
def _u(jar:wp.array(dtype=scalar.wp_real),fr:wp.array(dtype=scalar.wp_real),
       rb:int,fb:int,j:int)->scalar.wp_real:
    return jar[rb+j]*fr[fb+j-1]

@wp.func
def _cone_t(jar:wp.array(dtype=scalar.wp_real),fr:wp.array(dtype=scalar.wp_real),
            rb:int,fb:int,d:int)->scalar.wp_real:
    # The source mju_dot grouping must be identical in the preflight and
    # the numeric update so that a near-boundary check cannot take another path.
    dot=scalar.wp_real(0.0)
    if d==1:
        dot=scalar.wp_real(0.0)
    elif d==3:
        u1=_u(jar,fr,rb,fb,1);u2=_u(jar,fr,rb,fb,2)
        dot=dot+(u1*u1+u2*u2)
    elif d==4:
        u1=_u(jar,fr,rb,fb,1);u2=_u(jar,fr,rb,fb,2);u3=_u(jar,fr,rb,fb,3)
        dot=dot+(u1*u1+u2*u2+u3*u3)
    else:
        u1=_u(jar,fr,rb,fb,1);u2=_u(jar,fr,rb,fb,2)
        u3=_u(jar,fr,rb,fb,3);u4=_u(jar,fr,rb,fb,4)
        u5=_u(jar,fr,rb,fb,5)
        dot=((u1*u1+u3*u3)+(u2*u2+u4*u4))+u5*u5
    return wp.sqrt(dot)

@wp.kernel(enable_backward=False)
def _update(active:wp.array(dtype=wp.int32),count:wp.array(dtype=wp.int32),
            ne:wp.array(dtype=wp.int32),nf:wp.array(dtype=wp.int32),
            ncon:wp.array(dtype=wp.int32),flag:wp.array(dtype=wp.int32),
            jar:wp.array(dtype=scalar.wp_real),D:wp.array(dtype=scalar.wp_real),
            R:wp.array(dtype=scalar.wp_real),loss:wp.array(dtype=scalar.wp_real),
            typ:wp.array(dtype=wp.int32),cid:wp.array(dtype=wp.int32),
            J:wp.array(dtype=scalar.wp_real),
            mu:wp.array(dtype=scalar.wp_real),fr:wp.array(dtype=scalar.wp_real),
            dim:wp.array(dtype=wp.int32),force:wp.array(dtype=scalar.wp_real),
            state:wp.array(dtype=wp.int32),H:wp.array(dtype=scalar.wp_real),
            cost:wp.array(dtype=scalar.wp_real),status:wp.array(dtype=wp.int32),
            rows:int,contacts:int,nv:int):
    w=wp.tid()
    status[w]=0
    if active[w]==0:return
    if active[w]!=1:status[w]=1;return
    n=count[w];nc=ncon[w];eq=ne[w];fcount=nf[w]
    if n<0 or n>rows or nc<0 or nc>contacts or (flag[w]!=0 and flag[w]!=1):
        status[w]=2;return
    if eq<0 or eq>n:
        status[w]=2;return
    if fcount<0 or fcount>n-eq:
        status[w]=2;return
    rb=w*rows;cb=w*contacts
    # Validate all structural routes before mutating any output.
    i=eq+fcount
    while i<n:
        t=typ[rb+i]
        if t==7:
            c=cid[rb+i]
            if c<0 or c>=nc:status[w]=3;return
            d=dim[cb+c]
            if (d!=1 and d!=3 and d!=4 and d!=6) or i+d>n or mu[cb+c]<=scalar.wp_real(0.0) or not wp.isfinite(mu[cb+c]):
                status[w]=4;return
            for j in range(d):
                if typ[rb+i+j]!=7 or cid[rb+i+j]!=c:
                    status[w]=5;return
            for j in range(d-1):
                if fr[(cb+c)*5+j]<=scalar.wp_real(0.0) or not wp.isfinite(fr[(cb+c)*5+j]):
                    status[w]=6;return
            i+=d
        elif t==0 or t==1 or t==2 or t==3 or t==4 or t==5 or t==6:
            i+=1
        else:
            status[w]=7;return
    for r in range(n):
        if not wp.isfinite(jar[rb+r]) or not wp.isfinite(D[rb+r]) or not wp.isfinite(R[rb+r]) or not wp.isfinite(loss[rb+r]):
            status[w]=8;return
        for j in range(nv):
            if not wp.isfinite(J[(rb+r)*nv+j]):status[w]=9;return

    # Sufficient FP64 safety envelope for this fixed dense stage. All input
    # arithmetic stays below 1e105 before accumulation, even for int32 row
    # capacity, when the middle-cone denominator T is at least 1e-12.
    # D enters force/cost/H linearly; only its positive support bound expands.
    # Other magnitudes, cone denominators, source grouping and precision remain
    # unchanged. The closed D upper endpoint is an exact binary64 integer.
    cap=scalar.wp_real(10000.0);d_cap=scalar.wp_real(2.0e16)
    small=scalar.wp_real(0.0001);lower=_flip_sign(cap)
    for r in range(n):
        if (jar[rb+r]>cap or jar[rb+r]<lower or D[rb+r]>d_cap or D[rb+r]<lower or
            R[rb+r]>cap or R[rb+r]<lower or loss[rb+r]>cap or loss[rb+r]<lower):
            status[w]=10;return
        for j in range(nv):
            v=J[(rb+r)*nv+j]
            if v>cap or v<lower:status[w]=10;return
    i=eq+fcount
    while i<n:
        if typ[rb+i]==7:
            c=cid[rb+i];d=dim[cb+c];m=mu[cb+c];fb=(cb+c)*5
            if m<small or m>cap:status[w]=10;return
            for j in range(d-1):
                f=fr[fb+j]
                if f<small or f>cap:status[w]=10;return
            T=_cone_t(jar,fr,rb+i,fb,d);N=jar[rb+i]*m
            top=N>=m*T or (T<=scalar.wp_real(0.0) and N>=scalar.wp_real(0.0))
            bottom=m*N+T<=scalar.wp_real(0.0) or (T<=scalar.wp_real(0.0) and N<scalar.wp_real(0.0))
            if not top and not bottom and T<scalar.wp_real(1.0e-12):
                status[w]=10;return
            i+=d
        else:i+=1

    s=scalar.wp_real(0.0)
    if n==0:
        cost[w]=s
        return
    for r in range(n):force[rb+r]=_flip_sign(D[rb+r])*jar[rb+r]
    for r in range(eq):
        s=s+scalar.wp_real(0.5)*D[rb+r]*jar[rb+r]*jar[rb+r]
        state[rb+r]=1
    for r in range(eq,eq+fcount):
        v=jar[rb+r];vR=R[rb+r];fl=loss[rb+r]
        if v<=_flip_sign(vR)*fl:
            s=s+(scalar.wp_real(-0.5)*vR*fl*fl-fl*v)
            force[rb+r]=fl;state[rb+r]=2
        elif v>=vR*fl:
            s=s+(scalar.wp_real(-0.5)*vR*fl*fl+fl*v)
            force[rb+r]=_flip_sign(fl);state[rb+r]=3
        else:
            s=s+scalar.wp_real(0.5)*D[rb+r]*v*v
            state[rb+r]=1
    i=eq+fcount
    while i<n:
        if typ[rb+i]!=7:
            if jar[rb+i]>=scalar.wp_real(0.0):
                force[rb+i]=scalar.wp_real(0.0);state[rb+i]=0
            else:
                s=s+scalar.wp_real(0.5)*D[rb+i]*jar[rb+i]*jar[rb+i]
                state[rb+i]=1
            i+=1
        else:
            c=cid[rb+i];d=dim[cb+c];m=mu[cb+c];fb=(cb+c)*5
            N=jar[rb+i]*m
            T=_cone_t(jar,fr,rb+i,fb,d)
            if N>=m*T or (T<=scalar.wp_real(0.0) and N>=scalar.wp_real(0.0)):
                for j in range(d):force[rb+i+j]=scalar.wp_real(0.0)
                state[rb+i]=0
            elif m*N+T<=scalar.wp_real(0.0) or (T<=scalar.wp_real(0.0) and N<scalar.wp_real(0.0)):
                for j in range(d):
                    s=s+scalar.wp_real(0.5)*D[rb+i+j]*jar[rb+i+j]*jar[rb+i+j]
                state[rb+i]=1
            else:
                Dm=D[rb+i]/(m*m*(scalar.wp_real(1.0)+m*m))
                NmT=N-m*T
                s=s+scalar.wp_real(0.5)*Dm*NmT*NmT
                force[rb+i]=_flip_sign(Dm)*NmT*m
                for j in range(1,d):
                    Uj=_u(jar,fr,rb+i,fb,j)
                    force[rb+i+j]=_flip_sign(force[rb+i])/T*Uj*fr[fb+j-1]
                state[rb+i]=4
                if flag[w]!=0:
                    hb=(cb+c)*36
                    scl=_flip_sign(m)/T
                    H[hb]=scalar.wp_real(1.0)
                    for j in range(1,d):H[hb+j]=scl*_u(jar,fr,rb+i,fb,j)
                    scl=m*N/(T*T*T)
                    for k in range(1,d):
                        Uk=_u(jar,fr,rb+i,fb,k)
                        for j in range(k,d):
                            H[hb+k*d+j]=scl*_u(jar,fr,rb+i,fb,j)*Uk
                    scl=m*m-m*N/T
                    for j in range(1,d):H[hb+j*(d+1)]=H[hb+j*(d+1)]+scl
                    for k in range(d):
                        scale=Dm*(m if k==0 else fr[fb+k-1])
                        for j in range(k,d):
                            H[hb+k*d+j]=H[hb+k*d+j]*(scale*(m if j==0 else fr[fb+j-1]))
                    for k in range(d):
                        for j in range(k+1,d):H[hb+j*d+k]=H[hb+k*d+j]
            for j in range(1,d):state[rb+i+j]=state[rb+i]
            i+=d
    cost[w]=s

@wp.kernel(enable_backward=False)
def _transpose(active:wp.array(dtype=wp.int32),count:wp.array(dtype=wp.int32),
               J:wp.array(dtype=scalar.wp_real),force:wp.array(dtype=scalar.wp_real),
               qfrc:wp.array(dtype=scalar.wp_real),status:wp.array(dtype=wp.int32),
               rows:int,nv:int):
    tid=wp.tid();w=tid//nv;j=tid-w*nv
    if active[w]==0 or status[w]!=0:return
    acc=scalar.wp_real(0.0);n=count[w]
    for r in range(n):
        f=force[w*rows+r]
        if f!=scalar.wp_real(0.0):acc=acc+J[(w*rows+r)*nv+j]*f
    qfrc[tid]=acc

class ConstraintUpdateWorkspace:
    """Preallocated world-outer stage. upload is prepare; launch is resident."""
    def __init__(self,nworld:int,rows:int,contacts:int,nv:int=43,device='cuda:0'):
        if nworld<1 or rows<1 or contacts<1 or nv!=43:raise ValueError('fixed dense nv=43 and positive capacities required')
        limit=np.iinfo(np.int32).max
        if nworld*rows*nv>limit or nworld*contacts*36>limit:
            raise ValueError('flattened device index exceeds int32 capacity')
        if not str(device).startswith('cuda'):raise ValueError('S04 production requires CUDA device')
        self.nworld=nworld;self.rows=rows;self.contacts=contacts;self.nv=nv;self.device=device
        specs={'active':(nworld,wp.int32),'count':(nworld,wp.int32),'ne':(nworld,wp.int32),
          'nf':(nworld,wp.int32),'ncon':(nworld,wp.int32),'flag':(nworld,wp.int32),
          'jar':(nworld*rows,scalar.wp_real),'D':(nworld*rows,scalar.wp_real),
          'R':(nworld*rows,scalar.wp_real),'loss':(nworld*rows,scalar.wp_real),
          'type':(nworld*rows,wp.int32),'id':(nworld*rows,wp.int32),
          'J':(nworld*rows*nv,scalar.wp_real),'mu':(nworld*contacts,scalar.wp_real),
          'friction':(nworld*contacts*5,scalar.wp_real),'dim':(nworld*contacts,wp.int32),
          'force':(nworld*rows,scalar.wp_real),'state':(nworld*rows,wp.int32),
          'H':(nworld*contacts*36,scalar.wp_real),'qfrc':(nworld*nv,scalar.wp_real),
          'cost':(nworld,scalar.wp_real),'status':(nworld,wp.int32)}
        self.shapes={key:(nworld,) for key in ('active','count','ne','nf','ncon','flag','cost','status')}
        self.shapes.update({key:(nworld,rows) for key in ('jar','D','R','loss','type','id','force','state')})
        self.shapes.update(J=(nworld,rows,nv),mu=(nworld,contacts),friction=(nworld,contacts,5),
                           dim=(nworld,contacts),H=(nworld,contacts,36),qfrc=(nworld,nv))
        self.buffers={k:wp.zeros(shape,dtype=dtype,device=device) for k,(shape,dtype) in specs.items()}
        for k,v in self.buffers.items():setattr(self,k,v)

    def upload(self,**payload):
        required=set(self.buffers)-{'status'}
        if set(payload)!=required:raise ValueError(('payload keys',sorted(required-set(payload)),sorted(set(payload)-required)))
        validated={}
        for key in sorted(required):
            if not isinstance(payload[key],np.ndarray):raise ValueError(('schema:host_array',key))
            target=self.buffers[key];array=payload[key];dtype=np.int32 if target.dtype==wp.int32 else np.float64
            if array.dtype!=dtype or array.shape!=self.shapes[key]:
                raise ValueError(('schema',key,array.shape,str(array.dtype),self.shapes[key],str(dtype)))
            validated[key]=np.ascontiguousarray(array).reshape(-1)
        for key in sorted(required):self.buffers[key].assign(validated[key])

    def launch(self,stream=None):
        wp.launch(_update,dim=self.nworld,inputs=[self.active,self.count,self.ne,self.nf,self.ncon,self.flag,
            self.jar,self.D,self.R,self.loss,self.type,self.id,self.J,self.mu,self.friction,self.dim,
            self.force,self.state,self.H,self.cost,self.status,self.rows,self.contacts,self.nv],device=self.device,stream=stream)
        wp.launch(_transpose,dim=self.nworld*self.nv,inputs=[self.active,self.count,self.J,self.force,
            self.qfrc,self.status,self.rows,self.nv],device=self.device,stream=stream)

    def download(self):
        return {k:self.buffers[k].numpy().reshape(self.shapes[k]).copy()
                for k in ('force','state','H','qfrc','cost','status')}
