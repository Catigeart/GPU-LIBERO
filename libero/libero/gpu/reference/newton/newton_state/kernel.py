"""Four independent GPU stage entries for MuJoCo 2.3.7 Newton state writes.

Callers own and prefill all device buffers. The production launch methods do
no allocation, host arithmetic, upload, download or inter-stage transfer.
"""
import libero.gpu.precision as scalar
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True,
                       'fast_math': False})


@wp.kernel(enable_backward=False)
def _warm_select(active: wp.array(dtype=wp.int32),
                 nefc: wp.array(dtype=wp.int32),
                 warm_cost: wp.array(dtype=scalar.wp_real),
                 smooth_cost: wp.array(dtype=scalar.wp_real),
                 warm_qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                 smooth_qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                 smooth_force: wp.array(dtype=scalar.wp_real, ndim=2),
                 smooth_state: wp.array(dtype=wp.int32, ndim=2),
                 smooth_qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
                 qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                 force: wp.array(dtype=scalar.wp_real, ndim=2),
                 state: wp.array(dtype=wp.int32, ndim=2),
                 qfrc: wp.array(dtype=scalar.wp_real, ndim=2),
                 choice: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32),
                 nv: int, capacity: int):
    w=wp.tid()
    if status[w]!=0:
        return
    if active[w]==0:
        return
    if active[w]!=1:
        status[w]=1
        return
    n=nefc[w]
    if n<=0 or n>capacity:
        status[w]=2
        return
    if not wp.isfinite(warm_cost[w]) or not wp.isfinite(smooth_cost[w]):
        status[w]=3
        return
    for i in range(nv):
        if not wp.isfinite(warm_qacc[w,i]) or not wp.isfinite(smooth_qacc[w,i]) or not wp.isfinite(smooth_qfrc[w,i]):
            status[w]=3
            return
    for j in range(n):
        if not wp.isfinite(smooth_force[w,j]):
            status[w]=3
            return
    choose_smooth=wp.int32(0)
    if warm_cost[w]>smooth_cost[w]:
        choose_smooth=1
    choice[w]=choose_smooth
    for i in range(nv):
        if choose_smooth==1:
            qacc[w,i]=smooth_qacc[w,i]
        else:
            qacc[w,i]=warm_qacc[w,i]
        qfrc[w,i]=smooth_qfrc[w,i]
    for j in range(n):
        force[w,j]=smooth_force[w,j]
        state[w,j]=smooth_state[w,j]


@wp.kernel(enable_backward=False)
def _commit_step(active: wp.array(dtype=wp.int32),
                 nefc: wp.array(dtype=wp.int32),
                 alpha: wp.array(dtype=scalar.wp_real),
                 search: wp.array(dtype=scalar.wp_real, ndim=2),
                 mv: wp.array(dtype=scalar.wp_real, ndim=2),
                 jv: wp.array(dtype=scalar.wp_real, ndim=2),
                 qacc: wp.array(dtype=scalar.wp_real, ndim=2),
                 ma: wp.array(dtype=scalar.wp_real, ndim=2),
                 jaref: wp.array(dtype=scalar.wp_real, ndim=2),
                 committed: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32),
                 nv: int, capacity: int):
    w=wp.tid()
    if status[w]!=0:
        return
    if active[w]==0:
        return
    if active[w]!=1:
        status[w]=1
        return
    n=nefc[w]
    if n<=0 or n>capacity:
        status[w]=2
        return
    a=alpha[w]
    if not wp.isfinite(a):
        status[w]=3
        return
    if a==scalar.wp_real(0.0):
        committed[w]=0
        return
    for i in range(nv):
        if not wp.isfinite(search[w,i]) or not wp.isfinite(mv[w,i]) or not wp.isfinite(qacc[w,i]) or not wp.isfinite(ma[w,i]):
            status[w]=3
            return
        if not wp.isfinite(qacc[w,i]+a*search[w,i]) or not wp.isfinite(ma[w,i]+a*mv[w,i]):
            status[w]=3
            return
    for j in range(n):
        if not wp.isfinite(jv[w,j]) or not wp.isfinite(jaref[w,j]):
            status[w]=3
            return
        if not wp.isfinite(jaref[w,j]+a*jv[w,j]):
            status[w]=3
            return
    committed[w]=1
    for i in range(nv):
        qacc[w,i]=qacc[w,i]+a*search[w,i]
    for i in range(nv):
        ma[w,i]=ma[w,i]+a*mv[w,i]
    for j in range(n):
        jaref[w,j]=jaref[w,j]+a*jv[w,j]


@wp.func
def _source_dot_self(grad: wp.array(dtype=scalar.wp_real, ndim=2), w: int, nv: int) -> scalar.wp_real:
    # engine_util_blas.c mju_dot: four independent lanes, then source reduction.
    s0=scalar.wp_real(0.0);s1=scalar.wp_real(0.0)
    s2=scalar.wp_real(0.0);s3=scalar.wp_real(0.0)
    i=wp.int32(0)
    while i<=nv-4:
        x=grad[w,i];s0=s0+x*x
        x=grad[w,i+1];s1=s1+x*x
        x=grad[w,i+2];s2=s2+x*x
        x=grad[w,i+3];s3=s3+x*x
        i+=4
    res=(s0+s2)+(s1+s3)
    if nv-i==3:
        x=grad[w,i];y=grad[w,i+1];z=grad[w,i+2]
        res=res+(x*x+y*y+z*z)
    elif nv-i==2:
        x=grad[w,i];y=grad[w,i+1]
        res=res+(x*x+y*y)
    elif nv-i==1:
        x=grad[w,i]
        res=res+x*x
    return res


@wp.kernel(enable_backward=False)
def _stats_stop(active: wp.array(dtype=wp.int32),
                nefc: wp.array(dtype=wp.int32),
                stage: wp.array(dtype=wp.int32),
                committed: wp.array(dtype=wp.int32),
                oldstate: wp.array(dtype=wp.int32, ndim=2),
                newstate: wp.array(dtype=wp.int32, ndim=2),
                grad: wp.array(dtype=scalar.wp_real, ndim=2),
                oldcost: wp.array(dtype=scalar.wp_real),
                newcost: wp.array(dtype=scalar.wp_real),
                meaninertia: wp.array(dtype=scalar.wp_real),
                tolerance: wp.array(dtype=scalar.wp_real),
                lineslope: wp.array(dtype=scalar.wp_real),
                nactive: wp.array(dtype=wp.int32),
                nupdate: wp.array(dtype=wp.int32),
                neval: wp.array(dtype=wp.int32),
                iteration: wp.array(dtype=wp.int32),
                solver_iter_base: wp.array(dtype=wp.int32),
                maxiter: wp.array(dtype=wp.int32),
                stats_f: wp.array(dtype=scalar.wp_real, ndim=3),
                stats_i: wp.array(dtype=wp.int32, ndim=3),
                out_nchange: wp.array(dtype=wp.int32),
                out_improvement: wp.array(dtype=scalar.wp_real),
                out_gradient: wp.array(dtype=scalar.wp_real),
                out_stop: wp.array(dtype=wp.int32),
                out_reason: wp.array(dtype=wp.int32),
                out_iter: wp.array(dtype=wp.int32),
                status: wp.array(dtype=wp.int32),
                nv: int, capacity: int):
    w=wp.tid()
    if status[w]!=0:
        return
    if active[w]==0:
        return
    if active[w]!=1:
        status[w]=1
        return
    n=nefc[w];it=iteration[w];base=solver_iter_base[w]
    if n<=0 or n>capacity or it<0 or base<0 or base>2147483647-it or it>=maxiter[w] or maxiter[w]<0 or maxiter[w]>1000000:
        status[w]=2
        return
    if stage[w]!=1 or committed[w]!=1:
        status[w]=4
        return
    inertia=meaninertia[w];tol=tolerance[w]
    if not wp.isfinite(oldcost[w]) or not wp.isfinite(newcost[w]) or not wp.isfinite(inertia) or not wp.isfinite(tol) or not wp.isfinite(lineslope[w]) or inertia<=scalar.wp_real(0.0) or tol<scalar.wp_real(0.0):
        status[w]=3
        return
    if nactive[w]<0 or nupdate[w]<0 or neval[w]<0:
        status[w]=2
        return
    nchange=wp.int32(0)
    for j in range(n):
        if oldstate[w,j]!=newstate[w,j]:
            nchange+=1
    for i in range(nv):
        if not wp.isfinite(grad[w,i]):
            status[w]=3
            return
    dot=_source_dot_self(grad,w,nv)
    if not wp.isfinite(dot):
        status[w]=3
        return
    scale=inertia*scalar.wp_real(nv)
    improvement=(oldcost[w]-newcost[w])/scale
    gradient=wp.sqrt(dot)/scale
    if not wp.isfinite(improvement) or not wp.isfinite(gradient):
        status[w]=3
        return
    slot=base+it
    if slot<0:
        status[w]=2
        return
    if slot<1000:
        stats_f[w,slot,0]=improvement
        stats_f[w,slot,1]=gradient
        stats_f[w,slot,2]=lineslope[w]
        stats_i[w,slot,0]=nactive[w]
        stats_i[w,slot,1]=nchange
        stats_i[w,slot,2]=neval[w]
        stats_i[w,slot,3]=nupdate[w]
    nextit=it+1
    stop=wp.int32(0);reason=wp.int32(0)
    if improvement<tol or gradient<tol:
        stop=1;reason=1  # tolerance
    elif nextit>=maxiter[w]:
        stop=1;reason=2  # maxiter guard on following loop
    out_nchange[w]=nchange
    out_improvement[w]=improvement
    out_gradient[w]=gradient
    out_stop[w]=stop
    out_reason[w]=reason
    out_iter[w]=nextit


@wp.kernel(enable_backward=False)
def _finalize(active: wp.array(dtype=wp.int32),
              solver_exit: wp.array(dtype=wp.int32),
              iteration: wp.array(dtype=wp.int32),
              qacc: wp.array(dtype=scalar.wp_real, ndim=2),
              warmstart: wp.array(dtype=scalar.wp_real, ndim=2),
              solver_iter: wp.array(dtype=wp.int32),
              solver_nnz: wp.array(dtype=wp.int32),
              status: wp.array(dtype=wp.int32),
              nv: int):
    w=wp.tid()
    if status[w]!=0:
        return
    if active[w]==0:
        return
    if active[w]!=1:
        status[w]=1
        return
    if solver_exit[w]!=0 and solver_exit[w]!=1:
        status[w]=4
        return
    if solver_exit[w]==0:
        return
    it=iteration[w]
    if it<0 or solver_iter[w]<0 or solver_iter[w]>2147483647-it:
        status[w]=2
        return
    for i in range(nv):
        if not wp.isfinite(qacc[w,i]):
            status[w]=3
            return
    solver_iter[w]=solver_iter[w]+it
    solver_nnz[w]=nv*nv
    for i in range(nv):
        warmstart[w,i]=qacc[w,i]


class NewtonStateWorkspace:
    """Prepared buffer bundle. `bind_*` changes device aliases; launch is hot path."""
    def __init__(self, nworld: int, nv: int, capacity: int, device: str='cuda:0'):
        if nworld<1 or nv<1 or capacity<1 or nv*nv>2147483647:
            raise ValueError('invalid static shape')
        self.nworld=nworld;self.nv=nv;self.capacity=capacity;self.device=device
        self.a=self.b=self.c=self.d=None

    def bind_A(self, **arrays): self.a=arrays
    def bind_B(self, **arrays): self.b=arrays
    def bind_C(self, **arrays): self.c=arrays
    def bind_D(self, **arrays): self.d=arrays

    def launch_A(self):
        a=self.a
        wp.launch(_warm_select,self.nworld,
                  inputs=[a[k] for k in ('active','nefc','warm_cost','smooth_cost',
                                         'warm_qacc','smooth_qacc','smooth_force',
                                         'smooth_state','smooth_qfrc','qacc','force',
                                         'state','qfrc','choice','status')]+[self.nv,self.capacity],
                  device=self.device)

    def launch_B(self):
        a=self.b
        wp.launch(_commit_step,self.nworld,
                  inputs=[a[k] for k in ('active','nefc','alpha','search','Mv','Jv',
                                         'qacc','Ma','Jaref','committed','status')]+[self.nv,self.capacity],
                  device=self.device)

    def launch_C(self):
        a=self.c
        wp.launch(_stats_stop,self.nworld,
                  inputs=[a[k] for k in ('active','nefc','stage','committed','oldstate',
                                         'newstate','grad','oldcost','newcost',
                                         'meaninertia','tolerance','lineslope','nactive',
                                         'nupdate','neval','iteration','solver_iter_base',
                                         'maxiter','stats_f','stats_i','out_nchange',
                                         'out_improvement','out_gradient','out_stop',
                                         'out_reason','out_iter','status')]+[self.nv,self.capacity],
                  device=self.device)

    def launch_D(self):
        a=self.d
        wp.launch(_finalize,self.nworld,
                  inputs=[a[k] for k in ('active','solver_exit','iteration','qacc',
                                         'warmstart','solver_iter','solver_nnz','status')]+[self.nv],
                  device=self.device)
