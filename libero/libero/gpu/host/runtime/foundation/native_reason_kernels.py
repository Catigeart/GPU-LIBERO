"""S07 entry metadata guard, bound to frozen search_kernel 146-177.

No numeric input outside original guard read domains is checked. In particular,
NaN/+inf mu does not trigger the original <=0 guard; local5 remains the actual
producer post-status and is never manufactured from an input precheck.
"""
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})

@wp.kernel(enable_backward=False)
def _reason_linesearch(active:wp.array(dtype=wp.int32),nefc:wp.array(dtype=wp.int32),
        ne:wp.array(dtype=wp.int32),nf:wp.array(dtype=wp.int32),typ:wp.array(dtype=wp.int32),
        dim:wp.array(dtype=wp.int32),mu:wp.array(dtype=scalar.wp_real),rows:int,tracecap:int,
        reason_code:wp.array(dtype=wp.int32),reason_kind:wp.array(dtype=wp.int32)):
    w=wp.tid()
    reason_code[w]=0;reason_kind[w]=0
    if active[w]==0:return
    if active[w]!=1:reason_code[w]=1;reason_kind[w]=1;return
    n=nefc[w];e=ne[w];f=nf[w]
    if n<0:reason_code[w]=2;reason_kind[w]=2;return
    if n>rows:reason_code[w]=2;reason_kind[w]=3;return
    if e<0:reason_code[w]=2;reason_kind[w]=2;return
    if f<0:reason_code[w]=2;reason_kind[w]=2;return
    if e>n:reason_code[w]=2;reason_kind[w]=2;return
    if f>n-e:reason_code[w]=2;reason_kind[w]=2;return
    if tracecap<52:reason_code[w]=3;reason_kind[w]=3;return
    i=int(0)
    while i<n:
        if typ[w*rows+i]==7:
            d=dim[w*rows+i]
            if i<e+f:reason_code[w]=4;reason_kind[w]=2;return
            if d!=3 and d!=4 and d!=6:reason_code[w]=4;reason_kind[w]=4;return
            if i+d>n:reason_code[w]=4;reason_kind[w]=2;return
            if mu[w*rows+i]<=scalar.wp_real(0.0):reason_code[w]=4;reason_kind[w]=6;return
            i+=d-1
        i+=1
