"""Explicit N1 native fixed-buffer reset inventory; no arena tail memset."""
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
from libero.gpu.host.runtime.dynamics.recovery import ResetBinding

# Paths refer to actual retained owners, not metadata-selected replacements.
D_FIELDS={
 "xpos":"position.body.xpos","xquat":"position.body.xquat","xmat":"position.body.xmat",
 "xanchor":"position.body.xanchor","xaxis":"position.body.xaxis",
 "xipos":"position.frames.xipos","ximat":"position.frames.ximat",
 "geom_xpos":"position.frames.geom_xpos","geom_xmat":"position.frames.geom_xmat",
 "site_xpos":"position.frames.site_xpos","site_xmat":"position.frames.site_xmat",
 "subtree_com":"position.subtree_com","cinert":"position.cinert","cdof":"position.cdof",
 "crb":"position.crb","qM":"position.factor.qm","qLD":"position.factor.ld",
 "qLDiagInv":"position.factor.inv","qLDiagSqrtInv":"position.factor.sqrt_inv",
 "cvel":"cvel","cdof_dot":"cdof_dot"}

def _path(owner,path):
    for part in path.split("."):owner=getattr(owner,part)
    return owner

def add_native_inventory(bindings,*,forward,post):
    result=dict(bindings);catalog={}
    def add(name,array,role="derived"):
        if not isinstance(array,wp.array) or not array.is_contiguous:
            raise TypeError("native fixed buffer needs retained contiguous array: "+name)
        mapping={scalar.wp_real:np.dtype(scalar.name),wp.int32:np.int32,wp.int64:np.int64}
        if array.dtype not in mapping:raise TypeError("native fixed buffer dtype: "+name)
        span=(int(array.ptr),int(array.ptr)+array.size*np.dtype(mapping[array.dtype]).itemsize)
        for previous,binding in result.items():
            a=binding.array
            old=(int(a.ptr),int(a.ptr)+a.size*np.asarray(binding.defaults).dtype.itemsize)
            if span==old:
                if array.dtype!=a.dtype:raise ValueError("native aliases differ dtype")
                catalog[name]={"alias_of":previous,"role":role};return
            if max(span[0],old[0])<min(span[1],old[1]):
                raise ValueError("partial native buffer overlap: "+name+"/"+previous)
        if array.size:
            result[name]=ResetBinding(array,np.zeros(array.size,dtype=mapping[array.dtype]),role)
            catalog[name]={"ptr":int(array.ptr),"shape":array.shape,"dtype":str(array.dtype),"role":role}
    d=forward.d
    for name,path in D_FIELDS.items():add(name,_path(d,path))
    for name in ("qfrc_bias","qfrc_smooth","qacc_smooth"):
        add(name,d.public[name])
    add("qfrc_constraint",forward.n.public_arrays["qfrc_constraint"])
    for name,key in (("cfrc_ext","cfrc_ext"),("cfrc_int","cfrc_int"),("sensordata","sensor")):
        add(name,post.public[key])
    # These are native fixed diagnostic buffers, not efc force/state arena.
    for name in ("solver_iter","solver_nnz","solver_stats_f","solver_stats_i"):
        add(name,forward.n.public_arrays[name],"diagnostic")
    # Actual count holders are reset; efc/contact capacity tail bytes remain held.
    for name,array in (("ncon",forward.transport.ncon),("nefc",forward.n.ws.nefc),
            ("ne",forward.n.ws.ne),("nf",forward.n.ws.nf)):
        add(name,array,"diagnostic")
    return result,{"schema":"native-reset-fixed-buffer-inventory-v1","N":1,"fields":catalog,
        "not_backend_owned":("native timers","native energy","native maxuse headers","solver_fwdinv"),
        "pending_exact_owner_mapping":("native cacc canonical owner","native sparse D/B layout defaults"),
        "excluded_protocol":("first_error","state/forward/model epochs","controller cache/goal/q0/action"),
        "excluded_arena_zero":("contact capacity tails","efc force/state capacity tails")}
