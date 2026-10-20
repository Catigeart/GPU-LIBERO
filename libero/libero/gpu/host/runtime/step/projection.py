"""Typed device-only layout projection from true S public owners to N.
No numeric oracle or host counts classify the consumer. Unread metadata tails
are held. Original N validators retain invalid-source error precedence.
9001..9007 are bridge diagnostic codes, never N err_code/first_error codes.
Do not block original N validation using these diagnostics. Publication success
must additionally require projection_status==0; original N errors take priority.
"""
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.kernel
def project_counts(admitted:wp.array(dtype=wp.int32),
                   source_counts:wp.array(dtype=wp.int32),
                   source_ncon:wp.array(dtype=wp.int32),
                   nefc:wp.array(dtype=wp.int32),ne:wp.array(dtype=wp.int32),
                   nf:wp.array(dtype=wp.int32),ncon:wp.array(dtype=wp.int32),
                   count:wp.array(dtype=wp.int32),empty:wp.array(dtype=wp.int32),
                   newton:wp.array(dtype=wp.int32),active:wp.array(dtype=wp.int32),
                   projection_status:wp.array(dtype=wp.int32),
                   error_item:wp.array(dtype=wp.int32),rows:int,contacts:int):
 w=wp.tid()
 nefc[w]=0;ne[w]=0;nf[w]=0;ncon[w]=0;count[w]=0
 empty[w]=0;newton[w]=0;active[w]=0
 projection_status[w]=0;error_item[w]=-1
 if admitted[w]!=1:return
 # Original S public counts are [ne,nf,nlimit,nefc,nnzJ] per world.
 ne[w]=source_counts[w*5];nf[w]=source_counts[w*5+1]
 nefc[w]=source_counts[w*5+3];count[w]=nefc[w]
 ncon[w]=source_ncon[w]
 if nefc[w]==0:empty[w]=1
 elif nefc[w]>0:newton[w]=1;active[w]=1
 # Invalid counts remain actual copies so original N4002 is not hidden.
 n=nefc[w];eq=ne[w];fr=nf[w];nl=source_counts[w*5+2];nc=ncon[w]
 if n<0 or n>rows:projection_status[w]=9001;error_item[w]=3;return
 if eq<0 or eq>n:projection_status[w]=9001;error_item[w]=0;return
 if fr<0 or fr>n-eq:projection_status[w]=9001;error_item[w]=1;return
 if nl<0 or nl>n-eq-fr:projection_status[w]=9001;error_item[w]=2;return
 if source_counts[w*5+4]!=n*43:projection_status[w]=9001;error_item[w]=4;return
 if nc<0 or nc>contacts:projection_status[w]=9001;error_item[w]=5;return

@wp.kernel
def project_cone_heads(admitted:wp.array(dtype=wp.int32),
                       nefc:wp.array(dtype=wp.int32),ne:wp.array(dtype=wp.int32),
                       nf:wp.array(dtype=wp.int32),ncon:wp.array(dtype=wp.int32),
                       row_type:wp.array(dtype=wp.int32,ndim=2),
                       row_id:wp.array(dtype=wp.int32,ndim=2),
                       contact_dim:wp.array(dtype=wp.int32,ndim=2),
                       contact_mu:wp.array(dtype=scalar.wp_real,ndim=2),
                       contact_friction:wp.array(dtype=scalar.wp_real,ndim=3),
                       cone_dim:wp.array(dtype=wp.int32,ndim=2),
                       cone_mu:wp.array(dtype=scalar.wp_real,ndim=2),
                       cone_friction:wp.array(dtype=scalar.wp_real,ndim=3),
                       projection_status:wp.array(dtype=wp.int32),
                       error_item:wp.array(dtype=wp.int32),
                       rows:int,contacts:int):
 w=wp.tid()
 if admitted[w]!=1:return
 if projection_status[w]!=0:return
 n=nefc[w];i=ne[w]+nf[w]
 if n<0 or n>rows or ne[w]<0 or nf[w]<0 or i<0 or i>n:
  projection_status[w]=9001;error_item[w]=3;return
 if ncon[w]<0 or ncon[w]>contacts:
  projection_status[w]=9001;error_item[w]=5;return
 # Validate the complete source row structure before writing any cone cell.
 # This protects all metadata history on a malformed later block. Original
 # N4002/N8001/S04 validators still run with their unchanged precedence.
 while i<n:
  if row_type[w,i]==7:
   contact=row_id[w,i]
   if contact<0 or contact>=ncon[w]:projection_status[w]=9002;error_item[w]=i;return
   dim=contact_dim[w,contact]
   if dim!=1 and dim!=3 and dim!=4 and dim!=6:projection_status[w]=9003;error_item[w]=i;return
   if dim>n-i:projection_status[w]=9004;error_item[w]=i;return
   # Only the true elliptic block head is consumed by original S07.
   # Validate the grouping before projecting; original N/S04 retain errors.
   for component in range(dim):
    if row_type[w,i+component]!=7 or row_id[w,i+component]!=contact:
     projection_status[w]=9004;error_item[w]=i+component;return
   if not wp.isfinite(contact_mu[w,contact]) or contact_mu[w,contact]<=scalar.wp_real(0.0):
    projection_status[w]=9006;error_item[w]=i;return
   for component in range(dim-1):
    value=contact_friction[w,contact,component]
    if not wp.isfinite(value) or value<=scalar.wp_real(0.0):
     projection_status[w]=9007;error_item[w]=i;return
   i+=dim
  elif row_type[w,i]>=0 and row_type[w,i]<=6:
   # A frictionless contact is a scalar source row and has no cone write.
   if row_type[w,i]==5:
    contact=row_id[w,i]
    if contact<0 or contact>=ncon[w]:projection_status[w]=9002;error_item[w]=i;return
    if contact_dim[w,contact]!=1:projection_status[w]=9003;error_item[w]=i;return
   i+=1
  else:projection_status[w]=9005;error_item[w]=i;return
 i=ne[w]+nf[w]
 while i<n:
  if row_type[w,i]==7:
   contact=row_id[w,i];dim=contact_dim[w,contact]
   cone_dim[w,i]=dim;cone_mu[w,i]=contact_mu[w,contact]
   for component in range(dim-1):
    cone_friction[w,i,component]=contact_friction[w,contact,component]
   i+=dim
  else:i+=1

@wp.kernel
def reset_scalar_i32(value:wp.array(dtype=wp.int32),fill:int):
 value[wp.tid()]=fill

@wp.kernel
def reset_scalar_f64(value:wp.array(dtype=scalar.wp_real)):
 value[wp.tid()]=scalar.wp_real(0.0)

@wp.kernel
def prepare_static_run_scalars(model_scalars:wp.array(dtype=scalar.wp_real),
                               tolerance:wp.array(dtype=scalar.wp_real),
                               meaninertia:wp.array(dtype=scalar.wp_real),
                               stage:wp.array(dtype=wp.int32),
                               maxiter_arr:wp.array(dtype=wp.int32),
                               solver_iter_base:wp.array(dtype=wp.int32),
                               maxiter:int):
 w=wp.tid()
 tolerance[w]=model_scalars[0];meaninertia[w]=model_scalars[1]
 stage[w]=1;maxiter_arr[w]=maxiter;solver_iter_base[w]=0
