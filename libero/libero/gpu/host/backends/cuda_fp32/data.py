# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Owned rigid state/contact/constraint buffers and reusable Newton scratch.

Layouts and initialization follow pinned MJWarp io.make_data / solver context.
No upstream package is imported. Per-solve initialization is part of the
execution plan, rather than an accidental side effect of allocating scratch.
"""
from types import SimpleNamespace
import numpy as np
import mujoco
from . import device as cuda, layout_specs as specs, constants as types
from .model import storage_type


def allocate(layout,sizes):
    dimensions,name=layout
    shape=tuple(sizes.get(d,0) if isinstance(d,str) and not d.isdigit() else int(d) for d in dimensions)
    return cuda.zeros(shape,dtype=storage_type(name))


def make_data(host,model,worlds,contacts,rows):
    if min(worlds,contacts,rows)<1:raise ValueError('Positive world/contact/row capacities required')
    sizes=dict(model._sizes,nworld=worlds,naconmax=worlds*contacts,njmax=rows,njmax_pad=((rows+15)//16)*16,
        njmax_nnz=rows*host.nv,nvmax_pad=((max(host.nv,1)+16)//16)*16,qld_total=model.qLD_block_total)
    sizes['nvmax_pad_sq']=sizes['nvmax_pad']**2
    compact=('ctol','cls_tol','cdof_tri_row','cdof_tri_col','cM','cqLD','crhs','cx','cJ',
        'cMa','cqfrc_smooth','cqacc_smooth','cqacc_warmstart','cqacc','cqfrc_constraint')
    # Compaction is disabled for this admitted dense rigid path. io.make_data
    # publishes zero extents for these optional workspaces, retaining their rank.
    values={k:cuda.empty((0,0,1) if k in ('crhs','cx') else (0,)*len(v[0]),dtype=storage_type(v[1])) if k in compact else allocate(v,sizes)
            for k,v in specs.DATA_ARRAYS.items()}
    d=SimpleNamespace(**values,nworld=worlds,naconmax=sizes['naconmax'],naccdmax=sizes['naconmax'],
        njmax=rows,nvmax=host.nv,nvmax_pad=sizes['nvmax_pad'],njmax_pad=sizes['njmax_pad'],njmax_nnz=sizes['njmax_nnz'])
    d._cuda_native=True
    d.contact=SimpleNamespace(**{k:allocate(v,sizes) for k,v in specs.CONTACT_ARRAYS.items()})
    d.contact.efc_address.fill_(-1)
    for name in ('flex','elem','vert'):setattr(d.contact,name,cuda.empty(0,dtype=cuda.vec2i))
    efc=dict(specs.CONSTRAINT_ARRAYS)
    efc.update(J=(('nworld','njmax_pad','nv_pad'),'float'),J_rownnz=(('nworld',0),'int'),
        J_rowadr=(('nworld',0),'int'),J_colind=(('nworld',0,0),'int'))
    d.efc=SimpleNamespace(**{k:allocate(v,sizes) for k,v in efc.items()})
    d.qpos.assign(np.tile(host.qpos0,(worlds,1)))
    initial=mujoco.MjData(host);mujoco.mj_kinematics(host,initial)
    # Static geoms are not overwritten in the normal GPU kinematic leaf.
    for name in ('xquat','xmat','ximat','geom_xpos','geom_xmat'):
        field=getattr(d,name);field.assign(np.tile(getattr(initial,name),(worlds,1)))
    mocap=np.flatnonzero(host.body_mocapid>=0);mocap=mocap[np.argsort(host.body_mocapid[mocap])]
    d.mocap_pos.assign(np.tile(host.body_pos[mocap],(worlds,1)))
    d.mocap_quat.assign(np.tile(host.body_quat[mocap],(worlds,1)))
    d.eq_active.assign(np.tile(host.eq_active0.astype(bool),(worlds,1)))
    d.tree_asleep.fill_(-(1+int(mujoco.mjMINAWAKE)));d.tree_awake.fill_(1)
    for name in ('tree_island','dof_island','dof_islandid','efc_islandid','dof_cdof','cdof_dof'):
        getattr(d,name).fill_(-1)
    awake=np.zeros(host.nbody,np.int32)
    for body in range(host.nbody):
        tree=int(host.body_treeid[body]);root=int(host.body_rootid[body])
        awake[body]=int(types.SleepState.AWAKE if tree>=0 or host.body_mocapid[root]>=0 else types.SleepState.STATIC)
    d.body_awake.assign(np.tile(awake,(worlds,1)))
    return d


def newton_workspace(model,data):
    n,nv,pad,rows=data.nworld,model.nv,model.nv_pad,data.njmax
    widths=dict(Jaref=(n,rows),search_dot=(n,),done=(n,),grad=(n,pad),grad_dot=(n,),
        newton_decrement=(n,),search=(n,nv),mv=(n,nv),jv=(n,rows),quad=(n,rows),
        alpha=(n,),grad_scale=(n,),improvement=(n,),ls_exhausted=(n,),search_unchanged=(n,),
        h=(n,pad,pad),hfactor=(n,pad,pad) if nv>32 else (n,0,0))
    values={k:cuda.zeros(shape,dtype=cuda.boolean if k in ('done','ls_exhausted','search_unchanged') else cuda.vec3 if k=='quad' else cuda.float32) for k,shape in widths.items()}
    return SimpleNamespace(**values)
