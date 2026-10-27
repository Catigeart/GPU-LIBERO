# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rigid LIBERO model preprocessing adapted from pinned MJWarp io.py.

Preparation uses NumPy and the actual MuJoCo model, without a device compiler.
The resulting host descriptions can be independently tested before uploading.
Unsupported general-engine domains are rejected before allocating device data.
"""
from types import SimpleNamespace
import math
import numpy as np
import mujoco
from . import constants as types, layout_specs as specs, device as cuda


def storage_type(name):
    common={'float':cuda.float32,'int':cuda.int32,'bool':cuda.boolean,
            'vec5':types.vec5,'vec10':types.vec10}
    if name in common:return common[name]
    if hasattr(cuda,name):return getattr(cuda,name)
    if name.startswith('vec'):
        width=128 if name=='vec_pluginattr' else int(name[3:].rstrip('i'))
        return cuda.DType(name,'int32' if name.endswith('i') else 'float32',(width,))
    raise ValueError('Unrepresented storage type: '+name)


def padded_nv(nv):return ((nv+int(nv>32)+15)//16)*16 if nv>32 else ((nv+3)//4)*4


def admit(host):
    if host.opt.solver!=mujoco.mjtSolver.mjSOL_NEWTON or host.opt.cone!=mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError('Independent FP32 requires Newton and elliptic contacts')
    if host.opt.integrator!=mujoco.mjtIntegrator.mjINT_EULER or mujoco.mj_isSparse(host):
        raise ValueError('Independent FP32 requires Euler and dense Jacobian')
    if host.opt.enableflags & types.EnableBit.SLEEP or host.opt.noslip_iterations:
        raise ValueError('Sleep and noslip are outside the LIBERO domain')
    if any(getattr(host,k) for k in ('nflex','ntendon','neq','na','nhistory')):
        raise ValueError('Flex/tendon/equality/activation/history are outside the LIBERO domain')
    if np.any(~np.isin(host.sensor_type,(4,5))):raise ValueError('Unported sensor family')
    if np.any(host.cam_mode) or np.any(host.light_mode):raise ValueError('Unported tracking camera/light')
    if np.any(host.body_plugin!=-1) or np.any(host.geom_plugin!=-1):raise ValueError('Plugins are outside the LIBERO domain')


def mass_layout(host):
    """Same per-tree compact/scalar/dense classification as pinned io.py."""
    address=np.full(host.nv,-1,np.int32);scalar={};gather={};offset=0
    for first,size in zip(host.tree_dofadr,host.tree_dofnum):
        first,size=int(first),int(size)
        if not size:continue
        last=first+size-1;count=int(host.M_rowadr[last]+host.M_rownnz[last]-host.M_rowadr[first])
        compact=count==size;triangle=count==size*(size+1)//2
        if size<=6 and (compact or triangle):
            scalar.setdefault(size,[]).append(first);address[first:first+size]=-2 if compact else offset
            if not compact:offset+=size*size
        elif size<=64:
            gather.setdefault(size,[]).append(first);address[first:first+size]=offset;offset+=size*size
        else:raise ValueError('Unqualified mass block size: '+str(size))
    for starts in scalar.values():starts.sort(key=lambda start:address[start]>=0)
    return address,offset,scalar,gather


def prepare(host):
    """Return a host-only normalized model; no hidden exported-Warp inputs."""
    admit(host)
    m=SimpleNamespace(**{name:int(getattr(host,name,0) or 0) for name in specs.MODEL_SCALARS})
    for name in specs.MODEL_ARRAYS:setattr(m,name,getattr(host,name,None))
    m.nv_pad=padded_nv(host.nv);m.is_sparse=False;m.block_dim=SimpleNamespace(**specs.BLOCK_DIM)
    m.block_dim.update_gradient_grad=max(32,min(256,((host.nv+31)//32)*32))
    m.opt=SimpleNamespace(**{k:getattr(host.opt,k,False if kind=='bool' else 0) for k,kind in specs.OPTION_SCALARS.items()})
    for k in specs.OPTION_ARRAYS:setattr(m.opt,k,getattr(host.opt,k,None))
    m.opt.tolerance=max(float(host.opt.tolerance),1.e-6)
    m.opt.impratio_invsqrt=1./np.sqrt(np.maximum(host.opt.impratio,mujoco.mjMINVAL))
    m.opt.broadphase=types.BroadphaseType.NXN
    m.opt.broadphase_filter=types.BroadphaseFilter.PLANE|types.BroadphaseFilter.SPHERE|types.BroadphaseFilter.OBB
    m.opt.graph_conditional=True;m.opt.run_collision_detection=True;m.opt.warn_overflow=int(types.OverflowType.ALL)
    m.opt.run_rne_postconstraint=False;m.opt.contact_sensor_maxmatch=64
    m.stat=SimpleNamespace(meaninertia=host.stat.meaninertia)
    m.callback=SimpleNamespace(control=None,contactfilter=None)
    m.has_fluid=bool(host.opt.wind.any() or host.opt.density>0 or host.opt.viscosity>0)
    m.flex_interp_assemblable=True
    m.has_efm_actuator=bool(np.any(((host.actuator_gaintype==1)&np.any(host.actuator_gainprm[:,1:3]!=0,axis=1))|
        ((host.actuator_biastype==1)&np.any(host.actuator_biasprm[:,1:3]!=0,axis=1))|
        np.isin(host.actuator_gaintype,(2,3,4))|np.isin(host.actuator_biastype,(2,3,4))))
    m.has_plane_geom=bool(np.any(host.geom_type==0));m.has_ellipsoid_geom=bool(np.any(host.geom_type==4));m.has_sdf_geom=bool(np.any(host.geom_type==8))
    m.nmaxcondim=int(max(np.max(host.geom_condim,initial=0),np.max(host.pair_dim,initial=0)))
    m.nmaxpyramid=max(1,2*(m.nmaxcondim-1));m.nbranch=0
    m.body_subtreenum=np.ones(host.nbody,np.int32);depth=np.zeros(host.nbody,np.int32)
    for b in range(1,host.nbody):depth[b]=depth[host.body_parentid[b]]+1
    for b in reversed(range(1,host.nbody)):m.body_subtreenum[host.body_parentid[b]]+=m.body_subtreenum[b]
    m.body_tree=tuple(np.flatnonzero(depth==d).astype(np.int32) for d in range(int(depth.max())+1))
    children=np.bincount(host.body_parentid[1:],minlength=host.nbody);branches=[]
    for b in np.flatnonzero(children[1:]==0)+1:
        path=[]
        while b:path.append(int(b));b=host.body_parentid[b]
        branches.append(path[::-1])
    m.nbranch=len(branches);m.body_branches=np.array([b for path in branches for b in path],np.int32)
    m.body_branch_start=np.r_[0,np.cumsum([len(path) for path in branches])].astype(np.int32)
    m.mocap_bodyid=np.flatnonzero(host.body_mocapid>=0).astype(np.int32)
    m.mocap_bodyid=m.mocap_bodyid[np.argsort(host.body_mocapid[m.mocap_bodyid])]
    m.body_fluid_ellipsoid=np.zeros(host.nbody,bool)
    fluid=host.geom_fluid.reshape(host.ngeom,mujoco.mjNFLUID)[:,0]>0
    ids=host.geom_bodyid[fluid];m.body_fluid_ellipsoid[ids[host.body_mass[ids]>=types.MJ_MINVAL]]=True
    m.body_fluid_ellipsoid_adr=np.flatnonzero(m.body_fluid_ellipsoid).astype(np.int32)
    m.body_fluid_box_adr=np.flatnonzero((np.arange(host.nbody)>0)&(~m.body_fluid_ellipsoid)&(host.body_mass>=types.MJ_MINVAL)).astype(np.int32)
    m.body_is_free=np.zeros(host.nbody,bool)
    for b in range(1,host.nbody):
        if host.body_jntnum[b]==1 and host.jnt_type[host.body_jntadr[b]]==0:
            dof=host.jnt_dofadr[host.body_jntadr[b]];m.body_is_free[b]=host.tree_dofnum[host.dof_treeid[dof]]==6
    m.body_freeadr=np.flatnonzero(m.body_is_free).astype(np.int32)
    m.jnt_limited_slide_hinge_adr=np.flatnonzero(host.jnt_limited&np.isin(host.jnt_type,(2,3))).astype(np.int32)
    m.jnt_limited_ball_adr=np.flatnonzero(host.jnt_limited&(host.jnt_type==1)).astype(np.int32)
    m.dof_tri_row,m.dof_tri_col=np.triu_indices(host.nv)
    m.body_isdofancestor=np.zeros((host.nbody,m.nv_pad),np.int32)
    for body in range(host.nbody):
        b=body
        while b>0 and host.body_dofnum[b]==0:b=host.body_parentid[b]
        if not host.body_dofnum[b]:continue
        dof=int(host.body_dofadr[b]+host.body_dofnum[b]-1)
        while dof>=0:m.body_isdofancestor[body,dof]=1;dof=int(host.dof_parentid[dof])
    prepare_collision(host,m)
    address,total,scalar,gather=mass_layout(host);m.qLD_block_adr=address;m.qLD_block_total=total;m.M_tiles=[]
    elem=np.full((host.nv,host.nv),-1,np.int32);hrows=np.zeros(host.nC,np.int32)
    for i in range(host.nv):
        at=int(host.M_rowadr[i]);count=int(host.M_rownnz[i]);elem[i,host.M_colind[at:at+count]]=np.arange(at,at+count);hrows[at:at+count]=i
    m.M_elemid=elem;m.M_hinit_i=hrows
    for kind,groups in ((0,scalar),(1,gather)):
        for size in sorted(groups):
            starts=np.asarray(groups[size],np.int32);indices=[]
            if kind:
                dofs=starts[:,None]+np.arange(size)[None,:];i=dofs[:,:,None];j=dofs[:,None,:]
                indices=elem[np.maximum(i,j),np.minimum(i,j)];indices=np.where(indices>=0,indices,host.nC).reshape(-1)
            m.M_tiles.append(SimpleNamespace(adr=starts,size=size,elemid=np.array(indices,np.int32)))
    m.M_tiles=tuple(m.M_tiles);m.qLD_updates=();m.qLD_all_updates=np.array([[0,0,0]],np.int32);m.qLD_level_offsets=np.array([0],np.int32)
    m.M_fullm_i=[];m.M_fullm_j=[];mul=[[] for _ in range(host.nv)]
    for i in range(host.nv):
        j=i
        while j>=0:m.M_fullm_i.append(i);m.M_fullm_j.append(j);j=host.dof_parentid[j]
        for at in range(host.M_rowadr[i],host.M_rowadr[i]+host.M_rownnz[i]):
            j=int(host.M_colind[at]);mul[i].append((j,at))
            if i!=j:mul[j].append((i,at))
    m.M_mulm_rowadr=np.r_[0,np.cumsum([len(v) for v in mul])];m.M_mulm_col=[j for row in mul for j,at in row];m.M_mulm_madr=[at for row in mul for j,at in row]
    for stage,key in ((1,'pos'),(2,'vel'),(3,'acc')):setattr(m,'sensor_stage_'+key+'_adr',np.flatnonzero(host.sensor_needstage==stage).astype(np.int32))
    m.sensor_rne_postconstraint=bool(np.any(np.isin(host.sensor_type,(4,5))))
    for name in specs.MODEL_ARRAYS:
        if name.startswith(('eq_','tendon_','wrap_','sensor_')) and getattr(m,name,None) is None:setattr(m,name,np.empty(0,np.int32))
    for key in ('sensor_pos_adr','sensor_vel_adr'):setattr(m,key,np.empty(0,np.int32))
    m.sensor_acc_adr=m.sensor_stage_acc_adr.copy()
    m.efm_dofblk=np.full(host.nv,-1,np.int32)
    m.geom_plugin_index=np.full(host.ngeom,-1,np.int32)
    m.rangefinder_sensor_adr=np.full(host.nsensor,-1,np.int32)
    m.collision_sensor_adr=np.full(host.nsensor,-1,np.int32)
    m.weld_tactile_id=np.full(host.nbody,-1,np.int32)
    m.ten_wrapadr_site=np.array([0],np.int32)
    m.qD_fullm_i=[];m.qD_fullm_j=[]
    for i in range(host.nv):
        at=int(host.D_rowadr[i]);count=int(host.D_rownnz[i])
        m.qD_fullm_i.extend([i]*count);m.qD_fullm_j.extend(host.D_colind[at:at+count].tolist())
    sizes={k:getattr(m,k) for k in specs.MODEL_SCALARS};sizes.update({'*':1,'nbody_branches':len(m.body_branches),'nbranch_start':len(m.body_branch_start),
        'nbodyfree':len(m.body_freeadr),'nbody_fluid_ellipsoid':len(m.body_fluid_ellipsoid_adr),'nbody_fluid_box':len(m.body_fluid_box_adr),
        'njnt_limited_slide_hinge':len(m.jnt_limited_slide_hinge_adr),'njnt_limited_ball':len(m.jnt_limited_ball_adr),
        'ndof_tri':len(m.dof_tri_row),'nnxn_geom_pair':len(m.nxn_geom_pair),'nnxn_geom_pair_filtered':len(m.nxn_geom_pair_filtered),
        'nqLD_all_updates':1,'nqLD_level_offsets':1,'nM_fullm':len(m.M_fullm_i),'nv_plus_1':host.nv+1,'nM_mulm':len(m.M_mulm_col),
        'nqD_fullm':len(m.qD_fullm_i),'nten_wrapadr_site':1})
    for name in specs.MODEL_ARRAYS:
        value=getattr(m,name,None)
        if name.endswith('_adr') and name.startswith(('sensor_','eq_','tendon_','wrap_')) and value is not None:sizes['n'+name[:-4]]=len(value)
    m._sizes=sizes
    return m


def prepare_collision(host,m):
    g1,g2=np.triu_indices(host.ngeom,k=1);m.nxn_geom_pair=np.stack((g1,g2),axis=1)
    b1,b2=host.geom_bodyid[g1],host.geom_bodyid[g2];w1,w2=host.body_weldid[b1],host.body_weldid[b2]
    p1,p2=host.body_weldid[host.body_parentid[w1]],host.body_weldid[host.body_parentid[w2]]
    parent=not bool(host.opt.disableflags&types.DisableBit.FILTERPARENT)
    mask=((host.geom_contype[g1]&host.geom_conaffinity[g2])|(host.geom_contype[g2]&host.geom_conaffinity[g1]))!=0
    keep=mask&(w1!=w2)&~((host.body_dofnum[w1]==0)&(host.body_dofnum[w2]==0))
    keep&=~(parent&(w1!=0)&(w2!=0)&((w1==p2)|(w2==p1)))
    keep&=~np.isin((b1<<16)+b2,host.exclude_signature)
    pair=np.where(keep,-1,-2).astype(np.int32)
    for pid,(i,j) in enumerate(zip(host.pair_geom1,host.pair_geom2)):
        i,j=sorted((int(i),int(j)));pair[(i*(2*host.ngeom-i-3))//2+j-1]=pid
    keep=pair>-2;m.nxn_pairid=np.stack((pair,np.full(len(pair),-1,np.int32)),axis=1)
    m.nxn_geom_pair_filtered=m.nxn_geom_pair[keep];m.nxn_pairid_filtered=m.nxn_pairid[keep]
    ntypes=len(types.GeomType);a=np.minimum(host.geom_type[g1[keep]],host.geom_type[g2[keep]]);b=np.maximum(host.geom_type[g1[keep]],host.geom_type[g2[keep]])
    m.geom_pair_type_count=tuple(np.bincount(a*(2*ntypes-a-1)//2+b,minlength=ntypes*(ntypes+1)//2))
    m.has_boxbox_margin=False
    for i,j,pid in zip(g1[keep],g2[keep],pair[keep]):
        if host.geom_type[i]==6 and host.geom_type[j]==6:
            margin=host.pair_margin[pid] if pid>=0 else max(host.geom_margin[i],host.geom_margin[j])
            m.has_boxbox_margin|=bool(margin)


def host_array(value,layout,sizes):
    dimensions,name=layout;kind=storage_type(name)
    shape=tuple(sizes.get(d,0) if isinstance(d,str) and not d.isdigit() else int(d) for d in dimensions)
    if value is None:return np.zeros(shape+kind.components,dtype=kind.numpy_dtype),shape,kind
    value=np.asarray(value,dtype=kind.numpy_dtype)
    expected=math.prod(shape)*math.prod(kind.components)
    if value.size!=expected:
        if value.size==0 and expected==0:return np.zeros(shape+kind.components,dtype=kind.numpy_dtype),shape,kind
        raise ValueError('Prepared model field shape differs: '+str((value.shape,layout,shape)))
    return np.ascontiguousarray(value.reshape(shape+kind.components)),shape,kind


def upload(prepared):
    """Own every device array; immutable host descriptions remain inspectable."""
    m=SimpleNamespace(**vars(prepared));m._cuda_native=True
    for name,layout in specs.MODEL_ARRAYS.items():
        value,shape,kind=host_array(getattr(prepared,name,None),layout,prepared._sizes)
        setattr(m,name,cuda.array(value,shape=shape,dtype=kind))
    m.body_tree=tuple(cuda.array(a,dtype=int) for a in prepared.body_tree)
    m.M_tiles=tuple(SimpleNamespace(size=t.size,adr=cuda.array(t.adr,dtype=int),elemid=cuda.array(t.elemid,dtype=int)) for t in prepared.M_tiles)
    m.opt=SimpleNamespace(**vars(prepared.opt))
    for name,layout in specs.OPTION_ARRAYS.items():
        value,shape,kind=host_array(getattr(prepared.opt,name),layout,{'*':1})
        setattr(m.opt,name,cuda.array(value,shape=shape,dtype=kind))
    m.stat=SimpleNamespace(meaninertia=cuda.array([prepared.stat.meaninertia],dtype=float))
    return m
