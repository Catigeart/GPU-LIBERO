"""Device body-tree traversal and ordered geom invocation generation.

One CUDA lane owns one independent world. Scratch is bounded by the sum of two
body-tree sizes, never by the model's mesh-expanded nbvh.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.func
def _dot(a0: scalar.wp_real, a1: scalar.wp_real, a2: scalar.wp_real,
         b0: scalar.wp_real, b1: scalar.wp_real, b2: scalar.wp_real) -> scalar.wp_real:
    return a0*b0+a1*b1+a2*b2


@wp.func
def _base(i: wp.int32, a: wp.int32, b: wp.int32) -> wp.int32:
    if i == 0:
        return a
    return b


@wp.func
def _normal(mat: wp.array(dtype=scalar.wp_real), base: wp.int32,
            axis: wp.int32, coord: wp.int32) -> scalar.wp_real:
    return mat[base+3*coord+axis]


@wp.func
def _center(aa: wp.array(dtype=scalar.wp_real), ab: wp.int32,
            pos: wp.array(dtype=scalar.wp_real), pb: wp.int32,
            mat: wp.array(dtype=scalar.wp_real), mb: wp.int32,
            axis: wp.int32) -> scalar.wp_real:
    return _dot(aa[ab],aa[ab+1],aa[ab+2],
                mat[mb+3*axis],mat[mb+3*axis+1],mat[mb+3*axis+2])+pos[pb+axis]


@wp.func
def _obb(aa: wp.array(dtype=scalar.wp_real), ab0: wp.int32, ab1: wp.int32,
         pos: wp.array(dtype=scalar.wp_real), pb0: wp.int32, pb1: wp.int32,
         mat: wp.array(dtype=scalar.wp_real), mb0: wp.int32, mb1: wp.int32,
         cached: wp.int32, product: wp.array(dtype=scalar.wp_real),
         offset: wp.array(dtype=scalar.wp_real), sb: wp.int32,
         initialize: wp.int32) -> wp.int32:
    # Bit 0 overlap, bit 1 updated initialize, higher bits first reject axis+1.
    inf10=aa[ab0+3]>=scalar.wp_real(1.0e10)
    inf11=aa[ab0+4]>=scalar.wp_real(1.0e10)
    inf12=aa[ab0+5]>=scalar.wp_real(1.0e10)
    inf20=aa[ab1+3]>=scalar.wp_real(1.0e10)
    inf21=aa[ab1+4]>=scalar.wp_real(1.0e10)
    inf22=aa[ab1+5]>=scalar.wp_real(1.0e10)
    if (inf10 and inf11 and inf12) or (inf20 and inf21 and inf22):
        return 1+2*initialize
    c00=scalar.wp_real(0.0);c01=scalar.wp_real(0.0);c02=scalar.wp_real(0.0)
    c10=scalar.wp_real(0.0);c11=scalar.wp_real(0.0);c12=scalar.wp_real(0.0)
    if cached == 0:
        c00=_center(aa,ab0,pos,pb0,mat,mb0,0)
        c01=_center(aa,ab0,pos,pb0,mat,mb0,1)
        c02=_center(aa,ab0,pos,pb0,mat,mb0,2)
        c10=_center(aa,ab1,pos,pb1,mat,mb1,0)
        c11=_center(aa,ab1,pos,pb1,mat,mb1,1)
        c12=_center(aa,ab1,pos,pb1,mat,mb1,2)
    if cached != 0 and initialize != 0:
        for i in range(2):
            mi=_base(i,mb0,mb1)
            pi=_base(i,pb0,pb1)
            for j in range(2):
                mj=_base(j,mb0,mb1)
                for k in range(3):
                    for l in range(3):
                        product[sb+18*i+9*j+3*k+l]=_dot(
                            _normal(mat,mi,l,0),_normal(mat,mi,l,1),_normal(mat,mi,l,2),
                            _normal(mat,mj,k,0),_normal(mat,mj,k,1),_normal(mat,mj,k,2))
                    offset[(sb//36)*12+6*i+3*j+k]=_dot(
                        pos[pi],pos[pi+1],pos[pi+2],
                        _normal(mat,mj,k,0),_normal(mat,mj,k,1),_normal(mat,mj,k,2))
        initialize=0
    for j in range(2):
        if (j==0 and (inf20 or inf21 or inf22)) or (j==1 and (inf10 or inf11 or inf12)):
            continue
        mj=_base(j,mb0,mb1)
        for k in range(3):
            n0=_normal(mat,mj,k,0);n1=_normal(mat,mj,k,1);n2=_normal(mat,mj,k,2)
            p0=scalar.wp_real(0.0);p1=scalar.wp_real(0.0)
            r0=scalar.wp_real(0.0);r1=scalar.wp_real(0.0)
            for i in range(2):
                ab=_base(i,ab0,ab1);mi=_base(i,mb0,mb1)
                proj=scalar.wp_real(0.0);radius=scalar.wp_real(0.0)
                if cached == 0:
                    if i==0:
                        proj=_dot(c00,c01,c02,n0,n1,n2)
                    else:
                        proj=_dot(c10,c11,c12,n0,n1,n2)
                    d0=_dot(_normal(mat,mi,0,0),_normal(mat,mi,0,1),_normal(mat,mi,0,2),n0,n1,n2)
                    d1=_dot(_normal(mat,mi,1,0),_normal(mat,mi,1,1),_normal(mat,mi,1,2),n0,n1,n2)
                    d2=_dot(_normal(mat,mi,2,0),_normal(mat,mi,2,1),_normal(mat,mi,2,2),n0,n1,n2)
                    radius=wp.abs(aa[ab+3]*d0)+wp.abs(aa[ab+4]*d1)+wp.abs(aa[ab+5]*d2)
                else:
                    adr=sb+18*i+9*j+3*k
                    proj=aa[ab]*product[adr]+aa[ab+1]*product[adr+1]+aa[ab+2]*product[adr+2]+offset[(sb//36)*12+6*i+3*j+k]
                    radius=wp.abs(aa[ab+3]*product[adr])+wp.abs(aa[ab+4]*product[adr+1])+wp.abs(aa[ab+5]*product[adr+2])
                if i==0:
                    p0=proj;r0=radius
                else:
                    p1=proj;r1=radius
            if r0+r1<wp.abs(p1-p0):
                return 2*initialize+4*(3*j+k+1)
    return 1+2*initialize


@wp.func
def _sphere(g1: wp.int32,g2: wp.int32,
            rbound: wp.array(dtype=scalar.wp_real),gtype: wp.array(dtype=wp.int32),
            pos: wp.array(dtype=scalar.wp_real),mat: wp.array(dtype=scalar.wp_real),
            pbase: wp.int32,mbase: wp.int32) -> wp.int32:
    p1=pbase+3*g1;p2=pbase+3*g2
    if rbound[g1]>scalar.wp_real(0.0) and rbound[g2]>scalar.wp_real(0.0):
        bound=rbound[g1]+rbound[g2]+scalar.wp_real(0.0)
        d0=pos[p1]-pos[p2];d1=pos[p1+1]-pos[p2+1];d2=pos[p1+2]-pos[p2+2]
        if d0*d0+d1*d1+d2*d2>bound*bound:
            return 0
    if gtype[g1]==0 and rbound[g2]>scalar.wp_real(0.0):
        m1=mbase+9*g1
        d0=pos[p2]-pos[p1];d1=pos[p2+1]-pos[p1+1];d2=pos[p2+2]-pos[p1+2]
        if _dot(d0,d1,d2,mat[m1+2],mat[m1+5],mat[m1+8])>scalar.wp_real(0.0)+rbound[g2]:
            return 0
    if gtype[g2]==0 and rbound[g1]>scalar.wp_real(0.0):
        m2=mbase+9*g2
        d0=pos[p1]-pos[p2];d1=pos[p1+1]-pos[p2+1];d2=pos[p1+2]-pos[p2+2]
        if _dot(d0,d1,d2,mat[m2+2],mat[m2+5],mat[m2+8])>scalar.wp_real(0.0)+rbound[g1]:
            return 0
    return 1


@wp.func
def _suppressed(g1: wp.int32,g2: wp.int32,merged: wp.int32,
                start: wp.int32,end: wp.int32,
                pair1: wp.array(dtype=wp.int32),pair2: wp.array(dtype=wp.int32)) -> wp.int32:
    if merged != 0:
        for k in range(start,end):
            if (pair1[k]==g1 and pair2[k]==g2) or (pair1[k]==g2 and pair2[k]==g1):
                return 1
    return 0


@wp.func
def _surface(aa: wp.array(dtype=scalar.wp_real), base: wp.int32) -> scalar.wp_real:
    x=aa[base+3]-aa[base];y=aa[base+4]-aa[base+1];z=aa[base+5]-aa[base+2]
    return x*y+y*z+z*x


@wp.kernel(enable_backward=False)
def _tree_kernel(stage:wp.array(dtype=wp.int32),stage_count:wp.array(dtype=wp.int32),
                 active_in:wp.array(dtype=wp.int32),
                 xipos:wp.array(dtype=scalar.wp_real),ximat:wp.array(dtype=scalar.wp_real),
                 geom_pos:wp.array(dtype=scalar.wp_real),geom_mat:wp.array(dtype=scalar.wp_real),
                 body_adr:wp.array(dtype=wp.int32),body_num:wp.array(dtype=wp.int32),
                 geom_adr:wp.array(dtype=wp.int32),geom_num:wp.array(dtype=wp.int32),
                 child:wp.array(dtype=wp.int32),geomid:wp.array(dtype=wp.int32),
                 bvh_aa:wp.array(dtype=scalar.wp_real),geom_aa:wp.array(dtype=scalar.wp_real),
                 rbound:wp.array(dtype=scalar.wp_real),gtype:wp.array(dtype=wp.int32),
                 pair1:wp.array(dtype=wp.int32),pair2:wp.array(dtype=wp.int32),
                 stack1:wp.array(dtype=wp.int32),stack2:wp.array(dtype=wp.int32),
                 product:wp.array(dtype=scalar.wp_real),offset:wp.array(dtype=scalar.wp_real),
                 out_req:wp.array(dtype=wp.int32),out_visit:wp.array(dtype=wp.int32),
                 out_stack:wp.array(dtype=wp.int32),out_leaf:wp.array(dtype=wp.int32),
                 out_active:wp.array(dtype=wp.int32),
                 req_count:wp.array(dtype=wp.int32),visit_count:wp.array(dtype=wp.int32),
                 stack_count:wp.array(dtype=wp.int32),leaf_count:wp.array(dtype=wp.int32),
                 status:wp.array(dtype=wp.int32),first_stage:wp.array(dtype=wp.int32),
                 stage_slots:wp.int32,req_slots:wp.int32,visit_slots:wp.int32,
                 stack_slots:wp.int32,leaf_slots:wp.int32,nbody:wp.int32,ngeom:wp.int32,
                 body_nodes:wp.int32,max_stack:wp.int32,npair:wp.int32):
    w=wp.tid()
    req_count[w]=0;visit_count[w]=0;stack_count[w]=0;leaf_count[w]=0
    status[w]=0;first_stage[w]=-1
    count=stage_count[w]
    if count<0 or count>stage_slots:
        status[w]=1;first_stage[w]=0;return
    if active_in[w]==0:
        return
    for i in range(body_nodes):
        out_active[w*body_nodes+i]=0
    rc=wp.int32(0);vc=wp.int32(0);sc=wp.int32(0);lc=wp.int32(0)
    for s in range(count):
        sr=(w*stage_slots+s)*6
        b1=stage[sr];b2=stage[sr+1];merged=stage[sr+2]
        start=stage[sr+3];end=stage[sr+4];route=stage[sr+5]
        if b1<0 or b1>=nbody or b2<0 or b2>=nbody or (merged!=0 and merged!=1) or start<0 or end<start or end>npair or (route!=1 and route!=2):
            status[w]=2;first_stage[w]=s;break
        if route==2:
            for g1 in range(geom_adr[b1],geom_adr[b1]+geom_num[b1]):
                for g2 in range(geom_adr[b2],geom_adr[b2]+geom_num[b2]):
                    if _suppressed(g1,g2,merged,start,end,pair1,pair2)==0:
                        if rc>=req_slots:
                            status[w]=6;first_stage[w]=s;break
                        rr=(w*req_slots+rc)*4
                        out_req[rr]=s;out_req[rr+1]=g1;out_req[rr+2]=g2;out_req[rr+3]=0;rc+=1
                if status[w]!=0:break
            if status[w]!=0:break
            continue
        adr1=body_adr[b1];adr2=body_adr[b2]
        num1=body_num[b1];num2=body_num[b2]
        if adr1<0 or adr2<0 or num1<=0 or num2<=0 or adr1+num1>body_nodes or adr2+num2>body_nodes:
            status[w]=3;first_stage[w]=s;break
        limit=num1+num2
        if limit>max_stack:
            status[w]=4;first_stage[w]=s;break
        sb=w*max_stack
        stack1[sb]=0;stack2[sb]=0;nstack=wp.int32(1)
        if sc>=stack_slots:
            status[w]=5;first_stage[w]=s;break
        tr=(w*stack_slots+sc)*5
        out_stack[tr]=s;out_stack[tr+1]=1;out_stack[tr+2]=0;out_stack[tr+3]=0;out_stack[tr+4]=1;sc+=1
        initialize=wp.int32(1)
        while nstack>0:
            nstack-=1
            n1=stack1[sb+nstack];n2=stack2[sb+nstack]
            if n1<0 or n1>=num1 or n2<0 or n2>=num2:
                status[w]=7;first_stage[w]=s;break
            if sc>=stack_slots:
                status[w]=5;first_stage[w]=s;break
            tr=(w*stack_slots+sc)*5
            out_stack[tr]=s;out_stack[tr+1]=0;out_stack[tr+2]=n1;out_stack[tr+3]=n2;out_stack[tr+4]=nstack;sc+=1
            a1=adr1+n1;a2=adr2+n2
            leaf1=child[2*a1]==-1 and child[2*a1+1]==-1
            leaf2=child[2*a2]==-1 and child[2*a2+1]==-1
            id1=geomid[a1];id2=geomid[a2]
            if (leaf1 and (id1< -1 or id1>=ngeom)) or (leaf2 and (id2< -1 or id2>=ngeom)):
                status[w]=7;first_stage[w]=s;break
            if vc>=visit_slots:
                status[w]=8;first_stage[w]=s;break
            vr=(w*visit_slots+vc)*12
            out_visit[vr]=s;out_visit[vr+1]=n1;out_visit[vr+2]=n2
            out_visit[vr+3]=wp.int32(leaf1);out_visit[vr+4]=wp.int32(leaf2)
            out_visit[vr+5]=id1;out_visit[vr+6]=id2;out_visit[vr+7]=nstack
            out_visit[vr+8]=-1;out_visit[vr+9]=-1;out_visit[vr+10]=-1;out_visit[vr+11]=-1;vc+=1
            if leaf1 and leaf2 and id1!=-1 and id2!=-1:
                sphere=_sphere(id1,id2,rbound,gtype,geom_pos,geom_mat,w*ngeom*3,w*ngeom*9)
                out_visit[vr+8]=sphere
                if sphere!=0:
                    leafobb=_obb(geom_aa,6*id1,6*id2,
                                 geom_pos,w*ngeom*3+3*id1,w*ngeom*3+3*id2,
                                 geom_mat,w*ngeom*9+9*id1,w*ngeom*9+9*id2,
                                 0,product,offset,w*36,initialize)
                    out_visit[vr+9]=leafobb&1
                    out_visit[vr+11]=(leafobb>>2)-1
                    if (leafobb&1)!=0:
                        if lc>=leaf_slots:
                            status[w]=9;first_stage[w]=s;break
                        lr=(w*leaf_slots+lc)*3
                        out_leaf[lr]=s;out_leaf[lr+1]=id1;out_leaf[lr+2]=id2;lc+=1
                        if _suppressed(id1,id2,merged,start,end,pair1,pair2)==0:
                            if rc>=req_slots:
                                status[w]=6;first_stage[w]=s;break
                            rr=(w*req_slots+rc)*4
                            out_req[rr]=s;out_req[rr+1]=id1;out_req[rr+2]=id2;out_req[rr+3]=0;rc+=1
                        out_active[w*body_nodes+a1]=1;out_active[w*body_nodes+a2]=1
                continue
            obb=_obb(bvh_aa,6*a1,6*a2,
                     xipos,w*nbody*3+3*b1,w*nbody*3+3*b2,
                     ximat,w*nbody*9+9*b1,w*nbody*9+9*b2,
                     1,product,offset,w*36,initialize)
            out_visit[vr+10]=obb&1;initialize=(obb>>1)&1
            out_visit[vr+11]=(obb>>2)-1
            if (obb&1)==0:continue
            out_active[w*body_nodes+a1]=1;out_active[w*body_nodes+a2]=1
            split1=wp.int32(0)
            if not leaf1 and leaf2:
                split1=1
            elif leaf1 and not leaf2:
                split1=0
            else:
                if _surface(bvh_aa,6*a1)>_surface(bvh_aa,6*a2):split1=1
            for i in range(2):
                c=wp.int32(-1);p1=n1;p2=n2
                if split1!=0:
                    c=child[2*a1+i];p1=c
                else:
                    c=child[2*a2+i];p2=c
                if c!=-1:
                    if nstack>=limit or nstack>=max_stack:
                        status[w]=10;first_stage[w]=s;break
                    stack1[sb+nstack]=p1;stack2[sb+nstack]=p2;nstack+=1
                    if sc>=stack_slots:
                        status[w]=5;first_stage[w]=s;break
                    tr=(w*stack_slots+sc)*5
                    out_stack[tr]=s;out_stack[tr+1]=1;out_stack[tr+2]=p1;out_stack[tr+3]=p2;out_stack[tr+4]=nstack;sc+=1
            if status[w]!=0:break
        if status[w]!=0:break
    req_count[w]=rc;visit_count[w]=vc;stack_count[w]=sc;leaf_count[w]=lc


class CollisionTreeWorkspace:
    """Prepared fixed-model device buffers; launch performs no allocation/copy."""
    def __init__(self, static:dict, nworld:int, stage_slots:int, req_slots:int,
                 visit_slots:int, stack_slots:int, leaf_slots:int, device='cuda:0'):
        if min(nworld,stage_slots,req_slots,visit_slots,stack_slots,leaf_slots)<1:
            raise ValueError('positive capacities required')
        self.device=device;self.nworld=nworld
        self.stage_slots=stage_slots;self.req_slots=req_slots;self.visit_slots=visit_slots
        self.stack_slots=stack_slots;self.leaf_slots=leaf_slots
        self.nbody=len(static['body_bvhadr']);self.ngeom=len(static['geom_rbound'])
        self.body_nodes=int(np.sum(static['body_bvhnum']))
        self.max_stack=2*int(np.max(static['body_bvhnum']))
        self.npair=len(static['pair_geom1'])
        self.body_adr_host=np.asarray(static['body_bvhadr'])
        self.body_num_host=np.asarray(static['body_bvhnum'])
        self.geom_adr_host=np.asarray(static['body_geomadr'])
        self.geom_num_host=np.asarray(static['body_geomnum'])
        def exact(name,dtype,shape):
            a=np.asarray(static[name])
            if a.dtype!=dtype or a.shape!=shape:
                raise ValueError(f'{name} dtype/shape')
            return np.ascontiguousarray(a).ravel()
        ints={'body_adr':('body_bvhadr',(self.nbody,)),
              'body_num':('body_bvhnum',(self.nbody,)),
              'geom_adr':('body_geomadr',(self.nbody,)),
              'geom_num':('body_geomnum',(self.nbody,)),
              'child':('bvh_child',(self.body_nodes,2)),
              'geomid':('bvh_geomid',(self.body_nodes,)),
              'gtype':('geom_type',(self.ngeom,)),
              'pair1':('pair_geom1',(self.npair,)),
              'pair2':('pair_geom2',(self.npair,))}
        doubles={'bvh_aa':('bvh_aabb',(self.body_nodes,6)),
                 'geom_aa':('geom_aabb',(self.ngeom,6)),
                 'rbound':('geom_rbound',(self.ngeom,))}
        for target,(name,shape) in ints.items():
            arr=exact(name,np.int32,shape)
            if len(arr)==0:arr=np.zeros(1,np.int32)
            setattr(self,target,wp.array(arr,dtype=wp.int32,device=device))
        for target,(name,shape) in doubles.items():
            setattr(self,target,wp.array(exact(name,np.float64,shape),dtype=scalar.wp_real,device=device))
        for name,length,dtype in (
            ('stage',nworld*stage_slots*6,wp.int32),('stage_count',nworld,wp.int32),
            ('active_in',nworld,wp.int32),('xipos',nworld*self.nbody*3,scalar.wp_real),
            ('ximat',nworld*self.nbody*9,scalar.wp_real),
            ('geom_pos',nworld*self.ngeom*3,scalar.wp_real),
            ('geom_mat',nworld*self.ngeom*9,scalar.wp_real),
            ('stack1',nworld*self.max_stack,wp.int32),('stack2',nworld*self.max_stack,wp.int32),
            ('product',nworld*36,scalar.wp_real),('offset',nworld*12,scalar.wp_real),
            ('out_req',nworld*req_slots*4,wp.int32),
            ('out_visit',nworld*visit_slots*12,wp.int32),
            ('out_stack',nworld*stack_slots*5,wp.int32),
            ('out_leaf',nworld*leaf_slots*3,wp.int32),
            ('out_active',nworld*self.body_nodes,wp.int32),
            ('req_count',nworld,wp.int32),('visit_count',nworld,wp.int32),
            ('stack_count',nworld,wp.int32),('leaf_count',nworld,wp.int32),
            ('status',nworld,wp.int32),('first_stage',nworld,wp.int32)):
            setattr(self,name,wp.zeros(length,dtype=dtype,device=device))

    def upload(self,stage,stage_count,active,xipos,ximat,geom_pos,geom_mat):
        specs=((stage,(self.nworld,self.stage_slots,6),np.int32,self.stage),
               (stage_count,(self.nworld,),np.int32,self.stage_count),
               (active,(self.nworld,),np.int32,self.active_in),
               (xipos,(self.nworld,self.nbody,3),np.float64,self.xipos),
               (ximat,(self.nworld,self.nbody,9),np.float64,self.ximat),
               (geom_pos,(self.nworld,self.ngeom,3),np.float64,self.geom_pos),
               (geom_mat,(self.nworld,self.ngeom,9),np.float64,self.geom_mat))
        for value,shape,dtype,dst in specs:
            arr=np.asarray(value)
            if arr.shape!=shape or arr.dtype!=dtype:
                raise ValueError('input dtype/shape')
        if np.any((np.asarray(active)!=0)&(np.asarray(active)!=1)):
            raise ValueError('active mask')
        for w in range(self.nworld):
            if active[w]==0:continue
            count=int(stage_count[w])
            # Malformed count is a device error; do not read beyond host slots.
            for s in range(max(0,min(count,self.stage_slots))):
                b1,b2,_,start,end,route=map(int,stage[w,s])
                if b1<0 or b1>=self.nbody or b2<0 or b2>=self.nbody:
                    continue  # device status 2, with first stage
                if route==1:
                    for b in (b1,b2):
                        adr=int(self.body_adr_host[b]);num=int(self.body_num_host[b])
                        if adr<0 or num<=0 or adr+num>self.body_nodes:
                            raise ValueError('invalid body_bvhadr or empty body tree')
                if route==2:
                    for b in (b1,b2):
                        adr=int(self.geom_adr_host[b]);num=int(self.geom_num_host[b])
                        if adr<0 or num<0 or adr+num>self.ngeom:
                            raise ValueError('invalid body geom range')
        for value,shape,dtype,dst in specs:
            dst.assign(np.ascontiguousarray(value).ravel())

    def launch(self,stream=None):
        wp.launch(_tree_kernel,dim=self.nworld,inputs=[
            self.stage,self.stage_count,self.active_in,self.xipos,self.ximat,
            self.geom_pos,self.geom_mat,self.body_adr,self.body_num,
            self.geom_adr,self.geom_num,self.child,self.geomid,self.bvh_aa,
            self.geom_aa,self.rbound,self.gtype,self.pair1,self.pair2,
            self.stack1,self.stack2,self.product,self.offset,self.out_req,
            self.out_visit,self.out_stack,self.out_leaf,self.out_active,
            self.req_count,self.visit_count,self.stack_count,self.leaf_count,
            self.status,self.first_stage,self.stage_slots,self.req_slots,
            self.visit_slots,self.stack_slots,self.leaf_slots,self.nbody,
            self.ngeom,self.body_nodes,self.max_stack,self.npair],device=self.device,
            stream=stream)

    def download(self):
        shapes={'out_req':(self.nworld,self.req_slots,4),
                'out_visit':(self.nworld,self.visit_slots,12),
                'out_stack':(self.nworld,self.stack_slots,5),
                'out_leaf':(self.nworld,self.leaf_slots,3),
                'out_active':(self.nworld,self.body_nodes)}
        result={name:getattr(self,name).numpy().reshape(shape) for name,shape in shapes.items()}
        for name in ('req_count','visit_count','stack_count','leaf_count','status','first_stage'):
            result[name]=getattr(self,name).numpy().reshape(self.nworld)
        return result
