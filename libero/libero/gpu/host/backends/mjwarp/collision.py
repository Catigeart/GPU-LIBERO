# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Explicit full rigid collision schedule with independent broadphase selection."""
from mujoco_warp._src import collision_driver,collision_core,collision_convex,collision_primitive,types
from libero.gpu.host.runtime.collision.mjwarp_convex_collision import ConvexContacts,qualify as qualify_convex
from libero.gpu.host.runtime.collision.mjwarp_box import BoxContacts,qualify as qualify_box
from libero.gpu.host.runtime.collision.mjwarp_plane import PlaneContacts,qualify as qualify_plane
from libero.gpu.host.runtime.collision.mjwarp_broadphase import BroadphaseNumerics,qualify


def broadphase_product(root,*,fast_compile=False,component="broadphase"):
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_'+component+'_fp32';library=folder/('mjwarp_'+component+'_fp32.so')
    if not library.is_file():raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class Collision:
    def __init__(self,model,data,stream,*,broadphase='upstream',library=None,admission=None,planes='upstream',plane_library=None,plane_admission=None,boxes='upstream',box_library=None,box_admission=None,convex='upstream',convex_library=None,convex_admission=None):
        self.model,self.data=model,data
        self.admission=qualify(model,data,implementation=broadphase) if admission is None else admission
        if self.admission['model']!=id(model) or self.admission['data']!=id(data):raise ValueError('Collision admission owner differs')
        if self.admission['requested']!=broadphase:raise ValueError('Collision admission selector differs')
        self.plane_admission=qualify_plane(model,data,implementation=planes) if plane_admission is None else plane_admission
        if self.plane_admission['model']!=id(model) or self.plane_admission['data']!=id(data) or self.plane_admission['requested']!=planes:raise ValueError('Plane collision admission owner/selector differs')
        self.box_admission=qualify_box(model,data,implementation=boxes) if box_admission is None else box_admission
        if self.box_admission['model']!=id(model) or self.box_admission['data']!=id(data) or self.box_admission['requested']!=boxes:raise ValueError('Box collision admission owner/selector differs')
        self.convex_admission=qualify_convex(model,data,implementation=convex) if convex_admission is None else convex_admission
        if self.convex_admission['model']!=id(model) or self.convex_admission['data']!=id(data) or self.convex_admission['requested']!=convex:raise ValueError('Convex collision admission owner/selector differs')
        self.selected='cuda' if 'cuda' in (self.admission['selected'],self.plane_admission['selected'],self.box_admission['selected'],self.convex_admission['selected']) else 'upstream'
        self.context=self.binding=self.planes=self.boxes=self.convex=None
        self.disabled=data.naconmax==0 or bool(model.opt.disableflags & (types.DisableBit.CONSTRAINT|types.DisableBit.CONTACT))
        if self.selected=='cuda':
            self.context=collision_core.create_collision_context(data.naconmax)
        if self.admission['selected']=='cuda':
            self.binding=BroadphaseNumerics(model,data,self.context,library,stream,admission=self.admission)

        if self.plane_admission['selected']=='cuda':
            self.planes=PlaneContacts(model,data,self.context,plane_library,stream,admission=self.plane_admission)

        if self.box_admission['selected']=='cuda':
            self.boxes=BoxContacts(model,data,self.context,box_library,stream,admission=self.box_admission)

        if self.convex_admission['selected']=='cuda':
            self.convex=ConvexContacts(model,data,self.context,convex_library,stream,admission=self.convex_admission)

    @property
    def convex_identity(self):
        return dict(requested=self.convex_admission['requested'],selected=self.convex_admission['selected'],fallback=self.convex_admission['fallback'],**(self.convex.identity if self.convex is not None else {}))

    @property
    def box_identity(self):
        return dict(requested=self.box_admission['requested'],selected=self.box_admission['selected'],fallback=self.box_admission['fallback'],**(self.boxes.identity if self.boxes is not None else {}))

    @property
    def plane_identity(self):
        return dict(requested=self.plane_admission['requested'],selected=self.plane_admission['selected'],fallback=self.plane_admission['fallback'],**(self.planes.identity if self.planes is not None else {}))

    @property
    def identity(self):
        return dict(requested=self.admission['requested'],selected=self.admission['selected'],fallback=self.admission['fallback'],disabled=self.disabled,
            context_bytes=sum(getattr(self.context,k).capacity for k in ('collision_pair','collision_pairid','collision_worldid')) if self.context is not None else 0,
            context='one retained candidate-output context reused across Graph substeps' if self.context is not None else 'upstream',
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        m,d=self.model,self.data
        if self.selected=='upstream':collision_driver.collision(m,d);return
        if self.disabled:d.nacon.zero_();return
        d.ncollision.zero_();d.nacon.zero_()
        if self.binding is not None:self.binding.enqueue()
        elif m.opt.broadphase==types.BroadphaseType.NXN:collision_driver.nxn_broadphase(m,d,self.context)
        else:collision_driver.sap_broadphase(m,d,self.context)
        if self.planes is None and self.boxes is None and self.convex is None:collision_driver._narrowphase(m,d,self.context)
        else:self.narrowphase()
        if m.callback.contactfilter:m.callback.contactfilter(m,d)

    def narrowphase(self):
        m,d,ctx=self.model,self.data,self.context
        table=collision_driver.MJ_COLLISION_TABLE
        if m.has_boxbox_margin or m.opt.disableflags & types.DisableBit.NATIVECCD:
            table=table.copy();table[(types.GeomType.BOX,types.GeomType.BOX)]=types.CollisionType.PRIMITIVE
        convex=[key for key,value in table.items() if value==types.CollisionType.CONVEX]
        residual=[key for key,value in table.items() if value==types.CollisionType.PRIMITIVE
            and not (self.planes is not None and key[0]==types.GeomType.PLANE)
            and not (self.boxes is not None and tuple(map(int,key)) in ((2,6),(3,6),(6,6)))]
        if self.convex is None:collision_convex.convex_narrowphase(m,d,ctx,convex)
        else:self.convex.enqueue()
        if self.planes is not None:self.planes.enqueue()
        if self.boxes is not None:self.boxes.enqueue()
        index=lambda i,j:i*(2*len(types.GeomType)-i-1)//2+j
        if any(m.geom_pair_type_count[index(int(i),int(j))] for i,j in residual):collision_primitive.primitive_narrowphase(m,d,ctx,residual)
        if m.has_sdf_geom:collision_driver.sdf_narrowphase(m,d,ctx)
