"""Borrowed-array plane contacts: current materials, geometry and pooled output."""
import ctypes as ct
import hashlib
from ..dynamics.mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap

PARAMETERS=(('size','geom_size','vec3',3),('solmix','geom_solmix','float32',1),('solref','geom_solref','vec2',2),('solimp','geom_solimp','vec5',5),
    ('friction','geom_friction','vec3',3),('margin','geom_margin','float32',1),('gap','geom_gap','float32',1),('adhesion','geom_adhesion','float32',1),
    ('pair_solref','pair_solref','vec2',2),('pair_solreffriction','pair_solreffriction','vec2',2),('pair_solimp','pair_solimp','vec5',5),
    ('pair_friction','pair_friction','vec5',5),('pair_margin','pair_margin','float32',1),('pair_gap','pair_gap','float32',1),('pair_adhesion','pair_adhesion','float32',1))
CONTACTS=(('dist','float32'),('pos','vec3'),('frame','mat33'),('includemargin','float32'),('friction','vec5'),('solref','vec2'),('solreffriction','vec2'),('solimp','vec5'),
    ('adhesion','float32'),('dim','int32'),('geom','vec2i'),('worldid','int32'),('type','int32'),('geomcollisionid','int32'))
MESH_INTS=('mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','mesh_polynum','mesh_polyadr','mesh_polyvertadr','mesh_polyvertnum','mesh_polyvert','mesh_polymapadr','mesh_polymapnum','mesh_polymap')
POINTERS=('geom_type','condim','priority','pair_dim','geom_dataid','dataid_rows',*MESH_INTS,'mesh_vert','mesh_polynormal','xpos','xmat',
    'candidate_count','candidate_geom','candidate_ids','candidate_world','contact_count','contact_dim','contact_geom','contact_world','contact_type','contact_collisionid','contact_address',
    'contact_dist','contact_pos','contact_frame','contact_margin','contact_friction','contact_solref','contact_solreffriction','contact_solimp','contact_adhesion')


class ContactParameter(ct.Structure):_fields_=[('data',ct.c_void_p),('rows',ct.c_int)]
class ContactInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','geoms','explicit_pairs','capacity','address_columns')]+[(key,ContactParameter) for key,*_ in PARAMETERS]+[(k,ct.c_int if k=='dataid_rows' else ct.c_void_p) for k in POINTERS]


def qualify(model,data,*,implementation='upstream'):
    from libero.gpu.host.runtime.cuda_api import type_api
    types = type_api(model)
    if implementation not in ('upstream','cuda'):raise ValueError('plane_collision must be upstream or cuda')
    reason=None
    if implementation=='cuda':
        if model.opt.integrator!=types.IntegratorType.EULER:reason='plane collision caller requires Euler'
        elif model.opt.enableflags & types.EnableBit.SLEEP:reason='sleep/incremental collision retained upstream'
        elif model.nflex:reason='flex collision caller retained upstream'
        elif model.ngeom<1 or data.nworld<1:reason='empty plane collision domain'
    return dict(requested=implementation,selected='upstream' if reason else implementation,fallback=reason,model=id(model),data=id(data))


def validate_mesh(dataid,types,arrays):
    """Cold structural spans; poses/materials remain device-resident per call."""
    import numpy as np
    adr,num=arrays['mesh_vertadr'],arrays['mesh_vertnum'];pa,pn=arrays['mesh_polyadr'],arrays['mesh_polynum'];nm=len(adr)
    if any(len(arrays[k])!=nm for k in ('mesh_vertnum','mesh_graphadr','mesh_polyadr','mesh_polynum')):raise ValueError('Plane mesh table lengths differ')
    ids=dataid[:,types==7]
    if np.any(ids < -1) or np.any(ids>=nm):raise ValueError('Plane mesh geometry id exceeds table')
    nv=len(arrays['mesh_vert']);nf=len(arrays['mesh_polynormal']);pv=arrays['mesh_polyvert'];fg=arrays['mesh_graph']
    for first,count,end in ((adr,num,nv),(pa,pn,nf),(arrays['mesh_polyvertadr'],arrays['mesh_polyvertnum'],len(pv)),(arrays['mesh_polymapadr'],arrays['mesh_polymapnum'],len(arrays['mesh_polymap']))):
        if np.any(first<0) or np.any(count<0) or np.any(first.astype(np.int64)+count>end):raise ValueError('Plane mesh span exceeds storage')
    if any(len(arrays[k])!=nf for k in ('mesh_polyvertadr','mesh_polyvertnum')) or any(len(arrays[k])!=nv for k in ('mesh_polymapadr','mesh_polymapnum')):raise ValueError('Plane mesh polygon/map lengths differ')
    for m in range(nm):
        f0,fc=int(pa[m]),int(pn[m]);v0,vc=int(adr[m]),int(num[m])
        if vc<1:raise ValueError('Plane mesh must have vertices')
        for f in range(f0,f0+fc):
            start,count=int(arrays['mesh_polyvertadr'][f]),int(arrays['mesh_polyvertnum'][f]);values=pv[start:start+count]
            if count<3 or np.any(values<0) or np.any(values>=vc):raise ValueError('Plane mesh polygon vertices differ')
        for v in v0+np.flatnonzero(arrays['mesh_polymapnum'][v0:v0+vc]):
            start,count=int(arrays['mesh_polymapadr'][v]),int(arrays['mesh_polymapnum'][v]);values=arrays['mesh_polymap'][start:start+count]
            if np.any(values<0) or np.any(values>=fc):raise ValueError('Plane mesh vertex polygon map differs')
        start=int(arrays['mesh_graphadr'][m])
        if start==-1:continue
        if start<0 or start+2>len(fg):raise ValueError('Plane mesh graph address differs')
        count=int(fg[start]);base=start+2+2*count
        if count<1 or base>len(fg):raise ValueError('Plane mesh graph span differs')
        local=fg[start+2:start+2+count];vertices=fg[start+2+count:base]
        if np.any(local<0) or np.any(base+local>=len(fg)) or np.any(vertices<0) or np.any(vertices>=vc):raise ValueError('Plane mesh graph indices differ')
        # Header[1] is not used by support traversal. Graph storage also has
        # faces after the adjacency lists: validate only each terminated list,
        # avoiding an invented edge-count interpretation of that header.
        for i,offset in enumerate(local):
            first=base+int(offset);stop=base+int(local[i+1]) if i+1<count else len(fg)
            if stop<=first or stop>len(fg):raise ValueError('Plane mesh graph adjacency offsets differ')
            values=fg[first:stop];terminator=np.flatnonzero(values<0)
            if not len(terminator) or values[terminator[0]]!=-1 or np.any(values[:terminator[0]]>=count):raise ValueError('Plane mesh graph terminator/neighbor differs')



class PlaneContacts:
    component='plane'
    argument_type=ContactInputs
    entry=None
    families=['plane-sphere','plane-capsule','plane-ellipsoid','plane-cylinder','plane-box','plane-mesh']
    def __init__(self,model,data,context,library,stream,*,admission):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from libero.gpu.host.runtime.cuda_api import type_api
        types = type_api(model)
        if admission['model']!=id(model) or admission['data']!=id(data) or admission['selected']!='cuda':raise ValueError('Plane collision admission differs')
        if tuple(int(x) for x in (types.GeomType.PLANE,types.GeomType.SPHERE,types.GeomType.CAPSULE,types.GeomType.ELLIPSOID,types.GeomType.CYLINDER,types.GeomType.BOX,types.GeomType.MESH,types.ContactType.CONSTRAINT,types.ContactType.SENSOR))!=(0,2,3,4,5,6,7,1,2) or types.MJ_MINVAL!=1.e-15 or types.MJ_MINMU!=1.e-5:raise ValueError('Pinned plane constants differ')
        n,ng,nc,npair=data.nworld,model.ngeom,data.naconmax,model.npair
        columns=data.contact.efc_address.shape[1]
        if min(n,ng,columns)<1 or min(nc,npair)<0 or max(n*ng*9,nc*max(9,columns),npair)>2**31-256:raise ValueError('Plane dimensions exceed index storage')
        self.model,self.data,self.context,self.stream=model,data,context,stream
        self.path,self.library=load_leaf(self.component,self.argument_type,library,stream,entry=self.entry);self.arrays=a={};parameters={}
        for key,field,kind,width in PARAMETERS:
            value=getattr(model,field);count=npair if key.startswith('pair_') else ng
            if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*count*width>2**31-256:raise ValueError('Plane broadcast differs: '+key)
            bind_array(a,key,value,getattr(wp,kind,None) or getattr(types,kind),(value.shape[0],count),stream);parameters[key]=ContactParameter(value.ptr,value.shape[0])
        for key,field,shape in (('geom_type','geom_type',(ng,)),('condim','geom_condim',(ng,)),('priority','geom_priority',(ng,)),('pair_dim','pair_dim',(npair,)),('geom_dataid','geom_dataid',model.geom_dataid.shape)):
            bind_array(a,key,getattr(model,field),wp.int32,shape,stream)
        if model.geom_dataid.ndim!=2 or model.geom_dataid.shape[0]<1 or model.geom_dataid.shape[1]!=ng:raise ValueError('Plane mesh id broadcast differs')
        for key in MESH_INTS+('mesh_vert','mesh_polynormal'):
            value=getattr(model,key)
            if value.ndim!=1 or value.size*max(value.dtype._length_ if hasattr(value.dtype,'_length_') else 1,1)>2**31-256:raise ValueError('Plane mesh table layout differs')
            bind_array(a,key,value,wp.vec3 if key in ('mesh_vert','mesh_polynormal') else wp.int32,value.shape,stream)
        validate_mesh(a['geom_dataid'].numpy(),a['geom_type'].numpy(),{k:a[k].numpy() for k in MESH_INTS+('mesh_vert','mesh_polynormal')})
        for key,field,dtype,shape in (('xpos','geom_xpos',wp.vec3,(n,ng)),('xmat','geom_xmat',wp.mat33,(n,ng)),('candidate_count','ncollision',wp.int32,(1,)),('contact_count','nacon',wp.int32,(1,))):bind_array(a,key,getattr(data,field),dtype,shape,stream)
        for key,field,dtype in (('candidate_geom','collision_pair',wp.vec2i),('candidate_ids','collision_pairid',wp.vec2i),('candidate_world','collision_worldid',wp.int32)):bind_array(a,key,getattr(context,field),dtype,(nc,),stream)
        output_names={'includemargin':'margin','worldid':'world','geomcollisionid':'collisionid'}
        for field,kind in CONTACTS:
            key='contact_'+output_names.get(field,field);bind_array(a,key,getattr(data.contact,field),getattr(wp,kind,None) or getattr(types,kind),(nc,),stream)
        bind_array(a,'contact_address',data.contact.efc_address,wp.int32,(nc,columns),stream)
        reject_output_overlap(a,tuple(k for k in a if k.startswith('contact_')))
        self.arguments=ContactInputs(worlds=n,geoms=ng,explicit_pairs=npair,capacity=nc,address_columns=columns,**parameters,
            **{k:model.geom_dataid.shape[0] if k=='dataid_rows' else a[k].ptr for k in POINTERS})
        self.launch=getattr(self.library,'flash_mjwarp_'+self.component+('_'+self.entry if self.entry else '')+'_launch');self.launch.argtypes=[ct.POINTER(ContactInputs),ct.c_void_p];self.launch.restype=ct.c_int

    @property
    def identity(self):return dict(source_id=source_id(self.component),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),numerical_workspace_bytes=0,shared_bytes=0,
        families=self.families,values='borrow-current-input-no-cache',publication='pinned pooled atomic contact append; attempted count retained beyond capacity')
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Plane collision launch failed: CUDA error '+str(error))
