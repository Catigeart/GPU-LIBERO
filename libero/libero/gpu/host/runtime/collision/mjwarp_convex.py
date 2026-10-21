"""Load-only probes for qualified convex building blocks; no production selector."""
import ctypes as ct
import hashlib
from ..dynamics.mjwarp_motion import source_id,load_leaf,bind_array,reject_output_overlap


class ConvexSupportInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','geoms','requests','size_rows','dataid_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','size','vertices','xpos','xmat','world','geom','cached','direction','margin','point','cached_out','vertex_out')]


def validate_requests(world,geom,cached,types,dataid,vertadr,vertnum,graphadr,graph,nworld):
    import numpy as np
    if world.shape!=geom.shape or cached.shape!=geom.shape or np.any(world<0) or np.any(world>=nworld) or np.any(geom<0) or np.any(geom>=len(types)):raise ValueError('Convex request/world/geometry range differs')
    if np.any((types[geom]<2)|(types[geom]>7)):raise ValueError('Convex support probe admits rigid sphere/capsule/ellipsoid/cylinder/box/mesh')
    for w,g,index in zip(world,geom,cached):
        if index<-1:raise ValueError('Convex support cache index differs')
        if types[g]!=7:continue
        mesh=int(dataid[w%len(dataid),g])
        if mesh<0 or mesh>=len(vertadr):raise ValueError('Convex support mesh is absent')
        if index<0:continue
        if graphadr[mesh]<0 or vertnum[mesh]<10:
            if not vertadr[mesh]<=index<vertadr[mesh]+vertnum[mesh]:raise ValueError('Convex exhaustive cache must be an absolute vertex address')
        elif index>=graph[graphadr[mesh]]:raise ValueError('Convex graph cache must be a local graph vertex id')


def bind_geometry(model,data,a,stream):
    """Cold admission for borrowed geometry; launches never recopy metadata."""
    from libero.gpu.host.runtime.cuda_api import array_api
    wp = array_api(stream)
    from .mjwarp_plane import validate_mesh,MESH_INTS
    n,ng=data.nworld,model.ngeom
    rows={}
    for key,field,dtype,width in (('size','geom_size',wp.vec3,3),('geom_dataid','geom_dataid',wp.int32,1)):
        value=getattr(model,field)
        if value.ndim!=2 or value.shape[0]<1 or value.shape[0]*ng*width>2**31-256:raise ValueError('Convex geometry broadcast differs')
        bind_array(a,key,value,dtype,(value.shape[0],ng),stream);rows['size_rows' if key=='size' else 'dataid_rows']=value.shape[0]
    bind_array(a,'geom_type',model.geom_type,wp.int32,(ng,),stream)
    for key in MESH_INTS+('mesh_vert','mesh_polynormal'):
        value=getattr(model,key)
        if value.ndim!=1 or value.size>2**31-256:raise ValueError('Convex mesh table shape differs')
        bind_array(a,'vertices' if key=='mesh_vert' else key,value,wp.vec3 if key in ('mesh_vert','mesh_polynormal') else wp.int32,value.shape,stream)
    mesh={k:a['vertices' if k=='mesh_vert' else k].numpy() for k in MESH_INTS+('mesh_vert','mesh_polynormal')}
    validate_mesh(a['geom_dataid'].numpy(),a['geom_type'].numpy(),mesh)
    bind_array(a,'xpos',data.geom_xpos,wp.vec3,(n,ng),stream);bind_array(a,'xmat',data.geom_xmat,wp.mat33,(n,ng),stream)
    return rows


class SupportProbe:
    """Borrow actual Model/Data geometry and explicit frozen request/output arrays."""
    def __init__(self,model,data,requests,outputs,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        from .mjwarp_plane import validate_mesh,MESH_INTS
        n,ng,nr=data.nworld,model.ngeom,requests.world.shape[0]
        if n<1 or ng<1 or nr<0 or max(n*ng*9,nr*3)>2**31-256:raise ValueError('Convex support dimensions exceed index storage')
        self.model,self.data,self.requests,self.outputs,self.stream=model,data,requests,outputs,stream
        self.path,self.library=load_leaf('convex',ConvexSupportInputs,library,stream);self.arrays=a={}
        rows=bind_geometry(model,data,a,stream)
        for key,dtype in (('world',wp.int32),('geom',wp.int32),('cached',wp.int32),('direction',wp.vec3),('margin',wp.float32)):bind_array(a,key,getattr(requests,key),dtype,(nr,),stream)
        validate_requests(*(a[k].numpy() for k in ('world','geom','cached','geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph')),n)
        for key,dtype in (('point',wp.vec3),('cached_out',wp.int32),('vertex_out',wp.int32)):bind_array(a,key,getattr(outputs,key),dtype,(nr,),stream)
        reject_output_overlap(a,('point','cached_out','vertex_out'))
        self.arguments=ConvexSupportInputs(worlds=n,geoms=ng,requests=nr,**rows,**{k:a[k].ptr for k,_ in ConvexSupportInputs._fields_[5:]})
        self.launch=self.library.flash_mjwarp_convex_support_launch;self.launch.argtypes=[ct.POINTER(ConvexSupportInputs),ct.c_void_p];self.launch.restype=ct.c_int
    @property
    def identity(self):return dict(source_id=source_id('convex'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),component='support-function-probe-only',production_collision_replaced=False,numerical_workspace_bytes=0,shared_bytes=0,values='borrow-current-input-no-cache')
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex support launch failed: CUDA error '+str(error))


class ConvexSimplexInputs(ct.Structure):
    _fields_=[('requests',ct.c_int)]+[(key,ct.c_void_p) for key in ('simplex','count','weights','point','distance','mask')]


class SimplexProbe:
    """Reachable ordered-simplex function probe, not production GJK dispatch."""
    def __init__(self,simplex,count,outputs,library,stream):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        nr=count.shape[0]
        if nr<0 or nr*12>2**31-256:raise ValueError('Convex simplex dimensions exceed index storage')
        self.stream=stream;self.path,self.library=load_leaf('convex',ConvexSimplexInputs,library,stream,entry='simplex');self.arrays=a={}
        bind_array(a,'simplex',simplex,wp.vec3,(nr,4),stream);bind_array(a,'count',count,wp.int32,(nr,),stream)
        values=count.numpy()
        if np.any(values<1) or np.any(values>4):raise ValueError('Convex simplex count must be1..4')
        for key,dtype,shape in (('weights',wp.float32,(nr,4)),('point',wp.vec3,(nr,)),('distance',wp.float32,(nr,)),('mask',wp.int32,(nr,))):bind_array(a,key,getattr(outputs,key),dtype,shape,stream)
        reject_output_overlap(a,('weights','point','distance','mask'))
        self.arguments=ConvexSimplexInputs(requests=nr,**{key:value.ptr for key,value in a.items()});self.launch=self.library.flash_mjwarp_convex_simplex_launch;self.launch.argtypes=[ct.POINTER(ConvexSimplexInputs),ct.c_void_p];self.launch.restype=ct.c_int
    @property
    def identity(self):return dict(source_id=source_id('convex'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),component='ordered-simplex-function-probe-only',production_collision_replaced=False,numerical_workspace_bytes=0,shared_bytes=0)
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex simplex launch failed: CUDA error '+str(error))


class ConvexGJKInputs(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('worlds','geoms','requests','size_rows','dataid_rows')]
    _fields_ += [(k,ct.c_void_p) for k in ('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph','size','vertices','xmat','world','pair','iterations','discrete','cache1','cache2','position1','position2','initial1','initial2','margin1','margin2','tolerance','cutoff','separated_out','dim_out','index1_out','index2_out','vertex1_out','vertex2_out','valid_out','stop_out','iterations_out','queries_out','dist_out','x1_out','x2_out','simplex_out','simplex1_out','simplex2_out','direction_out','lower_out')]


def validate_gjk_queries(world,pair,iterations,discrete,tolerance,cutoff):
    import numpy as np
    n=len(world)
    if pair.shape!=(n,2) or any(v.shape!=(n,) for v in (iterations,discrete,tolerance,cutoff)):raise ValueError('GJK query shapes differ')
    if np.any(iterations<0) or np.any((discrete!=0)&(discrete!=1)) or np.any(tolerance<=0) or np.any(cutoff<0) or not np.all(np.isfinite(tolerance)) or not np.all(np.isfinite(cutoff)):raise ValueError('GJK query controls differ')


GJK_REQUESTS=(('world','int32'),('pair','vec2i'),('iterations','int32'),('discrete','int32'),('cache1','int32'),('cache2','int32'),('position1','vec3'),('position2','vec3'),('initial1','vec3'),('initial2','vec3'),('margin1','float32'),('margin2','float32'),('tolerance','float32'),('cutoff','float32'))
GJK_OUTPUTS=tuple((k,'int32',4 if k in ('vertex1_out','vertex2_out') else 1) for k in ('separated_out','dim_out','index1_out','index2_out','vertex1_out','vertex2_out','valid_out','stop_out','iterations_out','queries_out'))+tuple((k,'vec3' if k not in ('dist_out','lower_out') else 'float32',4 if k in ('simplex_out','simplex1_out','simplex2_out') else 1) for k in ('dist_out','x1_out','x2_out','simplex_out','simplex1_out','simplex2_out','direction_out','lower_out'))


class GJKProbe:
    """Explicit frozen pair/pose/controls; numerical library only, not a dispatcher."""
    def __init__(self,model,data,requests,outputs,library,stream):
        import numpy as np
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        n,ng,nr=data.nworld,model.ngeom,requests.world.shape[0]
        if n<1 or ng<1 or nr<0 or max(n*ng*9,nr*12)>2**31-256:raise ValueError('GJK dimensions exceed index storage')
        self.stream=stream;self.path,self.library=load_leaf('convex',ConvexGJKInputs,library,stream,entry='gjk');self.arrays=a={};rows=bind_geometry(model,data,a,stream)
        for key,kind in GJK_REQUESTS:bind_array(a,key,getattr(requests,key),getattr(wp,kind),(nr,),stream)
        q={k:a[k].numpy() for k,_ in GJK_REQUESTS}
        validate_gjk_queries(*(q[k] for k in ('world','pair','iterations','discrete','tolerance','cutoff')))
        for side in (1,2):
            validate_requests(q['world'],q['pair'][:,side-1],q['cache'+str(side)],*(a[k].numpy() for k in ('geom_type','geom_dataid','mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_graph')),n)
            if not all(np.all(np.isfinite(q[k+str(side)])) for k in ('position','initial','margin')):raise ValueError('GJK nonfinite geometry query')
        for key,kind,width in GJK_OUTPUTS:bind_array(a,key,getattr(outputs,key),getattr(wp,kind),(nr,4) if width==4 else (nr,),stream)
        reject_output_overlap(a,tuple(k for k,_,_ in GJK_OUTPUTS))
        self.arguments=ConvexGJKInputs(worlds=n,geoms=ng,requests=nr,**rows,**{k:a[k].ptr for k,_ in ConvexGJKInputs._fields_[5:]});self.launch=self.library.flash_mjwarp_convex_gjk_launch;self.launch.argtypes=[ct.POINTER(ConvexGJKInputs),ct.c_void_p];self.launch.restype=ct.c_int
    @property
    def identity(self):return dict(source_id=source_id('convex'),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),component='gjk-function-probe-only',production_collision_replaced=False,numerical_workspace_bytes=0,shared_bytes=0)
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex GJK launch failed: CUDA error '+str(error))


class ConvexFrozenGJK(ct.Structure):
    _fields_=[(k,ct.c_void_p) for k in ('simplex','simplex1','simplex2','distance','x1','x2','count','vertex1','vertex2')]


class ConvexCCDInputs(ct.Structure):
    _fields_=[('query',ConvexGJKInputs)]+[(k,ct.c_int) for k in ('epa_iterations','vertex_capacity','face_capacity','horizon_capacity','warn','use_frozen_gjk')]+[('frozen',ConvexFrozenGJK)]+[(k,ct.c_void_p) for k in ('vertices','projection','norm2','vertex_index','face','horizon','overflow','needs_epa_out','ncontact_out','face_index_out','nvert_out','nface_out','status_out')]


def validate_epa_capacity(requests,iterations,vertices,faces,horizon):
    if requests<0 or iterations<0 or vertices<10+2*iterations or faces<6 or horizon<1 or max(requests*vertices*3,requests*faces*3,requests*horizon)>2**31-256:raise ValueError('EPA scratch capacity/index span differs')


CCD_OUTPUTS=('needs_epa_out','ncontact_out','face_index_out','nvert_out','nface_out','status_out')


class CCDProbe:
    """GJK phase plus EPA with caller-owned per-request global scratch."""
    def __init__(self,model,data,requests,outputs,scratch,iterations,library,stream,*,warn=False,frozen=None):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        self.query=GJKProbe(model,data,requests,outputs,library,stream);self.stream=stream;self.arrays=a=dict(self.query.arrays);nr=requests.world.shape[0]
        nv,nf,nh=scratch.vertices.shape[1],scratch.face.shape[1],scratch.horizon.shape[1];validate_epa_capacity(nr,iterations,nv,nf,nh)
        for key,kind,width in (('vertices',wp.vec3,nv),('projection',wp.vec3,nf),('norm2',wp.float32,nf),('vertex_index',wp.int32,nv),('face',wp.int32,nf),('horizon',wp.int32,nh)):bind_array(a,'epa_'+key,getattr(scratch,key),kind,(nr,width),stream)
        bind_array(a,'overflow',scratch.overflow,wp.int32,(data.nworld,),stream)
        for key in CCD_OUTPUTS:bind_array(a,key,getattr(outputs,key),wp.int32,(nr,),stream)
        reject_output_overlap(a,tuple(CCD_OUTPUTS)+('epa_vertices','epa_projection','epa_norm2','epa_vertex_index','epa_face','epa_horizon','overflow')+tuple(k for k,_,_ in GJK_OUTPUTS))
        self.path,self.library=load_leaf('convex',ConvexCCDInputs,library,stream,entry='ccd')
        frozen_args=ConvexFrozenGJK()
        if frozen is not None:
            for key,kind,width in (('simplex',wp.vec3,4),('simplex1',wp.vec3,4),('simplex2',wp.vec3,4),('distance',wp.float32,1),('x1',wp.vec3,1),('x2',wp.vec3,1),('count',wp.int32,1),('vertex1',wp.int32,4),('vertex2',wp.int32,4)):bind_array(a,'frozen_'+key,getattr(frozen,key),kind,(nr,4) if width==4 else (nr,),stream)
            counts=frozen.count.numpy()
            if any((counts<2)|(counts>4)):raise ValueError('EPA frozen GJK must have2..4 vertices')
            frozen_args=ConvexFrozenGJK(**{k:a['frozen_'+k].ptr for k,_ in ConvexFrozenGJK._fields_})
            reject_output_overlap(a,tuple(CCD_OUTPUTS)+('epa_vertices','epa_projection','epa_norm2','epa_vertex_index','epa_face','epa_horizon','overflow')+tuple(k for k,_,_ in GJK_OUTPUTS))
        self.arguments=ConvexCCDInputs(use_frozen_gjk=int(frozen is not None),frozen=frozen_args,query=self.query.arguments,epa_iterations=iterations,vertex_capacity=nv,face_capacity=nf,horizon_capacity=nh,warn=int(warn),**{k:a['epa_'+k if 'epa_'+k in a else k].ptr for k,_ in ConvexCCDInputs._fields_[8:]})
        self.launch=self.library.flash_mjwarp_convex_ccd_launch;self.launch.argtypes=[ct.POINTER(ConvexCCDInputs),ct.c_void_p];self.launch.restype=ct.c_int
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex CCD probe launch failed: CUDA error '+str(error))


class ConvexManifoldInputs(ct.Structure):
    _fields_=[('query',ConvexGJKInputs),('vertex_capacity',ct.c_int),('face_capacity',ct.c_int)]+[(k,ct.c_void_p) for k in ('vertices','vertex_index','face','face_index','x1','x2','count','witness1','witness2','distance')]


def validate_box_manifold(world,pair,types,margin1,margin2,faces,face_index,vertex_index):
    import numpy as np
    nr=len(world)
    if np.any(types[pair]!=6) or np.any(margin1!=0) or np.any(margin2!=0):raise ValueError('Box manifold requires two uninflated boxes')
    if faces.ndim!=2 or vertex_index.ndim!=2 or face_index.shape!=(nr,) or faces.shape[0]!=nr or vertex_index.shape[0]!=nr or vertex_index.shape[1]%2 or np.any(face_index<0) or np.any(face_index>=faces.shape[1]):raise ValueError('Box manifold frozen face range differs')
    for i,face in enumerate(face_index):
        packed=int(faces[i,face]);vertices=[packed&1023,(packed>>10)&1023,(packed>>20)&1023]
        if any(v>=vertex_index.shape[1]//2 for v in vertices) or np.any(vertex_index[i,np.array(vertices)[:,None]*2+np.arange(2)]<0) or np.any(vertex_index[i,np.array(vertices)[:,None]*2+np.arange(2)]>7):raise ValueError('Box manifold packed feature range differs')


class ManifoldProbe:
    """Box-only manifold from borrowed immutable original EPA face storage."""
    def __init__(self,model,data,requests,query_outputs,frozen,outputs,library,stream):
        from libero.gpu.host.runtime.cuda_api import array_api
        wp = array_api(stream)
        self.query=GJKProbe(model,data,requests,query_outputs,library,stream);self.stream=stream;self.arrays=a=dict(self.query.arrays);nr=requests.world.shape[0];nv,nf=frozen.vertices.shape[1],frozen.face.shape[1]
        for key,dtype,shape in (('vertices',wp.vec3,(nr,nv)),('vertex_index',wp.int32,(nr,nv)),('face',wp.int32,(nr,nf)),('face_index',wp.int32,(nr,)),('x1',wp.vec3,(nr,)),('x2',wp.vec3,(nr,))):bind_array(a,'frozen_'+key,getattr(frozen,key),dtype,shape,stream)
        validate_box_manifold(*(a[k].numpy() for k in ('world','pair','geom_type','margin1','margin2','frozen_face','frozen_face_index','frozen_vertex_index')))
        for key,dtype,shape in (('count',wp.int32,(nr,)),('witness1',wp.vec3,(nr,4)),('witness2',wp.vec3,(nr,4)),('distance',wp.float32,(nr,4))):bind_array(a,'manifold_'+key,getattr(outputs,key),dtype,shape,stream)
        reject_output_overlap(a,('manifold_count','manifold_witness1','manifold_witness2','manifold_distance'))
        self.path,self.library=load_leaf('convex',ConvexManifoldInputs,library,stream,entry='manifold')
        self.arguments=ConvexManifoldInputs(query=self.query.arguments,vertex_capacity=nv,face_capacity=nf,**{k:a['frozen_'+k if 'frozen_'+k in a else 'manifold_'+k].ptr for k,_ in ConvexManifoldInputs._fields_[3:]})
        self.launch=self.library.flash_mjwarp_convex_manifold_launch;self.launch.argtypes=[ct.POINTER(ConvexManifoldInputs),ct.c_void_p];self.launch.restype=ct.c_int
    def enqueue(self):
        error=self.launch(ct.byref(self.arguments),self.stream.cuda_stream)
        if error:raise RuntimeError('Convex box manifold probe launch failed: CUDA error '+str(error))
