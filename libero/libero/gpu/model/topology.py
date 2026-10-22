"""Cold, qualified integer topology; no simulation arithmetic or implicit upload."""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import numpy as np

FIELDS = ('body_parentid', 'body_dofadr', 'body_dofnum', 'dof_parentid',
          'dof_Madr', 'jnt_type', 'jnt_dofadr', 'jnt_qposadr',
          'body_sameframe', 'geom_sameframe', 'site_sameframe',
          'pair_geom1', 'pair_geom2', 'dof_simple')
ALIASES = {'body_level_offsets':'level_offsets', 'body_level_ids':'level_bodies',
           'body_children_offsets':'children_offsets', 'body_children_ids':'children_indices',
           'joints_by_type_offsets':'joint_type_offsets', 'joint_ids':'joint_type_indices'}
_CPU_CACHE = OrderedDict()
_DEVICE_CACHE = OrderedDict()

def _get(model, name):
    return model[name] if isinstance(model, dict) else getattr(model, name)

def _source(model):
    result = {}
    for field in FIELDS:
        try:
            a = np.asarray(_get(model, field))
        except (KeyError, AttributeError):
            continue
        if a.dtype.kind not in 'iu' or np.any(a < -2147483648) or np.any(a > 2147483647):
            raise ValueError('topology requires int32-representable integers: '+field)
        result[field] = np.array(a, dtype=np.int32).reshape(-1)
    return result

def _digest(source, ancestor=None, pairs=None):
    h = sha256(b'libero-static-topology-v1')
    for key, a in sorted(source.items()):
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    for key, a in (('qualified_ancestor',ancestor),('requested_body_pairs',pairs)):
        if a is not None:
            h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def _immutable(values, dtype=np.int32):
    # bytes-backed storage cannot be made writeable by setflags(write=True).
    a = np.asarray(values, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)

def contiguous_subtree_sizes(parent):
    """Qualify contiguous descendant intervals, without assuming DFS numbering.

    A preceding parent alone is insufficient: an unrelated body can sit inside
    a descendant's ID range. Count and maximum ID together prove no holes.
    Return None for such trees so callers retain their generic recurrence.
    """
    parent = _ints(parent, 'body_parentid').reshape(-1)
    count = len(parent)
    if not count or parent[0] != 0 or any(not 0 <= parent[b] < b for b in range(1, count)):
        raise ValueError('body parents must precede children and root at zero')
    sizes = np.ones(count, np.int32)
    maximum = np.arange(count, dtype=np.int32)
    for body in range(count - 1, 0, -1):
        p = parent[body]
        sizes[p] += sizes[body]
        maximum[p] = max(maximum[p], maximum[body])
    if np.any(maximum != np.arange(count) + sizes - 1):
        return None
    return _immutable(sizes)

def _csr(rows):
    offsets = [0]; indices = []
    for row in rows:
        indices.extend(row); offsets.append(len(indices))
    if len(indices) > 2147483647:
        raise ValueError('CSR exceeds int32 capacity')
    return _immutable(offsets), _immutable(indices)

def _body_rows(parent):
    """Linear buckets; preserve descending child and ascending level ordinals."""
    nb = len(parent)
    children = [[] for _ in range(nb)]
    depth = np.zeros(nb, np.int32)
    for b in range(nb - 1, 0, -1):
        children[int(parent[b])].append(b)
    for b in range(1, nb):
        depth[b] = depth[int(parent[b])] + 1
    levels = [[] for _ in range(int(depth.max()))]
    for b in range(1, nb):
        levels[int(depth[b]) - 1].append(b)
    return children, levels

@dataclass(frozen=True)
class StaticTopology:
    arrays: object
    proof: object
    source_sha256: str

    def validate_source(self, model, ancestor_mask=None, body_pairs=None):
        ancestor = None if ancestor_mask is None else np.asarray(ancestor_mask,dtype=np.int32)
        pairs = None if body_pairs is None else np.asarray(body_pairs,dtype=np.int32).reshape(-1,2)
        return _digest(_source(model),ancestor,pairs) == self.source_sha256

@dataclass(frozen=True)
class DeviceTopology:
    arrays: object
    host: StaticTopology
    device: str

    @property
    def proof(self):
        return self.host.proof

    @property
    def source_sha256(self):
        return self.host.source_sha256

def build_static_topology(model, *, ancestor_mask=None, body_pairs=None):
    """Qualify original tree descriptors, then derive compact ordered indices.

    No O(nbody**2) body-pair table is built by default. Explicit requested pairs
    receive sparse CSR rows in their supplied order, capped at one million cells.
    Original model descriptors remain authoritative and are not re-uploaded here.
    """
    s = _source(model)
    ancestor=None if ancestor_mask is None else _ints(ancestor_mask,'ancestor_mask').reshape(np.asarray(ancestor_mask).shape)
    pairs=None if body_pairs is None else _ints(body_pairs,'body_pairs').reshape(-1,2)
    source_sha=_digest(s,ancestor,pairs)
    if source_sha in _CPU_CACHE:
        _CPU_CACHE.move_to_end(source_sha); return _CPU_CACHE[source_sha]
    parent=s['body_parentid']; nb=len(parent); dp=s['dof_parentid']; nv=len(dp)
    madr=s['dof_Madr']; jt=s['jnt_type']; nj=len(jt)
    if not nb or parent[0]!=0 or len(madr)!=nv:
        raise ValueError('body root / packed address shape')
    if len(s['body_dofadr'])!=nb or len(s['body_dofnum'])!=nb:
        raise ValueError('body DOF shape')
    if len(s['jnt_dofadr'])!=nj or len(s['jnt_qposadr'])!=nj or np.any((jt<0)|(jt>3)):
        raise ValueError('joint descriptors')
    if any(not 0<=int(parent[b])<b for b in range(1,nb)):
        raise ValueError('body parents must precede children')
    if any(not -1<=int(dp[d])<d for d in range(nv)) or np.any(madr<0):
        raise ValueError('DOF parent ordering / packed addresses')
    chains=[]
    for i in range(nv):
        row=[]; j=i
        while j>=0:
            row.append(j); j=int(dp[j])
        chains.append(row)
        end=int(madr[i])+len(row)
        if i+1<nv and end>int(madr[i+1]):
            raise ValueError('packed row overlaps next row')
    dense=np.zeros((nb,nv),np.int32)
    for b in range(1,nb):
        body=b
        while body and s['body_dofnum'][body]==0:
            body=int(parent[body])
        if body:
            first=int(s['body_dofadr'][body]); count=int(s['body_dofnum'][body])
            if first<0 or count<0 or first+count>nv:
                raise ValueError('body DOF range')
            dense[b,chains[first+count-1]]=1
    if ancestor is not None and (ancestor.shape!=dense.shape or not np.array_equal(ancestor,dense)):
        raise ValueError('ancestor owner does not match qualified DOF chains')
    if pairs is not None and (np.any(pairs<0) or np.any(pairs>=nb)):
        raise ValueError('requested body pair range')
    out={}
    out['body_dof_offsets'],out['body_dof_indices']=_csr([np.flatnonzero(row) for row in dense])
    children,levels=_body_rows(parent)
    out['children_offsets'],out['children_indices']=_csr(children)
    out['level_offsets'],out['level_bodies']=_csr(levels)
    out['joint_type_offsets'],out['joint_type_indices']=_csr([np.flatnonzero(jt==t) for t in range(4)])
    quat=[]; index=0
    for t in jt:
        quat.append(index if t<2 else -1); index+=int(t<2)
    out['joint_quat_index']=_immutable(quat)
    out['dof_chain_offsets'],out['dof_chain_indices']=_csr(chains)
    out['dof_chain_packed']=_immutable([int(madr[i])+k for i,row in enumerate(chains) for k in range(len(row))])
    for prefix in ('body','geom','site'):
        name=prefix+'_sameframe'
        if name in s:
            same=s[name]
            rows=[np.flatnonzero(same==0),np.flatnonzero(same==1),np.flatnonzero((same!=0)&(same!=1))]
            out[prefix+'_frame_offsets'],out[prefix+'_frame_indices']=_csr(rows)
    if pairs is not None:
        if len(pairs)*nv>1000000:
            raise ValueError('requested body pair metadata worst-case capacity exceeded')
        rows=[]; members=[]
        for b1,b2 in pairs:
            ids=np.flatnonzero(dense[b1]|dense[b2]); rows.append(ids)
            members.extend((dense[b1,ids]+2*dense[b2,ids]).tolist())
        if sum(map(len,rows))>1000000:
            raise ValueError('requested body pair metadata capacity exceeded')
        out['body_pair_keys']=_immutable(pairs)
        out['body_pair_offsets'],out['body_pair_indices']=_csr(rows)
        out['body_pair_membership']=_immutable(members)
    # Sparse normalized geom-pair keys avoid an ngeom**2 table. Each row keeps
    # every original explicit-pair ordinal, including duplicate/reversed pairs.
    if 'pair_geom1' in s and 'pair_geom2' in s:
        p1=s['pair_geom1']; p2=s['pair_geom2']
        if len(p1)!=len(p2) or np.any(p1<0) or np.any(p2<0):
            raise ValueError('explicit geom pair range')
        grouped={}
        for i,(a,b) in enumerate(zip(p1,p2)):
            key=(min(int(a),int(b)),max(int(a),int(b)))
            grouped.setdefault(key,[]).append(i)
        keys=sorted(grouped)
        out['pair_keys']=_immutable(np.asarray(keys,dtype=np.int32).reshape(-1,2))
        out['pair_key_offsets'],out['pair_key_indices']=_csr([grouped[key] for key in keys])
    def disjoint(address,width):
        cells=[]
        for i,t in enumerate(jt): cells.extend(range(int(address[i]),int(address[i])+width[int(t)]))
        return len(set(cells))==len(cells) and all(x>=0 for x in cells)
    proof=dict(schema=1,nbody=nb,nv=nv,njnt=nj,nlevels=len(levels),
               joint_dof_disjoint=disjoint(s['jnt_dofadr'],(6,3,1,1)),
               joint_qpos_disjoint=disjoint(s['jnt_qposadr'],(7,4,1,1)),
               joint_quat_count=index,source_fields=tuple(sorted(s)),
               canonical_bytes=sum(a.nbytes for a in out.values()),
               body_pairs_sparse=pairs is not None,
               children_order='descending original body id',
               active_dof_order='ascending original DOF id',
               topology_lifetime='cold snapshot; owner must enforce source immutability')
    for alias,canonical in ALIASES.items(): out[alias]=out[canonical]
    result=StaticTopology(MappingProxyType(out),MappingProxyType(proof),source_sha)
    _CPU_CACHE[source_sha]=result
    while len(_CPU_CACHE)>16: _CPU_CACHE.popitem(last=False)
    return result

def prepare_device_topology(model, device='cuda:0', *, ancestor_mask=None, body_pairs=None):
    """Explicit cold upload, cached by qualified source digest and CUDA device.

    Callers retain this owner and its arrays for the kernel lifetime. Aliases
    reference the same uploaded object. No runtime flags are implicitly trusted.
    Unsupported topology raises ValueError so bindings can choose scalar fallback.
    """
    host=build_static_topology(model,ancestor_mask=ancestor_mask,body_pairs=body_pairs)
    return upload_topology(host,device)

def upload_topology(host, device='cuda:0'):
    """Upload a qualified CPU descriptor; aliases share allocation ownership."""
    import warp as wp
    device=wp.get_device(device); key=(host.source_sha256,str(device))
    if key in _DEVICE_CACHE:
        _DEVICE_CACHE.move_to_end(key); return _DEVICE_CACHE[key]
    arrays={name:wp.array(value,dtype=wp.uint32 if value.dtype==np.uint32 else wp.int32,device=device) for name,value in host.arrays.items() if name not in ALIASES}
    for alias,canonical in ALIASES.items():
        if canonical in arrays: arrays[alias]=arrays[canonical]
    result=DeviceTopology(MappingProxyType(arrays),host,str(device))
    _DEVICE_CACHE[key]=result
    while len(_DEVICE_CACHE)>16: _DEVICE_CACHE.popitem(last=False)
    return result

def _part_digest(kind, inputs):
    source={'descriptor_kind':np.frombuffer(kind.encode(),np.uint8)}
    source.update({k:np.ascontiguousarray(v) for k,v in inputs.items()})
    return _digest(source)

def _cached_part(kind, inputs):
    digest=_part_digest(kind,inputs)
    if digest in _CPU_CACHE:
        _CPU_CACHE.move_to_end(digest)
        return _CPU_CACHE[digest]
    return None

def _part(kind, inputs, arrays, proof):
    digest=_part_digest(kind,inputs)
    if digest in _CPU_CACHE:
        _CPU_CACHE.move_to_end(digest); return _CPU_CACHE[digest]
    for alias,canonical in ALIASES.items():
        if canonical in arrays: arrays[alias]=arrays[canonical]
    proof=dict(proof,schema=1,kind=kind,source_fields=tuple(inputs),
               canonical_bytes=sum(v.nbytes for k,v in arrays.items() if k not in ALIASES),
               topology_lifetime='cold snapshot; caller must qualify source owner and immutability')
    result=StaticTopology(MappingProxyType(arrays),MappingProxyType(proof),digest)
    _CPU_CACHE[digest]=result
    while len(_CPU_CACHE)>16: _CPU_CACHE.popitem(last=False)
    return result

def _ints(value, name):
    a=np.asarray(value)
    if a.dtype.kind not in 'iu' or np.any(a < -2147483648) or np.any(a > 2147483647):
        raise ValueError('invalid int32 source: '+name)
    return np.asarray(a,dtype=np.int32).reshape(-1)

def build_body_topology(parent):
    parent=_ints(parent,'body_parentid'); nb=len(parent)
    if not nb or parent[0]!=0 or any(not 0<=int(parent[b])<b for b in range(1,nb)):
        raise ValueError('body parent ordering')
    cached=_cached_part('body',{'body_parentid':parent})
    if cached is not None: return cached
    children,levels=_body_rows(parent)
    lo,li=_csr(levels)
    co,ci=_csr(children)
    return _part('body',{'body_parentid':parent},dict(level_offsets=lo,level_bodies=li,
                 children_offsets=co,children_indices=ci),dict(nbody=nb,nlevels=len(levels)))

def build_ancestor_topology(ancestor_mask):
    original=np.asarray(ancestor_mask)
    if original.ndim!=2: raise ValueError('ancestor mask shape')
    mask=_ints(original,'ancestor_mask').reshape(original.shape)
    if np.any((mask!=0)&(mask!=1)): raise ValueError('ancestor mask must be binary')
    cached=_cached_part('ancestor',{'ancestor_mask':mask})
    if cached is not None: return cached
    offsets,indices=_csr([np.flatnonzero(row) for row in mask])
    return _part('ancestor',{'ancestor_mask':mask},dict(body_dof_offsets=offsets,body_dof_indices=indices),
                 dict(nbody=len(mask),nv=mask.shape[1],source_owner_qualification_required=True))

def build_joint_topology(jnt_type, qposadr, dofadr, *, nq=None, nv=None):
    typ=_ints(jnt_type,'jnt_type'); qa=_ints(qposadr,'jnt_qposadr'); da=_ints(dofadr,'jnt_dofadr')
    if len(qa)!=len(typ) or len(da)!=len(typ) or np.any((typ<0)|(typ>3)):
        raise ValueError('joint descriptor shape/type')
    if nq is not None and nq<0 or nv is not None and nv<0: raise ValueError('negative joint dimensions')
    source={'jnt_type':typ,'jnt_qposadr':qa,'jnt_dofadr':da,
            'dimensions':np.asarray([nq if nq is not None else -1,nv if nv is not None else -1],np.int32)}
    cached=_cached_part('joint',source)
    if cached is not None: return cached
    def eligible(address,width,limit):
        cells=[]
        for i,t in enumerate(typ):
            start=int(address[i]); end=start+width[int(t)]
            if start<0 or (limit is not None and end>limit): return False
            cells.extend(range(start,end))
        return len(cells)==len(set(cells))
    offsets,indices=_csr([np.flatnonzero(typ==t) for t in range(4)])
    quat=[]; number=0
    for t in typ:
        quat.append(number if t<2 else -1);number+=int(t<2)
    return _part('joint',source,
                dict(joint_type_offsets=offsets,joint_type_indices=indices,joint_quat_index=_immutable(quat)),
                dict(njnt=len(typ),joint_quat_count=number,
                     joint_dof_disjoint=eligible(da,(6,3,1,1),nv),
                     joint_qpos_disjoint=eligible(qa,(7,4,1,1),nq)))

def build_packed_topology(dof_parentid, dof_Madr, *, simple=None):
    parent=_ints(dof_parentid,'dof_parentid');madr=_ints(dof_Madr,'dof_Madr');nv=len(parent)
    if len(madr)!=nv or np.any(madr<0) or any(not -1<=int(parent[d])<d for d in range(nv)):
        raise ValueError('packed topology ordering/address')
    source=dict(dof_parentid=parent,dof_Madr=madr)
    if simple is not None:
        simple=_ints(simple,'dof_simple')
        if len(simple)!=nv or np.any((simple!=0)&(simple!=1)): raise ValueError('simple mask')
        source['dof_simple']=simple
    cached=_cached_part('packed',source)
    if cached is not None: return cached
    rows=[]
    for i in range(nv):
        row=[];j=i
        while j>=0: row.append(j);j=int(parent[j])
        rows.append(row)
        if i+1<nv and int(madr[i])+len(row)>int(madr[i+1]): raise ValueError('packed row overlap')
    offsets,indices=_csr(rows)
    arrays=dict(dof_chain_offsets=offsets,dof_chain_indices=indices,
                dof_chain_packed=_immutable([int(madr[i])+k for i,row in enumerate(rows) for k in range(len(row))]))
    if simple is not None:
        arrays['dof_simple_offsets'],arrays['dof_simple_indices']=_csr([np.flatnonzero(simple==0),np.flatnonzero(simple==1)])
    return _part('packed',source,arrays,dict(nv=nv))

def build_frame_topology(sameframe):
    same=_ints(sameframe,'sameframe')
    cached=_cached_part('frame',{'sameframe':same})
    if cached is not None: return cached
    offsets,indices=_csr([np.flatnonzero(same==0),np.flatnonzero(same==1),np.flatnonzero((same!=0)&(same!=1))])
    return _part('frame',{'sameframe':same},dict(frame_offsets=offsets,frame_indices=indices),dict(nobjects=len(same)))

def prepare_body_topology(parent, device='cuda:0'):
    return upload_topology(build_body_topology(parent),device)

def prepare_ancestor_topology(ancestor_mask, device='cuda:0'):
    return upload_topology(build_ancestor_topology(ancestor_mask),device)

def prepare_joint_topology(jnt_type, qposadr, dofadr, device='cuda:0', *, nq=None, nv=None):
    return upload_topology(build_joint_topology(jnt_type,qposadr,dofadr,nq=nq,nv=nv),device)

def prepare_packed_topology(dof_parentid, dof_Madr, device='cuda:0', *, simple=None):
    return upload_topology(build_packed_topology(dof_parentid,dof_Madr,simple=simple),device)

def prepare_frame_topology(sameframe, device='cuda:0'):
    return upload_topology(build_frame_topology(sameframe),device)

def build_pair_suppression(pair_geom1, pair_geom2, ngeom, *, max_cells=1000000):
    """Bounded optional ordered-key lookup, preserving duplicate pair ordinals."""
    if ngeom<0 or ngeom*ngeom>max_cells: return None
    a=_ints(pair_geom1,'pair_geom1');b=_ints(pair_geom2,'pair_geom2')
    if len(a)!=len(b) or np.any(a<0) or np.any(b<0) or np.any(a>=ngeom) or np.any(b>=ngeom): return None
    source={'pair_geom1':a,'pair_geom2':b,'ngeom':np.asarray([ngeom],np.int32)}
    cached=_cached_part('pair_suppression',source)
    if cached is not None: return cached
    rows=[[] for _ in range(ngeom*ngeom)]
    for i,(x,y) in enumerate(zip(a,b)):
        rows[int(x)*ngeom+int(y)].append(i)
        if x!=y: rows[int(y)*ngeom+int(x)].append(i)
    offsets,indices=_csr(rows)
    return _part('pair_suppression',source,
                 dict(pair_key_offsets=offsets,pair_key_indices=indices),dict(ngeom=ngeom,npair=len(a),max_cells=max_cells))

def build_geom_pair_routes(geom_type, contype, conaffinity, priority, condim, present, *, max_cells=1000000):
    """Optional C6 bit-packed routing; scalar runtime floats remain authoritative."""
    typ=_ints(geom_type,'geom_type');ng=len(typ)
    if ng*ng>max_cells: return None
    ct=_ints(contype,'geom_contype');ca=_ints(conaffinity,'geom_conaffinity')
    pr=_ints(priority,'geom_priority');dim=_ints(condim,'geom_condim');present=_ints(present,'present')
    if any(len(a)!=ng for a in (ct,ca,pr,dim)) or len(present)!=64 or np.any((typ<0)|(typ>7)) or np.any((dim<0)|(dim>65535)): return None
    source=dict(geom_type=typ,geom_contype=ct,geom_conaffinity=ca,geom_priority=pr,geom_condim=dim,present=present)
    cached=_cached_part('geom_pair_routes',source)
    if cached is not None: return cached
    route=np.zeros(ng*ng,np.uint32)
    for g1 in range(ng):
        for g2 in range(ng):
            swap=int(typ[g1]>typ[g2]);a,b=(g2,g1) if swap else (g1,g2)
            d=int(dim[a] if pr[a]>pr[b] else dim[b] if pr[a]<pr[b] else max(dim[a],dim[b]))
            affine=bool((int(ct[a])&int(ca[b])) or (int(ct[b])&int(ca[a])))
            route[g1*ng+g2]=int(typ[a])|(int(typ[b])<<3)|(swap<<6)|(int(present[int(typ[a])*8+int(typ[b])]!=0)<<7)|(int(affine)<<8)|(d<<16)
    return _part('geom_pair_routes',source,dict(geom_pair_route=_immutable(route,np.uint32)),dict(ngeom=ng,max_cells=max_cells))


def build_packed_mass_plan(madr, parent, simple, nv, nm):
    arrays=[np.asarray(a) for a in (madr,parent,simple)]
    if any(a.dtype!=np.int32 or a.shape!=(nv,) for a in arrays):
        raise ValueError('packed mass topology must be int32 NV vectors')
    madr,parent,simple=arrays
    if any(int(parent[i])<-1 or int(parent[i])>=i for i in range(nv)):
        raise ValueError('parent must be a preceding DOF or -1')
    if any(int(madr[i])<0 or int(madr[i])>=nm for i in range(nv)):
        raise ValueError('diagonal packed address is out of bounds')
    events=[[] for _ in range(nv)]
    for i in range(nv):
        adr=int(madr[i])
        events[i].append((adr,i,int(simple[i]!=0)))
        if simple[i]:
            continue
        j=int(parent[i]);adr+=1
        while j>=0:
            if adr>=nm:
                raise ValueError('ancestor packed address is out of bounds')
            events[i].append((adr,j,0))
            events[j].append((adr,i,0))
            j=int(parent[j]);adr+=1
    offsets=np.zeros(nv+1,np.int32)
    for row in range(nv):
        offsets[row+1]=offsets[row]+len(events[row])
    cells=[event for row in events for event in row]
    cells=np.asarray(cells,np.int32).reshape(-1,3)
    return (offsets,*[np.ascontiguousarray(cells[:,i]) for i in range(3)])
