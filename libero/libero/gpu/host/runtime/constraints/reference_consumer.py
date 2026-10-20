"""Opt-in complete reference function with immutable compiled support hints.

Hints are mathematical hints, never authentication of mutable state. The
original reference gate/commit stay byte-for-byte source-derived; current J is
checked by the qualified omitted-column guard on every group invocation. Any
count change uses the complete original dense reference. No production default
selects this module. Imports are CPU-only; caller creates NativePlan explicitly.
"""
import libero.gpu.precision as scalar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ReferenceHint:
    nv: int
    worlds: tuple
    byte_budget: int = 60000

    @classmethod
    def create(cls, nv, worlds, byte_budget=60000):
        if type(nv) is not int or nv <= 0 or type(byte_budget) is not int or byte_budget < 0:
            raise ValueError('Positive nv and nonnegative explicit budget required')
        frozen=[]
        for count, groups in worlds:
            if type(count) is not int or count < 0:
                raise ValueError('Nonnegative owned prefix required')
            packed=[]; allrows=[]
            for row_ids,columns in groups:
                rr=np.asarray(row_ids);cc=np.asarray(columns)
                if rr.dtype!=np.int32 or cc.dtype!=np.int32 or rr.ndim!=1 or cc.ndim!=1:
                    raise ValueError('Exact one-dimensional int32 descriptors required')
                if not len(rr) or not len(cc) or np.any(rr<0) or np.any(rr>=count) or np.any(np.diff(rr)<=0):
                    raise ValueError('Positive ordered owned rows required')
                if np.any(cc<0) or np.any(cc>=nv) or np.any(np.diff(cc)<=0):
                    raise ValueError('Positive ascending support within nv required')
                packed.append((tuple(map(int,rr)),tuple(map(int,cc))))
                allrows.extend(map(int,rr))
            if sorted(allrows)!=list(range(count)):
                raise ValueError('Hint groups must partition the complete owned prefix exactly once')
            frozen.append((count,tuple(packed)))
        if not frozen:
            raise ValueError('Explicit world capacity required')
        result=cls(nv,tuple(frozen),byte_budget)
        if result.storage_bytes()>byte_budget:
            raise ValueError('Compiled constant hint budget exceeded; use original dense')
        return result

    def arrays(self):
        # Deduplicate complete arrays by type+integer contents across worlds.
        values={}
        for _,groups in self.worlds:
            for rr,cc in groups:
                for seq in (rr,cc,tuple(int(k in cc) for k in range(self.nv))):
                    key=hashlib.sha256(np.asarray(seq,np.int32).tobytes()).hexdigest()
                    old=values.setdefault(key,seq)
                    if old!=seq:raise ValueError('Ambiguous constant array digest')
        full=tuple(range(self.nv));values[hashlib.sha256(np.asarray(full,np.int32).tobytes()).hexdigest()]=full
        return values

    def storage_bytes(self):
        return sum(len(x)*4 for x in self.arrays().values())

    def description(self):
        value={'nv':self.nv,'world_capacity':len(self.worlds),'worlds':self.worlds,
               'constant_bytes':self.storage_bytes(),'byte_budget':self.byte_budget}
        value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return value


def extract_function(source, declaration):
    """Extract exact project-written body, changing no numerical/error tree."""
    begin=source.index(declaration);opening=source.index('{',begin)
    depth=1;end=opening+1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}');end+=1
    return source[begin:end]


GATE_NAMES=('active','status','committed','counts','qvel','epoch','expected','ready','rows','nv')
GATE_TYPES=('const int32_t*','int32_t*','int32_t*','const int32_t*','const real_t*','const int64_t*','const int64_t*','int32_t*','int32_t','int32_t')
REF_NAMES=('committed','counts','efc_J','kbip','efc_pos','efc_margin','qvel','qvel_epoch','vel','aref','ref_epoch','rows','nv')
REF_TYPES=('const int32_t*','const int32_t*',*('const real_t*',)*5,'const int64_t*','real_t*','real_t*','const int64_t*','int32_t','int32_t')
COMMIT_NAMES=('ready','status','committed','counts','sv','sa','epoch','pv','pa','ref_epoch','rows')
COMMIT_TYPES=('int32_t*','int32_t*','int32_t*','const int32_t*','const real_t*','const real_t*','const int64_t*','real_t*','real_t*','int64_t*','int32_t')


def signature(macro,name,names,types):
    return '#define '+macro+' '+name+'(int w, '+', '.join(t+' '+n for n,t in zip(names,types))+')\n'


def complete_reference_source(gpu_root, hint=None):
    root=Path(gpu_root)
    original=(root/'cuda/constraints/constraints.cuh').read_text()
    dense=(root/'cuda/constraints/constraints_reference.cuh').read_text()
    definitions=''.join((signature('CF_S_reference_gate','reference_gate',GATE_NAMES,GATE_TYPES),
                        signature('CF_S_reference','reference',REF_NAMES,REF_TYPES),
                        signature('CF_S_commit_reference','commit_reference',COMMIT_NAMES,COMMIT_TYPES)))
    # Gate and commit come directly from the production handwritten source.
    bodies='namespace cf_s {\n'+extract_function(original,'__device__ __forceinline__ bool s1_block_any(')+'\n'
    bodies+=extract_function(original,'__device__ void CF_S_reference_gate')+'\n'
    bodies+=extract_function(original,'__device__ void CF_S_commit_reference')+'\n}\n'
    if hint is None:return definitions+bodies+dense
    # Validate detached descriptor again: a subclass/forged dataclass cannot
    # authorize invalid partition or unbounded capacity by editing metadata.
    if type(hint) is not ReferenceHint:raise ValueError('Exact immutable ReferenceHint required')
    detached=[(n,[(np.asarray(rr,np.int32),np.asarray(cc,np.int32)) for rr,cc in groups]) for n,groups in hint.worlds]
    exact=ReferenceHint.create(hint.nv, detached,hint.byte_budget)
    if exact != hint:raise ValueError('Exact immutable ReferenceHint required')
    compact=(root/'cuda/constraints/constraints_reference_compact.cuh').read_text().replace('namespace cf_s {','namespace cf_s_grouped {')
    definitions+=signature('CF_S_reference_dense_fallback','reference_dense_fallback',REF_NAMES,REF_TYPES)
    dense=dense.replace('CF_S_reference {','CF_S_reference_dense_fallback {')
    arrays=hint.arrays()
    name=lambda seq:'hint_'+hashlib.sha256(np.asarray(seq,np.int32).tobytes()).hexdigest()[:24]
    constants='namespace cf_s_grouped {\n'
    for seq in arrays.values():
        constants+='__device__ __constant__ int '+name(seq)+'['+str(len(seq))+']={'+','.join(map(str,seq))+'};\n'
    constants+='}\n'
    call='w, '+', '.join(REF_NAMES)
    dispatch='namespace cf_s { __device__ void CF_S_reference {\nif(committed[w]!=1)return;\n'
    dispatch+='if(nv!='+str(hint.nv)+'){reference_dense_fallback('+call+');return;}\n'
    for w,(count,groups) in enumerate(hint.worlds):
        dispatch+='if(w=='+str(w)+'&&counts[w*5+3]=='+str(count)+'){\n'
        for rr,cc in groups:
            mask=tuple(int(k in cc) for k in range(hint.nv))
            dispatch+='cf_s_grouped::reference_group_guarded<'+str(len(cc))+','+str(hint.nv)+'>(w,rows,nv,'+str(len(rr))+', '
            dispatch+=', '.join('cf_s_grouped::'+name(x) for x in (rr,cc,mask,tuple(range(hint.nv))))
            dispatch+=',efc_J,qvel,kbip,efc_pos,efc_margin,vel,aref);\n'
        dispatch+='return;}\n'
    dispatch+='reference_dense_fallback('+call+');\n} }\n'
    return definitions+bodies+dense+compact+constants+dispatch


def complete_reference_stages(buffers, rows, nv):
    def args(names):return [rows if name=='rows' else nv if name=='nv' else buffers[name] for name in names]
    return [('cf_s::reference_gate',args(GATE_NAMES),'block',list(GATE_TYPES)),
            ('cf_s::reference',args(REF_NAMES),'block',list(REF_TYPES)),
            ('cf_s::commit_reference',args(COMMIT_NAMES),'block',list(COMMIT_TYPES))]
