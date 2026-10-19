"""Opt-in handwritten CUDA source instrumentation; no CUDA import or execution.

Counters are world-private, invocation-local, and single-CTA only. Nested phase
times overlap parent times. They are diagnostics, never a whole-step benchmark.
"""
import re

PHASES = ('jv', 'constraint', 'transpose', 'gram', 'factor_total',
          'direction', 'line_search_total', 'stop', 'warm_select', 'commit',
          'other_prefix', 'other_loop', 'other_tail', 'main_potrf',
          'contact_potrf', 'rank_update', 'ls_evaluation', 's07_jv_quad',
          'external_prefix_total', 'post_tail', 'entry_seed', 'public_commit',
          'resident_load', 'resident_store', 'kernel_total')
STATIC_SHARED_BYTES = len(PHASES) * 3 * 8
STAGE_PHASE = {'_jacvec_sub': 'jv', '_update': 'constraint',
               '_transpose': 'transpose', 'assemble': 'gram',
               'factor_update': 'factor_total', '_direction': 'direction',
               'S07': 'line_search_total', '_stats_stop': 'stop',
               '_warm_select': 'warm_select', '_commit_step': 'commit',
               'external_prefix': 'external_prefix_total', 'external_tail': 'post_tail',
               'entry_seed': 'entry_seed', 'public_commit': 'public_commit',
               'resident_load': 'resident_load', 'resident_store': 'resident_store'}

TRACE_WORKSPACE_FIELDS = ('nefc', 'ne', 'nf', 'ncon', 'newton_world', 'choice',
                         'iteration', 'solver_iter', 'solver_exit', 'status',
                         'nactive', 'ncone', 'nupdate', 'stop_reason')


def trace_store_source(values, trace, counters, metrics, capacity, operands):
    """World-private bounded trace; counters also report dropped invocations.

    This runs only in explicitly instrumented kernels. Each world has one CTA,
    so the leader owns its counter without atomics or cross-world ordering.
    """
    phases = len(PHASES)
    copies = '\n'.join(f'm[{i}]={operand};' for i, operand in enumerate(operands))
    return f'''if(threadIdx.x==0){{
const int64_t ordinal=p->{counters}[w]++;
if(ordinal<{capacity}){{
int64_t* t=p->{trace}+(int64_t(w)*{capacity}+ordinal)*{phases*2};
for(int i=0;i<{phases*2};++i)t[i]=p->{values}[w*{phases*2}+i];
int64_t* m=p->{metrics}+(int64_t(w)*{capacity}+ordinal)*{len(operands)};
{copies}
}}
}}'''


def wrap_stage(code, key, mode):
    phase = STAGE_PHASE.get(key, 'other_' + mode)
    slot = PHASES.index(phase)
    return (f'__syncthreads();libero_profile_begin({slot});__syncthreads();\n'
            + code + f'\n__syncthreads();libero_profile_end({slot});')


def prelude():
    n = len(PHASES)
    return '''#include <stdint.h>
struct LiberoPhaseState {
    unsigned long long ns[%d], calls[%d], start[%d];
};
__device__ __forceinline__ LiberoPhaseState* libero_profile_state(){
    __shared__ LiberoPhaseState state; return &state;
}
__device__ __forceinline__ unsigned long long libero_profile_clock(){
    unsigned long long value; asm volatile("mov.u64 %%0, %%globaltimer;" : "=l"(value) :: "memory");return value;
}
__device__ __forceinline__ void libero_profile_begin(int phase){
    if(threadIdx.x==0)libero_profile_state()->start[phase]=libero_profile_clock();
}
__device__ __forceinline__ void libero_profile_end(int phase){
    if(threadIdx.x==0){auto* s=libero_profile_state();s->ns[phase]+=libero_profile_clock()-s->start[phase];++s->calls[phase];}
}
__device__ __forceinline__ void libero_profile_init(){
    if(threadIdx.x==0){auto* s=libero_profile_state();for(int i=0;i<%d;++i){s->ns[i]=0;s->calls[i]=0;s->start[i]=0;}}
    __syncthreads();
}
__device__ __forceinline__ void libero_profile_store(int64_t* out){
    __syncthreads();if(threadIdx.x==0){auto* s=libero_profile_state();for(int i=0;i<%d;++i){out[2*i]=s->ns[i];out[2*i+1]=s->calls[i];}}
}
''' % (n, n, n, n, n)


def _function(source, name):
    # Definitions only. Find braces with comment/string awareness, so diagnostic
    # instrumentation cannot terminate a function at a brace inside a comment.
    match = re.search(r'__device__\s+(?:inline\s+)?[^;{}]*?\b' + re.escape(name) + r'\s*\([^;{}]*?\)\s*\{', source)
    if match is None:
        raise ValueError('profiling function unavailable: ' + name)
    start, brace = match.start(), match.end()-1
    depth, state, i = 1, 'code', brace+1
    branches=[]
    while i < len(source) and depth:
        c, pair = source[i], source[i:i+2]
        if state=='code' and c=='#' and not source[source.rfind('\n',0,i)+1:i].strip():
            # Alternative loop headers can each open the same shared body.
            # Count each preprocessor branch from its common entry depth,
            # rather than adding mutually exclusive braces together.
            endline=source.find('\n',i)
            if endline<0:endline=len(source)
            directive=source[i+1:endline].strip().split(maxsplit=1)[0]
            if directive in ('if','ifdef','ifndef'):branches.append(depth)
            elif directive in ('else','elif') and branches:depth=branches[-1]
            elif directive=='endif' and branches:branches.pop()
            i=endline
            continue
        if state == 'line':
            if c == '\n': state='code'
        elif state == 'comment':
            if pair == '*/': state='code';i+=1
        elif state in ('"', "'"):
            if c == '\\': i+=1
            elif c == state: state='code'
        elif pair == '//': state='line';i+=1
        elif pair == '/*': state='comment';i+=1
        elif c in ('"', "'"): state=c
        elif c == '{': depth+=1
        elif c == '}': depth-=1
        i+=1
    if depth: raise ValueError('unterminated profiling function: '+name)
    return start, brace, i


def _wrap(source, name, arguments, phase, *, template=False, body_replacements=()):
    start, brace, end = _function(source, name)
    signature = source[start:brace]
    body = source[brace:end]
    for old, new in body_replacements: body=body.replace(old,new)
    renamed = signature.replace(name+'(', name+'_profile_impl(', 1)
    if renamed == signature:
        renamed = re.sub(r'\b'+name+r'\s*\(', name+'_profile_impl(', signature, count=1)
    call = name+'_profile_impl'+('<N>' if template else '')+'('+arguments+')'
    slot = PHASES.index(phase) if phase in PHASES else phase
    prefix = '\ntemplate<int N>\n' if template else '\n'
    returns_void = bool(re.search(r'\bvoid\s+'+name+r'\s*\(',signature))
    wrapper = prefix+signature+'{libero_profile_begin('+str(slot)+');'
    wrapper += call+';' if returns_void else 'auto result='+call+';'
    wrapper += 'libero_profile_end('+str(slot)+');'
    wrapper += '}' if returns_void else 'return result;}'
    return source[:start]+renamed+body+wrapper+source[end:]


def instrument_headers(source, *, use_dx, use_blas):
    # Rank block implementation bypasses the scalar wrapper for NV<=32 to avoid
    # double-counting and overwriting its nested start time.
    source = _wrap(source, 'chol_update', 'H,v,plus', 'rank_update')
    source = _wrap(source, 'chol_update_block', 'H,v,plus', 'rank_update',
                   body_replacements=(('chol_update(H,v,plus)', 'chol_update_profile_impl(H,v,plus)'),))
    if use_dx:
        source = _wrap(source, 'block_potrf', 'H,temporary,diagnostic',
                       '(N==NV?13:14)', template=True)
    if use_blas:
        source = _wrap(source, 's07_eval_point_block',
                       'a,w,rows,nefc,ne,nf,alpha,tag,niter,group_head,compact_heads,head_count,head_mu',
                       'ls_evaluation')
        source = _wrap(source, 's07_jv_blas_dx',
                       'a,w,nefc,rows,workspace,diagnostic,prepare_quad,group_head,compact_heads,tile_head_offsets,head_dim,head_mu,head_friction',
                       's07_jv_quad')
    return prelude()+source


def decode(values):
    result=[]
    for world in values:
        result.append({name:dict(nanoseconds=int(world[i][0]),calls=int(world[i][1]),
                                milliseconds=int(world[i][0])/1e6)
                       for i,name in enumerate(PHASES)})
    return result
