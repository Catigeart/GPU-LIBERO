"""Untimed original-against-original trajectory diagnostics.

The strict numerical screens remain unchanged. A second original rollout is
performed only after a failed endpoint screen, with the exact original root
state and controller buffers. This measures whether a discrepancy is also
present in the reference itself; it cannot readmit a failed numerical check.
"""
import math
import numpy as np


def variation_checks(candidate,original,repeated):
    rows=[]
    for name in candidate:
        a,b,c=(np.asarray(values[name]) for values in (candidate,original,repeated))
        if a.shape!=b.shape or b.shape!=c.shape:raise ValueError('Trajectory endpoint shapes differ: '+name)
        for world in range(a.shape[0]):
            x,y,z=(value[world].astype(float) for value in (a,b,c))
            delta=float(abs(x-y).max(initial=0.));variation=float(abs(z-y).max(initial=0.))
            strict=float(1024*np.finfo(np.float32).eps*max(1.,float(abs(y).max(initial=0.))))
            # Report a decimal order derived solely from the original repeat,
            # never from the candidate. This is an observation, not a new
            # floating-point tolerance or a replacement correctness gate.
            decade=0. if variation==0 else 10.**math.ceil(math.log10(variation))
            rows.append(dict(field=name,world=world,candidate_error=delta,
                             original_repeat_error=variation,strict_arithmetic_screen=strict,
                             original_repeat_decade=decade,
                             within_original_repeat_order=bool(np.all(np.isfinite(x)) and delta<=max(strict,decade))))
    return rows


def replay(env,reference,endpoints,folder,fields):
    import warp as wp
    from .common import actions
    from .mjwarp_complete import save
    stream=env.stream;runtime=reference.runtime;d=reference.data
    wp.synchronize_stream(stream)
    for key,value in reference.initial_roots.items():wp.copy(getattr(d,key),value,stream=stream)
    for key,value in reference.initial_controller.items():wp.copy(runtime.controller.buffers[key],value,stream=stream)
    for key in ('integration_count','state_epoch','newton_attempts','position_math'):getattr(runtime,key).zero_()
    d.overflow.zero_();reference.observer.reset(env._selection,stream=stream)
    for k in range(25):runtime.enqueue_action(actions(k,2))
    wp.synchronize_stream(stream)
    repeated={key:getattr(d,key).numpy() for key in fields}
    np.savez(folder/'original-repeat-endpoints.npz',**repeated)
    values={mode:{key:endpoints[mode+'.'+key] for key in fields} for mode in ('cuda','upstream')}
    rows=variation_checks(values['cuda'],values['upstream'],repeated)
    result=dict(role='diagnostic only; strict checks and passed flag unchanged',checks=rows,
                integrations=runtime.integration_count.numpy().tolist(),overflow=d.overflow.numpy().tolist(),
                performance=None,timing_repeated=False,initial_roots_and_controller='exact retained original buffers')
    save(folder/'original-repeat.json',result)
    # Restore the timed original endpoint for subsequent common-input probes.
    for name in fields:getattr(d,name).assign(endpoints['upstream.'+name])
    wp.synchronize_stream(stream)
    return result
