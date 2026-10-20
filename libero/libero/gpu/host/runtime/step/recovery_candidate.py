"""Diagnostic classification of an actual same-call acceleration candidate.

This module does not publish bad qacc, clear any error, or reset native data.
A solver-owner producer must supply candidate bytes AND a source-qualified
completion event. Work qacc, err_code alone, or a held canonical buffer is not
that producer. Intermediate arithmetic failures remain unresolved.
"""
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})

@wp.kernel(enable_backward=False)
def classify_acceleration_candidate(
        candidate:wp.array(dtype=scalar.wp_real,ndim=2), enabled:wp.array(dtype=wp.int32),
        upstream_error:wp.array(dtype=wp.int32), publication_error:wp.array(dtype=wp.int32),
        attempted:wp.array(dtype=wp.int64), planned:wp.array(dtype=wp.int64),
        candidate_attempt:wp.array(dtype=wp.int64), candidate_completed:wp.array(dtype=wp.int32),
        candidate_origin:wp.array(dtype=wp.int32), source_stage:wp.array(dtype=wp.int32),
        classification:wp.array(dtype=wp.int32), bad_index:wp.array(dtype=wp.int32), nv:int):
    w=wp.tid()
    classification[w]=0
    bad_index[w]=-1
    if enabled[w]!=1:
        return
    # -1 means source/entry invalid, never a native warning recovery licence.
    if upstream_error[w]!=0 or publication_error[w]!=0:
        classification[w]=-1
        return
    # -2 is an absent/stale/intermediate candidate, not completed current QA.
    if attempted[w]!=planned[w] or candidate_attempt[w]!=attempted[w]:
        classification[w]=-2
        return
    # origin=1 is the explicit same-call native final-acceleration boundary.
    # This has to be produced by an audited hook, never inferred from N error.
    if candidate_completed[w]!=1 or candidate_origin[w]!=1 or source_stage[w]<=0:
        classification[w]=-2
        return
    classification[w]=1
    for j in range(nv):
        value=candidate[w,j]
        if value!=value or value>scalar.wp_real(1.0e10) or value<scalar.wp_real(-1.0e10):
            classification[w]=2
            bad_index[w]=j
            break
