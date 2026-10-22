#pragma once
// All callers enter collectively. Start/continuation initialization has one
// owner: the motion kernel, or preparation when the component path is selected.
__device__ inline bool staged_enter(const StagedSubstepArguments &s,
    const SubstepArguments &args, int w, SubstepEntry entry, int phase) {
  const int first = entry == SubstepEntry::Start ? 0 :
                    entry == SubstepEntry::ControllerPosition ? 1 : 2;
  if (phase == first) {
    if (threadIdx.x == 0) {
      s.phase[w] = entry == SubstepEntry::Start || args.continuation[w] == int(entry) ? phase : -1;
      s.smooth_reuse_status[w] = -1;
      if (entry == SubstepEntry::Start) {
        args.continuation[w] = int(SubstepEntry::Complete);
        args.forward_workspace.metadata[size_t(w) * ForwardCacheWords + ForwardCacheValid] = 0;
      }
    }
    __syncthreads();
  }
  return s.phase[w] == phase;
}
