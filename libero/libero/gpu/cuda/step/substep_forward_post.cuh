#pragma once
// Post observes the Newton-private qacc view, including rejected candidates.
template<class Layout = flash_libero::NewtonLayout>
__device__ __forceinline__ void publish_forward(const SubstepArguments &args,
    int world, unsigned char *shared, bool post_outputs
#if defined(FLASH_SUBSTEP_PROFILE)
    , substep_profile::Stages &step_profile_stages
#endif
    , const int *subtree_sizes = nullptr
) {
  const auto &request = *args.newton.parameters;
  const bool post_evaluated = physics_post::run(request.post, world,
                    flash_libero::shared_at<real_t>(shared, Layout::qacc),
                    // Only qacc survives the completed solve. Post may reuse the entire
                    // following arena, including the dead search/Hessian vectors.
                    flash_libero::shared_at<real_t>(shared,
                        Layout::qacc + flash_libero::shared_vector_bytes), post_outputs, subtree_sizes);
  __syncthreads();
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(ForwardPublish));
  if (threadIdx.x == 0) {
    const auto &in = args.forward_publish;
    // Intermediate completion proves current dynamics, not current Post data.
    // Post.valid stays false until a real force/sensor publication succeeds.
    const int *completed = post_evaluated ? in.nvalid : request.post.valid;
    forward_cuda::publish(world, in.active, completed, in.nerror, in.first, in.projection, in.source,
                          in.state, in.valid, in.written, in.error);
  }
  __syncthreads();
}
