from contextlib import contextmanager, nullcontext
"""Automatic public LIBERO task provider using the current GPU implementation."""
from libero.gpu.host.runtime.core.product_paths import (
    product_build, mathdx_root, toolchain_scope, cache_root, build_project)


@contextmanager
def compatibility_scope():
    """Explicit offline authority; ordinary compatibility construction is load-only."""
    from libero.gpu.host.build.compilation import (
        compilation_scope, current_scope, MissingCompiledProduct, install_warp_guard)
    from libero.gpu.host.build.aot import configure_warp_cache
    scope = current_scope()
    # Reuse explicit authority and its event log across nested providers. Only
    # a standalone public constructor needs to install its own frozen scope.
    authority = compilation_scope(False) if scope is None else nullcontext(scope)
    with authority, build_project() as project:
        install_warp_guard()
        configure_warp_cache(project['root'])
        try:
            yield project
        except MissingCompiledProduct as error:
            if not getattr(error, 'compatibility_profile', False):
                error.args = (str(error) + '\nFor OffScreenRenderEnv, prepare the matching profile offline: '
                    'python -m libero.gpu.build --compatibility --precision fp64 '
                    '--suite <suite> --task-ids <id>',)
                error.compatibility_profile = True
            raise


def create(built):
    from libero.gpu.host.factory import prepare_cuda_physics
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    from libero.gpu.host.bindings.position_scene import install_scene_from_environment
    with compatibility_scope() as project, product_build():
        project_root = project['root']
        sdk = mathdx_root()

        @contextmanager
        def compilation_scope():
            from libero.gpu.host.build.product_cache import product_cache_scope
            with toolchain_scope(sdk), product_cache_scope(project['root'] / 'products'):
                yield

        def prepare(*args, **kwargs):
            # Replacements execute after the first constructor scope has ended.
            # Each generation owns a fresh root, including its compiler copies.
            with build_project(project_root), compatibility_scope(), product_build() as generation_root, compilation_scope():
                kwargs.setdefault('device', getattr(built, '_gpu_device', 'cuda:0'))
                bundle = prepare_cuda_physics(*args, cuda_build_root=generation_root / 'cuda',
                                              mathdx_root=sdk, **kwargs)
                bundle.build_project_root = project_root
                return bundle

        def install_scene(result):
            bundle = result.observation_fields.__self__.prepared
            profile = bundle.dataflow_profile
            if profile not in ('compatible', 'fast'):
                raise ValueError('actual physics assembly dataflow profile is absent')
            with build_project(project_root), compatibility_scope(), compilation_scope():
                install_scene_from_environment(result, bundle.disk_build_root / ('scene-' + str(id(result))),
                                               dataflow_profile=profile)

        def assemble_replacement(updated, bundle, *, observation_continuation=None):
            from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import assemble_prepared
            with build_project(project_root), compatibility_scope(), compilation_scope():
                fresh = assemble_prepared(updated, bundle)
            try:
                if observation_continuation is not None:
                    shell = updated.domain_shell
                    shell.sim = fresh.sim
                    for robot in shell.robots: robot.sim = fresh.sim
                    fresh.observations.host_environments[fresh.world] = shell
                    fresh.observations.restore_continuation(observation_continuation, stream=fresh.stream)
                install_scene(fresh)
                return fresh
            except BaseException:
                fresh.release()
                raise

        with compilation_scope():
            result = assemble(built, physics_factory=prepare,
                              factory_entry='libero.gpu.host.adapters.environment.product_factory:create')
        try:
            install_scene(result)
            from .native_update_runtime import bind_native_update_runtime
            binding = result.observation_fields.__self__
            bind_native_update_runtime(result, binding.prepared, prepare=prepare,
                                       assemble=assemble_replacement)
        except BaseException:
            result.release()
            raise
        return result


def rebuild_components(built, previous):
    from libero.gpu.host.runtime.step.shape_schema import RuntimeShape
    from libero.gpu.host.runtime.step.recovery_final_event_environment_factory import (
        default_integration_state, actual_decoder_record)
    RuntimeShape.from_native_metadata(built.metadata)
    default_integration_state(built.native_model)
    actual_decoder_record(built)
    previous.release()
    return create(built)


create.rebuild_components = rebuild_components
create.construction_scope = compatibility_scope
