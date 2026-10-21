"""Explicit GPU task construction, before the legacy CPU environment constructor."""
from threading import RLock
from .facade import UnsupportedGpuOperation

_FACTORIES = {}
_LOCK = RLock()


def register_gpu_task_factory(problem_name, factory, *, replace=False):
    """Register a model-only task builder connected to a GPU runtime.

    The builder must not construct a CPU task environment and replace its sim
    afterwards: that constructor already executes CPU physical initialization.
    Registration is integration plumbing, not a numerical acceptance certificate.
    """
    if not isinstance(problem_name, str) or not problem_name or not callable(factory):
        raise ValueError("a problem name and callable factory are required")
    with _LOCK:
        if problem_name in _FACTORIES and not replace:
            raise ValueError(f"GPU task factory already registered: {problem_name}")
        _FACTORIES[problem_name] = factory


def create_gpu_environment(problem_name, bddl_file_name, **environment_kwargs):
    with _LOCK:
        factory = _FACTORIES.get(problem_name)
    if factory is None:
        # Lazy internal selection preserves the original public constructor.
        # An explicitly registered provider still takes precedence.
        from .product_factory import create
        with _LOCK:
            factory = _FACTORIES.get(problem_name)
            if factory is None:
                factory = install_model_only_gpu_factory(problem_name, create)
    env = factory(bddl_file_name=bddl_file_name, **environment_kwargs)
    try:
        if getattr(env, "physics_backend", None) != "gpu":
            raise UnsupportedGpuOperation("GPU factory did not return a GPU task environment")
        if getattr(getattr(env, "sim", None), "physics_backend", None) != "gpu":
            raise UnsupportedGpuOperation("GPU task environment has no GPU sim facade")
        for name in ("step", "reset", "close", "reset_from_xml_string", "seed", "get_sim_state", "set_state", "regenerate_obs_from_state", "_check_success", "_post_process", "_update_observables", "_get_observations"):
            if not callable(getattr(env, name, None)):
                raise UnsupportedGpuOperation(f"GPU task environment lacks {name}")
    except BaseException:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise
    return env


def install_model_only_gpu_factory(problem_name, runtime_factory, *, replace=False):
    """Wire a true runtime assembler into the original ControlEnv GPU entry.

    runtime_factory(built) returns GpuTaskComponents with actual device owners.
    It must reject unsupported model identities/layouts; no CPU task constructor,
    CPU physical data or fake ready fields may be used to supply missing stages.
    """
    if not callable(runtime_factory):
        raise ValueError("runtime_factory must be callable")

    def construct(*, bddl_file_name, **options):
        from libero.gpu.model import build_task_model
        from .task_driver import GpuTaskEnvironment

        def rebuild(xml_string, previous=None):
            built = build_task_model(bddl_file_name, problem_name=problem_name,
                                     xml_string_override=xml_string, **options)
            reuse = getattr(runtime_factory, "rebuild_components", None)
            components = reuse(built, previous) if previous is not None and callable(reuse) else runtime_factory(built)
            try:
                components.validate()
            except BaseException:
                if components is previous:
                    # Reuse provider owns rollback/quarantine of its transaction.
                    raise
                try:
                    components.observations.close()
                finally:
                    components.close_runtime()
                raise
            return components

        components = rebuild(None)
        try:
            return GpuTaskEnvironment(components, rebuild=rebuild)
        except BaseException as exc:
            # Constructor failure otherwise loses the only actual GPU owner
            # before ControlEnv returns. Attach diagnostic-only facts before
            # release, preserving the original execution exception.
            reader = getattr(getattr(components.sim, '_runtime', None), 'diagnostic_evidence', None)
            if callable(reader):
                try: exc.gpu_diagnostic_only = reader()
                except BaseException as diagnostic_error:
                    exc.gpu_diagnostic_reader_error = repr(diagnostic_error)
            cleanup_errors = []
            for close in (components.release,):
                try: close()
                except BaseException as close_error: cleanup_errors.append(repr(close_error))
            if cleanup_errors: exc.gpu_cleanup_errors = cleanup_errors
            raise

    def factory(*, bddl_file_name, **options):
        from contextlib import nullcontext
        scope = getattr(runtime_factory, 'construction_scope', nullcontext)
        with scope():
            return construct(bddl_file_name=bddl_file_name, **options)

    register_gpu_task_factory(problem_name, factory, replace=replace)
    return factory
