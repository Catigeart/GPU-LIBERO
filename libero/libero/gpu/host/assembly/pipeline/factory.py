"""Cold integration of the eight large-fusion work packages.

All substitutions happen during construction. Original factories, source
modules and the preceding fusion baseline are retained.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ast
import hashlib
from pathlib import Path
from functools import partial
from types import SimpleNamespace

from libero.gpu.host.assembly.compose import factory as first
from libero.gpu.host.assembly.compose.backup import FusedEventRecoveringRuntime
from libero.gpu.host.assembly.compose.s_rows import fused_workspace_type
from libero.gpu.host.runtime.dynamics import wide_factory, wide_runtime
from libero.gpu.host.runtime.newton import live_copy_api
from libero.gpu.host.runtime.step import recovery_event_factory, s04_jar_specialization_r01

from . import dynamics, euler, recovery, newton
from .collision import make_fused_adapter
from .copies import FusedNoForwardHold, FusedRuntimePostAdapter as CopyPost


class _HoldConstructor(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node) == "r._NoForwardHold":
            return ast.Name(id="_FusedHold", ctx=ast.Load())
        return self.generic_visit(node)


_original_initialize = first._derive(wide_runtime.GenericWideRuntimeDynamicsAdapter.__init__,
                                    {"_FusedHold": FusedNoForwardHold}, _HoldConstructor())


def _initialize_dynamics(self, *args, **kwargs):
    _original_initialize(self, *args, **kwargs)
    dynamics.prepare_dynamics(self)


FusedWideDynamicsAdapter = type("FusedWideDynamicsAdapterR03",
    (wide_runtime.GenericWideRuntimeDynamicsAdapter,),
    dict(dynamics.runtime_methods(), __init__=_initialize_dynamics, __module__=__name__))


class FusedRuntimePostAdapter(dynamics.FusedRuntimePostAdapter, CopyPost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _DynamicsSources(ast.NodeTransformer):
    count = 0
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "names" for t in node.targets):
            if not isinstance(node.value, ast.Tuple):
                raise ValueError("original source bundle tuple required")
            gpu = (GPU_ROOT / '.')
            paths = sorted(((GPU_ROOT / 'host/assembly/compose')).glob("*.py")) + sorted((GPU_ROOT / 'host/assembly/pipeline').glob("*.py"))
            node.value.elts.extend(ast.Constant(value=str(p.relative_to(gpu))) for p in paths)
            self.count += 1
        return self.generic_visit(node)


def prepare_from_built(*args, **kwargs):
    transform = _DynamicsSources()
    prepare = first._derive(wide_factory.prepare_from_built,
        {"GenericRuntimeDynamicsAdapter": FusedWideDynamicsAdapter}, transform)
    if transform.count != 1:
        raise ValueError("one cold dynamics source bundle required")
    return prepare(*args, **kwargs)


def prepare_post(*args, **kwargs):
    adapter = first._derive(wide_factory.prepare_post,
        {"RuntimePostAdapter": FusedRuntimePostAdapter})(*args, **kwargs)
    # The original factory binds and freezes the post workspace before the
    # source launcher's validation can be used for cold planning.
    dynamics.prepare_post(adapter)
    return adapter


def _load_candidate(nv, nm, build_root):
    package, manifest = s04_jar_specialization_r01.load_candidate(nv, nm, build_root)
    view = SimpleNamespace(workspace=package.workspace,
        solver=SimpleNamespace(FwdConstraintSolver=newton.fused_solver_class(package)))
    return view, dict(manifest, large_fusion_batch=3)


def _newton_factory(**kwargs):
    adapter = first._derive(live_copy_api.prepare_live_copy_runtime,
        {"load_candidate": _load_candidate,
         "ColdNewtonRuntimeAdapter": newton.FusedFinalEventAdapter})(**kwargs)
    adapter.final_event = adapter.solver
    return adapter


def _cold_factories():
    return make_fused_adapter, _newton_factory, "cn"


FusedRuntime = recovery.runtime_type(euler.runtime_type(FusedEventRecoveringRuntime))
_recovery_loop = first._derive(recovery_event_factory._recovery_loop,
    {"RecoveringShapeRuntime": FusedRuntime,
     "DataRecoveryWorkspace": recovery.FusedDataRecoveryWorkspace})


class _ColdImports(ast.NodeTransformer):
    dynamics = 0
    euler = 0
    inserted = 0

    def visit_Constant(self, node):
        if node.value == "libero.gpu.host.runtime.dynamics.wide_factory":
            self.dynamics += 1
            return ast.Constant(value="libero.gpu.host.assembly.pipeline.factory")
        return node

    def visit_ImportFrom(self, node):
        if node.module == "libero.gpu.reference.integration.euler_integrator.wide_angle":
            node.module = "libero.gpu.host.assembly.pipeline.euler"
            node.names = [ast.alias(name="FusedEulerWorkspace", asname="WideAngleEulerWorkspace")]
            self.euler += 1
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.iter) == "ast.walk(tree)":
            node.body.extend(ast.parse('''
if isinstance(node, ast.ImportFrom) and node.module == "shape_physics":
    node.module = "libero.gpu.host.assembly.pipeline.forward"
    node.level = 0
    node.names = [ast.alias(name="FusedShapePhysicsForward", asname="ShapePhysicsForward")]
if isinstance(node, ast.ImportFrom) and node.module == "libero.gpu.host.runtime.dynamics.runtime_factory":
    if any(alias.name == "prepare_post" for alias in node.names):
        node.module = "libero.gpu.host.assembly.pipeline.factory"
if isinstance(node, ast.ImportFrom) and node.module == "libero.gpu.host.runtime.foundation.runtime_adapter":
    node.module = "libero.gpu.host.assembly.pipeline.forward"
    node.names = [ast.alias(name="FusedFoundationForwardAdapter", asname="FoundationForwardAdapter")]
if isinstance(node, ast.ImportFrom) and node.module == "libero.gpu.host.runtime.dynamics.runtime_adapter":
    if any(alias.name == "RuntimeControllerAdapter" for alias in node.names):
        node.module = "libero.gpu.host.assembly.pipeline.osc"
        node.names = [ast.alias(name="FusedRuntimeControllerAdapter", asname="RuntimeControllerAdapter")]
if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "sm"
        and node.func.attr == "PositionReferenceWorkspace"):
    node.func = ast.Call(func=ast.Name(id="_fused_s_workspace_type", ctx=ast.Load()),
                         args=[node.func], keywords=[])
''').body)
            self.inserted += 1
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict" and any(
                k.arg == "_recovery_loop" for k in node.keywords):
            node.keywords.append(ast.keyword(arg="_fused_s_workspace_type",
                                value=ast.Name(id="_fused_s_workspace_type", ctx=ast.Load())))
        return node


def _validate_controller_selection(dynamics_owner, controller):
    """Match the actual controller family and its frozen workspace owner."""
    if getattr(dynamics_owner, "joint_controller", False):
        from libero.gpu.host.runtime.control.joint import RuntimeJointControllerAdapter
        assert isinstance(controller, RuntimeJointControllerAdapter)
    else:
        from .osc import FusedRuntimeControllerAdapter
        assert isinstance(controller, FusedRuntimeControllerAdapter)
    assert controller.controller is dynamics_owner.control
    return True


def _validate_selection(bundle):
    from .collision import FusedCollisionChainWorkspace
    from .forward import FusedShapePhysicsForward, FusedFoundationForwardAdapter
    from .osc import FusedRuntimeControllerAdapter
    loop = bundle.runtime
    assert isinstance(bundle.dynamics, FusedWideDynamicsAdapter)
    assert isinstance(loop.forward.c.workspace, FusedCollisionChainWorkspace)
    assert isinstance(loop.forward, FusedShapePhysicsForward)
    assert isinstance(loop.forward.f, FusedFoundationForwardAdapter)
    assert isinstance(bundle.newton, newton.FusedFinalEventAdapter)
    assert isinstance(loop.euler, euler.FusedEulerWorkspace)
    assert isinstance(loop.recovery, recovery.FusedDataRecoveryWorkspace)
    assert isinstance(bundle.post, FusedRuntimePostAdapter)
    if bundle.controller is not None:
        _validate_controller_selection(bundle.dynamics, bundle.controller)
    assert all(isinstance(hold, FusedNoForwardHold) for hold in bundle.dynamics._holds.values())
    assert hasattr(loop.forward.s.workspace, "_fusion_s_before")
    assert loop.recovery._fusion_protocol_owner is loop
    assert hasattr(loop, "_fusion_euler_kernel")
    assert not loop.capture_events or hasattr(loop, "_fusion_warning_copy")
    return True


def prepare_fused_physics(*args, newton_diagnostic_profile='compatible', **kwargs):
    if kwargs.get("mode", "diagnostic") != "diagnostic":
        raise ValueError("large fusion candidate requires explicit diagnostic mode")
    transform = _ColdImports()
    def cold_factories():
        return (make_fused_adapter,
                partial(_newton_factory, diagnostic_profile=newton_diagnostic_profile), 'cn')
    prepare = first._derive(recovery_event_factory.prepare_recovery_event_physics,
        {"_event_cold_factories": cold_factories, "_recovery_loop": _recovery_loop,
         "_fused_s_workspace_type": fused_workspace_type}, transform)
    if (transform.dynamics, transform.euler, transform.inserted) != (1, 1, 1):
        raise ValueError("exact original cold substitutions required")
    bundle = prepare(*args, **kwargs)
    try:
        bundle.fusion_batch = 3
        gpu = (GPU_ROOT / '.')
        paths = sorted(((GPU_ROOT / 'host/assembly/compose')).glob("*.py")) + sorted((GPU_ROOT / 'host/assembly/pipeline').glob("*.py"))
        bundle.fusion_source_manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        bundle.fusion_dynamics_catalog = dynamics.plan_catalog(bundle.dynamics)
        bundle.fusion_post_catalog = dynamics.plan_catalog(bundle.post)
        bundle.validate_fusion_selection = lambda: _validate_selection(bundle)
        bundle.validate_fusion_selection()
    except BaseException:
        bundle.close()
        raise
    return bundle


def create(built):
    from libero.gpu.host.runtime.step.recovery_cleanup_factory_r02 import create as assemble
    return assemble(built, physics_factory=prepare_fused_physics,
                    factory_entry="libero.gpu.host.assembly.pipeline.factory:create")
