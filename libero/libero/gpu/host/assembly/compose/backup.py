"""Cold multi-field backup copies; original ordering is retained for overlaps.

The copies use an element grid, so compose's world ownership contract does not
apply. A single generated kernel copies disjoint fields without arithmetic.
Event selection and physics remain separate producer/consumer launches.
"""
import libero.gpu.precision as scalar
import ast
import inspect
import textwrap

import warp as wp

from libero.gpu.host.runtime.step.recovery_event_runtime import EventRecoveringRuntime, _copy_f64, _copy_i32
from libero.gpu.host.runtime.step.shape_physics import ShapePhysicsForward
from libero.gpu.host.runtime.step.shape_runtime import _token
from .compose import _load


def _range(array):
    return int(array.ptr), int(array.ptr) + array.size * wp.types.type_size_in_bytes(array.dtype)


def _overlap(a, b):
    lo, hi = _range(a)
    other_lo, other_hi = _range(b)
    return lo < other_hi and other_lo < hi


class MultiFieldCopyPlan:
    """Retained contiguous fields, with optional stable event row selection.

    Source/source aliases are safe. Any destination/source, destination/slot,
    or destination/destination overlap uses the original sequential copies.
    """
    def __init__(self, pairs, *, slot=None):
        self.pairs = tuple(pairs)
        self.slot = slot
        self.nworld = 1 if slot is None else int(slot.size)
        if slot is not None and (slot.dtype != wp.int32 or slot.ndim != 1 or self.nworld < 1):
            raise ValueError("stable i32 event slot per world required")
        self.sources, self.targets = [], []
        self.widths, self.rows = [], []
        for source, target in self.pairs:
            if not source.is_contiguous or not target.is_contiguous:
                raise ValueError("contiguous backup buffers required")
            if source.dtype != target.dtype or source.device != target.device:
                raise ValueError("backup field schema differs")
            if source.dtype not in (scalar.wp_real, wp.int32, wp.int64):
                raise TypeError("backup scalar dtype unsupported")
            if slot is not None and (source.device != slot.device or source.dtype == wp.int64):
                raise ValueError("original event copy requires f64/i32 on the slot device")
            if slot is None and source.size != target.size:
                raise ValueError("backup field extent differs")
            width, rows = source.size, 1
            if slot is not None:
                if source.size % self.nworld:
                    raise ValueError("event source world extent differs")
                width = source.size // self.nworld
                if target.ndim == 3 and target.shape[0] == self.nworld and target.shape[2] == width:
                    rows = target.shape[1]
                elif target.ndim == 2 and target.shape[1] == width and target.shape[0] % self.nworld == 0:
                    rows = target.shape[0] // self.nworld
                else:
                    raise ValueError("event backup requires world-major rows and per-world width")
                if rows < 1:
                    raise ValueError("positive event backup capacity required")
            self.widths.append(width)
            self.rows.append(rows)
            self.sources.append(source.reshape((-1,)))
            self.targets.append(target.reshape((-1,)))
        if len({str(source.device) for source, _ in self.pairs}) > 1:
            raise ValueError("one backup device required")
        self._owner_tokens = tuple((_token(source), _token(target)) for source, target in self.pairs)
        self._slot_token = None if slot is None else _token(slot)
        targets = [target for _, target in self.pairs]
        self.fallback = any(_overlap(target, source) for target in targets for source, _ in self.pairs)
        self.fallback |= any(_overlap(a, b) for i, a in enumerate(targets) for b in targets[i + 1:])
        if slot is not None:
            self.fallback |= any(_overlap(target, slot) for target in targets)
        if self.fallback and self.nworld != 1:
            raise ValueError("world-major aliased event copies require an explicit ordered fallback")
        args = ["slot: wp.array(dtype=wp.int32)"] if slot is not None else []
        body = ["    cell = wp.tid()"] if slot is None else ["    world, cell = wp.tid()"]
        if slot is not None:
            body += ["    row = slot[world]", "    if row < 0: return"]
        names = {scalar.wp_real: scalar.name, wp.int32: "int32", wp.int64: "int64"}
        for index, (source, _) in enumerate(self.pairs):
            dtype = names[source.dtype]
            args += [f"s{index}: wp.array(dtype=wp.{dtype})", f"t{index}: wp.array(dtype=wp.{dtype})"]
            width, rows = self.widths[index], self.rows[index]
            origin = f"world * {width} + cell" if slot is not None else "cell"
            destination = f"(world * {rows} + row) * {width} + cell" if slot is not None else "cell"
            admitted = f"cell < {width} and row < {rows}" if slot is not None else f"cell < {width}"
            body += [f"    if {admitted}: t{index}[{destination}] = s{index}[{origin}]"]
        code = "@wp.kernel(enable_backward=False)\ndef copy_fields(" + ", ".join(args) + "):\n" + "\n".join(body) + "\n"
        self.kernel = _load("backup_fields", code, {"wp": wp}).copy_fields
        self.inputs = ([slot] if slot is not None else []) + [a for pair in zip(self.sources, self.targets) for a in pair]
        extent = max(self.widths, default=0)
        self.dim = extent if slot is None else (self.nworld, extent)

    def launch(self, stream):
        if tuple((_token(source), _token(target)) for source, target in self.pairs) != self._owner_tokens:
            raise ValueError("prepared backup buffer metadata changed")
        if self.slot is not None and _token(self.slot) != self._slot_token:
            raise ValueError("prepared event backup slot changed")
        if self.pairs and self.pairs[0][0].device != stream.device:
            raise ValueError("foreign backup device")
        if self.fallback:
            for index, (source, target) in enumerate(self.pairs):
                if not source.size:
                    continue
                if self.slot is None:
                    wp.copy(target, source, stream=stream)
                else:
                    kernel = _copy_f64 if source.dtype == scalar.wp_real else _copy_i32
                    if source.dtype == wp.int64:
                        raise TypeError("original event copy supports f64/i32 only")
                    width, rows = self.widths[index], self.rows[index]
                    view = target.reshape((self.nworld * rows, width))
                    wp.launch(kernel, (self.nworld, width), inputs=[source.reshape((-1,)), self.slot, view, width, rows],
                              device=stream.device, stream=stream)
        elif any(self.widths):
            wp.launch(self.kernel, self.dim, inputs=self.inputs, device=stream.device, stream=stream)


class FusedEventRecoveringRuntime(EventRecoveringRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.capture_events:
            return
        self._fusion_event_plans = {id(outputs): MultiFieldCopyPlan(
            [(self.event_sources[name], target) for name, target in outputs.items()], slot=self.event_slot)
            for outputs in (self.event_before, self.event_after, self.event_after_euler)}

    def _copy_event(self, outputs):
        plan = self._fusion_event_plans.get(id(outputs))
        if plan is None:
            raise ValueError("prepared event copy mapping changed")
        current = tuple((self.event_sources[name], target) for name, target in outputs.items())
        if len(current) != len(plan.pairs) or any(
                source is not old_source or target is not old_target
                for (source, target), (old_source, old_target) in zip(current, plan.pairs)):
            raise ValueError("prepared event copy field owner changed")
        if self.event_slot is not plan.slot:
            raise ValueError("prepared event copy slot owner changed")
        plan.launch(self.stream)


def _entry_enqueue():
    tree = ast.parse(textwrap.dedent(inspect.getsource(ShapePhysicsForward.enqueue)))
    class Replace(ast.NodeTransformer):
        count = 0
        def visit_For(self, node):
            if isinstance(node.iter, ast.Call) and ast.unparse(node.iter) == "self.entry_witness.items()":
                self.count += 1
                return ast.parse("if self._fusion_entry_copy is not None:\n    self._fusion_entry_copy.launch(self.stream)").body[0]
            return self.generic_visit(node)
    replace = Replace()
    tree = replace.visit(tree)
    if replace.count != 1:
        raise ValueError("one original witness copy loop required")
    ast.fix_missing_locations(tree)
    namespace = dict(ShapePhysicsForward.enqueue.__globals__)
    exec(compile(tree, __file__ + ":entry-cold", "exec"), namespace)
    return namespace["enqueue"]


class FusedShapePhysicsForward(ShapePhysicsForward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fusion_entry_copy = (MultiFieldCopyPlan(
            [(self.entry_sources[name], target) for name, target in self.entry_witness.items()])
            if self.entry_witness else None)
        self._fusion_entry_owners = tuple((name, source, self.entry_witness.get(name)) for name, source in self.entry_sources.items())

    def validate(self):
        super().validate()
        if hasattr(self, "_fusion_entry_owners"):
            if tuple(self.entry_sources) != tuple(name for name, _, _ in self._fusion_entry_owners):
                raise ValueError("prepared entry source names changed")
            for name, source, target in self._fusion_entry_owners:
                if self.entry_sources[name] is not source or self.entry_witness.get(name) is not target:
                    raise ValueError("prepared entry copy owner changed: " + name)

    enqueue = _entry_enqueue()
