"""Prefetch hints for the original LIBERO action-boundary predicates.

The hints describe common read sets, not a proof of callback identity. A callback
may change after construction: boundary_predicates fetches any additional goal
fields from the same synchronized sample without evaluating the callback again.
"""
from dataclasses import dataclass, field


FULL_GOAL_FIELDS = ("qpos", "body_xpos", "body_xquat", "site_xpos",
                    "site_xmat", "contact", "ncon")


@dataclass(frozen=True)
class GoalReadPlan:
    fields: tuple = FULL_GOAL_FIELDS
    qualified: bool = False
    reason: str = "unrecognized task descriptor"
    _owner: object = field(default=None, repr=False, compare=False)

    def fields_for(self, shell):
        """Return this shell's prefetch hints; missing reads are resolved lazily."""
        return self.fields if shell is self._owner else FULL_GOAL_FIELDS

    @classmethod
    def from_shell(cls, shell):
        try:
            fields = _plan(shell)
            ordered = tuple(name for name in FULL_GOAL_FIELDS if name in fields)
            return cls(ordered, True, "known LIBERO predicate read hints", shell)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return cls(reason=str(error), _owner=shell)


def _plan(shell):
    # These read sets follow CPU ObjectState/SiteObjectState and their predicate
    # dispatch. Only task descriptors are inspected; no callback is invoked.
    fields = set()
    for atom in shell.parsed_problem["goal_state"]:
        if not isinstance(atom, (list, tuple)) or len(atom) not in (2, 3):
            raise ValueError("unsupported goal expression")
        predicate = atom[0]
        if predicate in ("open", "close", "turnon", "turnoff") and len(atom) == 2:
            fields.add("qpos")
            continue
        if predicate not in ("on", "in") or len(atom) != 3:
            raise ValueError("unknown predicate read set")
        source, target = (shell.object_states_dict[name] for name in atom[1:])
        if source.object_state_type != "object":
            raise ValueError("unknown relation source")
        fields.add("body_xpos")
        if target.object_state_type == "site":
            fields.update(("site_xpos", "site_xmat"))
            # Site containment itself has no contact requirement. On a site
            # checks contacts only when that site belongs to a physical object.
            parent = target.parent_name
            if predicate == "on" and (parent in shell.objects_dict or parent in shell.fixtures_dict):
                fields.update(("contact", "ncon"))
        elif target.object_state_type == "object" and predicate == "on":
            fields.update(("contact", "ncon"))
        else:
            # Object-specific containment and custom states use the full schema.
            raise ValueError("unknown relation target")
    if shell.tracking_object_states_change:
        # The original visual tracker checks the articulated stove's qpos.
        fields.add("qpos")
    return fields
