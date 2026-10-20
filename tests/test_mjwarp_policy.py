"""Independent upstream references, not a candidate, define the error envelope."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'tools/validation/mjwarp_policy.py'
spec = importlib.util.spec_from_file_location('mjwarp_replay_policy', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_candidate_does_not_change_reference_budget():
    a = {'qpos': np.zeros((2, 3))}
    b = {'qpos': np.full((2, 3), .01)}
    near = module.compare_replays({'qpos': np.full((2, 3), .02)}, a, b)
    far = module.compare_replays({'qpos': np.full((2, 3), 10.)}, a, b)
    assert all(x['passed'] for x in near)
    assert not any(x['passed'] for x in far)
    assert [x['limit'] for x in near] == [x['limit'] for x in far] == [.1] * 4


def test_control_and_time_are_exact():
    reference = {'ctrl': np.zeros((2, 9)), 'time': np.zeros(2)}
    changed = {key: value + 1e-7 for key, value in reference.items()}
    assert not any(x['passed'] for x in module.compare_replays(changed, reference, reference))


def test_bad_reference_shapes_and_nonfinite_values_fail():
    ref = {'qpos': np.zeros((2, 3))}
    with pytest.raises(ValueError):
        module.compare_replays(ref, ref, {'qpos': np.zeros((1, 3))})
    with pytest.raises(ValueError):
        module.compare_replays(ref, ref, {'qpos': np.full((2, 3), np.nan)})
    assert not any(x['passed'] for x in module.compare_replays({'qpos': np.full((2, 3), np.nan)}, ref, ref))
