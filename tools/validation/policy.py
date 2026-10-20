"""One frozen-input numerical policy for live and offline comparisons.

This module performs no physics and does not select the GPU process precision.
The established FP64 envelope and FP32 sensitivity/residual rules are unchanged.
"""
from dataclasses import dataclass

import numpy as np

from .common import compare_endpoints
from .precision import acceleration_residuals, frozen_acceleration_floors


@dataclass
class ComparisonPolicy:
    precision: str
    worlds: int
    field_floors: dict
    model: object
    metadata: dict

    def evaluate(self, actual, previous, cpu, *, baseline_samples=()):
        checks = compare_endpoints(actual, previous, cpu, self.worlds,
            baseline_samples=baseline_samples,
            scalar_epsilon=np.finfo(np.float32).eps if self.precision == 'fp32' else 0.,
            field_floors=self.field_floors)
        result = dict(checks=checks, passed=all(c['passed'] for c in checks))
        if self.model is not None:
            residuals = acceleration_residuals(actual, self.model, self.worlds)
            result['acceleration_residuals'] = residuals
            result['passed'] &= all(c['passed'] for c in residuals)
        return result


def prepare_policy(precision, previous, worlds, *, model=None):
    """Construct all thresholds from frozen data before evaluating a candidate."""
    if precision not in ('fp64', 'fp32') or worlds < 1:
        raise ValueError('explicit fp64/fp32 precision and positive worlds required')
    metadata = dict(schema='libero-endpoint-policy-v1', precision=precision,
                    multiplier=3., absolute_floor=1e-10,
                    roundoff_multiplier=1024 if precision == 'fp32' else 0)
    floors = {}
    if precision == 'fp32':
        if model is None:
            raise ValueError('FP32 comparison requires the matching MuJoCo 2.3.7 model')
        floors, frozen = frozen_acceleration_floors(previous, model, worlds)
        metadata['acceleration'] = dict(
            kind='frozen-mass-conditioning-and-independent-force-balance',
            formula='2*kappa*u/(1-kappa*u) * frozen_acceleration_norm',
            unit_roundoff=float(np.finfo(np.float32).eps / 2), frozen=frozen)
    return ComparisonPolicy(precision, worlds, floors,
                            model if precision == 'fp32' else None, metadata)
