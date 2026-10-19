"""Offline substep instances shared by all shipped task models.

Model dimensions select a compiled library; live constraint counts never select
or generate a new binary. This module has no CUDA or Warp dependency.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path


PARAMETERS = Path(__file__).resolve().parents[2] / 'data/libero_task_parameters.json'


@dataclass(frozen=True)
class SubstepVariant:
    nv: int
    nq: int
    nM: int
    njnt: int
    nu: int = 9
    contacts: int = 5000

    @property
    def rows(self):
        return 6 * self.contacts + self.nv + self.njnt

    @property
    def quaternion_joints(self):
        return self.nq - self.nv

    @property
    def hessian_tile_rows(self):
        # Match the existing generic Newton pending-row producer.
        return 16 if self.nv <= 43 else 8

    @property
    def suffix(self):
        return '' if self.nv == 39 else f'_nv{self.nv}'

    def product(self, profile=False):
        return 'substep' + ('_newton_profile' if profile else '') + self.suffix


@lru_cache(maxsize=1)
def variants():
    data = json.loads(PARAMETERS.read_text())
    result = {}
    for task in data['tasks']:
        d = task['dimensions']
        variant = SubstepVariant(*(d[k] for k in ('nv', 'nq', 'nM', 'njnt', 'nu')))
        previous = result.setdefault(variant.nv, variant)
        if previous != variant:
            raise ValueError('substep catalogue needs a wider specialization key')
    return tuple(result[k] for k in sorted(result))


def select_variant(nv, nM=None, *, nq=None, njnt=None, nu=None, rows=None, contacts=None):
    for variant in variants():
        if variant.nv == nv and all(value is None or value == getattr(variant, key)
                for key, value in dict(nM=nM, nq=nq, njnt=njnt, nu=nu,
                                       rows=rows, contacts=contacts).items()):
            return variant
    raise ValueError('substep dimensions are absent from the offline task catalogue: ' +
                     str((nv, nM, nq, njnt, nu, rows, contacts)))
