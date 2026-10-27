"""Offline substep instances shared by all shipped task models.

Model dimensions select a compiled library; live constraint counts never select
or generate a new binary. This module has no CUDA or Warp dependency.
"""
from dataclasses import dataclass
from functools import lru_cache
import re
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
        suffix = self.family_suffix
        return '' if suffix == '_nv39' else suffix

    @property
    def family_suffix(self):
        if sum(v.nv == self.nv for v in variants()) == 1:
            return f'_nv{self.nv}'
        return f'_nv{self.nv}_nq{self.nq}_nm{self.nM}_nj{self.njnt}_nu{self.nu}'

    def product(self, profile=False):
        return 'substep' + ('_newton_profile' if profile else '') + self.suffix


@lru_cache(maxsize=1)
def variants():
    declarations = PARAMETERS.parent.parent / 'cuda/step/instantiations/libero_models.def'
    return tuple(SubstepVariant(*map(int, row)) for row in re.findall(
        r'^LIBERO_MODEL\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)$',
        declarations.read_text(), re.MULTILINE))


def select_variant(nv, nM=None, *, nq=None, njnt=None, nu=None, rows=None, contacts=None):
    matches = [variant for variant in variants()
        if variant.nv == nv and all(value is None or value == getattr(variant, key)
            for key, value in dict(nM=nM, nq=nq, njnt=njnt, nu=nu,
                                   rows=rows, contacts=contacts).items())]
    if len(matches) == 1:
        return matches[0]
    raise ValueError('substep dimensions are absent or ambiguous in the offline task catalogue: ' +
                     str((nv, nM, nq, njnt, nu, rows, contacts)))
