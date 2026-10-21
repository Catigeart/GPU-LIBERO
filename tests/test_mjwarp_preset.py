"""Preset expansion is explicit and does not change individual defaults."""
import pytest
from libero.gpu.host.backends.mjwarp.options import CUDA_LIBERO,expand


def test_default_does_not_enable_optional_mathematics():assert expand({})==({},'upstream')


def test_complete_shipped_preset_and_explicit_override():
    value,mode=expand(dict(preset='cuda_libero',fast_compile=True))
    assert mode=='cuda_libero' and all(value[k]==v for k,v in CUDA_LIBERO.items()) and value['fast_compile']
    value,_=expand(dict(preset='cuda_libero',newton='upstream'));assert value['newton']=='upstream'


def test_typo_rejected():
    with pytest.raises(ValueError):expand(dict(preset='typo'))
