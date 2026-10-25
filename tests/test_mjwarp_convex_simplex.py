"""Independent QP known solutions and per-entry load-only ABI admission."""
import ctypes as ct
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation.mjwarp_convex_simplex_reference import closest,check
from libero.gpu.host.runtime.dynamics import mjwarp_motion as loader
from libero.gpu.host.build.compilation import MissingCompiledProduct


@pytest.mark.parametrize('vertices,point',[
    ([[1.,2.,3.]],[1.,2.,3.]),
    ([[1.,0.,0.],[0.,2.,0.]],[.8,.4,0.]),
    ([[1.,0.,1.],[0.,1.,1.],[-1.,-1.,1.]],[0.,0.,1.]),
    ([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.]],[0.,0.,0.]),
    ([[1.,2.,3.],[1.,2.,3.]],[1.,2.,3.])])
def test_independent_qp_known_solutions(vertices,point):
    got=closest(np.array(vertices));np.testing.assert_allclose(got['point'],point,atol=1.e-14);assert abs(sum(got['weights'])-1)<1.e-14


def test_residual_contract_rejects_invalid_mask_and_combination():
    simplex=np.array([[1,0,0],[0,2,0],[0,0,0],[0,0,0]],float);got=dict(weights=np.array([.8,.2,0,0]),point=np.array([.8,.4,0]),distance=np.sqrt(.8),mask=3)
    assert all(v['passed'] for v in check(simplex,2,got))
    got['mask']=1;assert not all(v['passed'] for v in check(simplex,2,got))
    got['mask']=3;got['point']=np.array([0,0,0]);assert not all(v['passed'] for v in check(simplex,2,got))


class Args(ct.Structure):_fields_=[('value',ct.c_int)]


def fake_loader(monkeypatch,tmp_path,*,entry=True,identity='expected',size=None,arch=80):
    p=tmp_path/'leaf.so';p.write_bytes(b'fixture');f=NS(flash_mjwarp_convex_source_id=lambda:identity.encode(),flash_mjwarp_convex_arch=lambda:arch,flash_mjwarp_convex_arguments_bytes=lambda:ct.sizeof(Args))
    if entry:setattr(f,'flash_mjwarp_convex_simplex_arguments_bytes',lambda:ct.sizeof(Args) if size is None else size)
    monkeypatch.setattr(loader.ct,'CDLL',lambda path:f);monkeypatch.setattr(loader,'source_id',lambda component:'expected');stream=NS(device=NS(is_cuda=True,arch=80));return p,f,stream


def test_default_and_entry_specific_abi(monkeypatch,tmp_path):
    p,f,s=fake_loader(monkeypatch,tmp_path);assert loader.load_leaf('convex',Args,p,s)[1] is f;assert loader.load_leaf('convex',Args,p,s,entry='simplex')[1] is f


def test_missing_new_entry_is_checked_without_source_authentication(monkeypatch,tmp_path):
    p,f,s=fake_loader(monkeypatch,tmp_path,entry=False,identity='old')
    with pytest.raises(MissingCompiledProduct,match='lacks simplex entry'):loader.load_leaf('convex',Args,p,s,entry='simplex')


def test_missing_entry_demands_offline_build(monkeypatch,tmp_path):
    p,f,s=fake_loader(monkeypatch,tmp_path,entry=False)
    with pytest.raises(MissingCompiledProduct,match='lacks simplex entry'):loader.load_leaf('convex',Args,p,s,entry='simplex')


def test_entry_abi_mismatch_rejected(monkeypatch,tmp_path):
    p,f,s=fake_loader(monkeypatch,tmp_path,size=999)
    with pytest.raises(ValueError,match='ABI'):loader.load_leaf('convex',Args,p,s,entry='simplex')


def test_entry_architecture_mismatch_rejected(monkeypatch,tmp_path):
    p,f,s=fake_loader(monkeypatch,tmp_path,arch=90)
    with pytest.raises(MissingCompiledProduct,match='architecture'):loader.load_leaf('convex',Args,p,s,entry='simplex')
