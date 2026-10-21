"""Offline modern scene admission; corruption and processed-XML changes fail."""
import hashlib,json
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('mujoco')
from libero.gpu.host.backends.mjwarp import scene_cache as c
from libero.gpu.host.build.compilation import MissingCompiledProduct


def setup_cache(monkeypatch):
    binding={'engine':'3.16.0','asset':'fixed','options':{}}
    monkeypatch.setattr(c,'binding',lambda *a:binding)
    return binding


def test_miss_is_load_only_and_preparation_is_explicit(tmp_path,monkeypatch):
    setup_cache(monkeypatch)
    with pytest.raises(MissingCompiledProduct,match='CPU2.3.7'):c.locate(tmp_path,'xml',{},None,prepare=False)
    folder,model,row,record=c.locate(tmp_path,'xml',{},None,prepare=True)
    assert model is None and row is None and not folder.exists()


def test_cache_rejects_corrupt_binary_and_different_binding(tmp_path,monkeypatch):
    binding=setup_cache(monkeypatch);folder,_,_,_=c.locate(tmp_path,'x',{},None,prepare=True);folder.mkdir(parents=True);model=folder/'model.mjb';model.write_bytes(b'modern-model');sha=hashlib.sha256(model.read_bytes()).hexdigest();receipt=folder/'receipt.json';row=dict(binding=binding,model_sha256=sha,processed_xml_sha256='x');receipt.write_text(json.dumps(row))
    assert c.locate(tmp_path,'x',{},None,prepare=False)[1]==model
    model.write_bytes(b'corrupt')
    with pytest.raises(MissingCompiledProduct,match='changed'):c.locate(tmp_path,'x',{},None,prepare=False)
    model.write_bytes(b'modern-model');row['binding']={'engine':'2.3.7'};receipt.write_text(json.dumps(row))
    with pytest.raises(MissingCompiledProduct,match='changed'):c.locate(tmp_path,'x',{},None,prepare=False)


def test_processed_xml_and_reserialized_identity_are_verified(tmp_path):
    built=NS(xml='actual-processed-xml',metadata={'model_sha256':'same'})
    row=dict(processed_xml_sha256=hashlib.sha256(built.xml.encode()).hexdigest(),model_sha256='same')
    c.finish(tmp_path,built,row,{})
    built.xml+='-changed'
    with pytest.raises(MissingCompiledProduct,match='processed XML'):c.finish(tmp_path,built,row,{})
