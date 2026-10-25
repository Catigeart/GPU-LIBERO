"""Scene runtime reads and explicit offline validation/migration."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
mujoco = pytest.importorskip('mujoco')
from libero.gpu.host.backends.mjwarp import scene_cache as c
from libero.gpu.host.build.compilation import MissingCompiledProduct

XML = '<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
TASK = dict(key='test/task', files={'scene.xml': 'catalogue-digest'})


def build_cache(root):
    folder, _, _, record = c.locate(root, XML, {}, TASK, prepare=True)
    model = mujoco.MjModel.from_xml_string(XML)
    model_file = root / 'test.mjb'
    mujoco.mj_saveModel(model, str(model_file), None)
    built = NS(xml=XML, native_model=model, metadata={'model_sha256': c.digest(model_file)})
    c.finish(folder, built, None, record, prepare=True, xml=XML)
    model_file.unlink()
    return folder, built, record


def test_runtime_miss_never_scans_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'glob', lambda *a, **k: pytest.fail('runtime directory scan'))
    with pytest.raises(MissingCompiledProduct, match='offline'):
        c.locate(tmp_path, XML, {}, TASK, prepare=False)
    assert not list(tmp_path.iterdir())


def test_runtime_hit_never_reads_hash_inputs_or_reserializes(tmp_path, monkeypatch):
    folder, built, _ = build_cache(tmp_path)
    monkeypatch.setattr(c, 'digest', lambda *a: pytest.fail('runtime content hashing'))
    monkeypatch.setattr(Path, 'glob', lambda *a, **k: pytest.fail('runtime directory scan'))
    _, model, row, record = c.locate(tmp_path, XML, {}, TASK, prepare=False)
    assert model == folder / 'model.mjb'
    monkeypatch.setattr(mujoco, 'mj_saveModel', lambda *a: pytest.fail('runtime publication'))
    c.finish(folder, built, row, record)
    assert mujoco.MjModel.from_binary_path(str(model)).nv == 6


def test_offline_rejects_changed_binary_and_assets(tmp_path, monkeypatch):
    folder, _, _ = build_cache(tmp_path)
    model = folder / 'model.mjb'
    original = model.read_bytes()
    model.write_bytes(b'corrupt')
    with pytest.raises(MissingCompiledProduct, match='changed'):
        c.locate(tmp_path, XML, {}, TASK, prepare=True)
    model.write_bytes(original)
    monkeypatch.setattr(c, 'offline_inputs', lambda xml: {'assets': {'changed': 'digest'}})
    with pytest.raises(MissingCompiledProduct, match='changed'):
        c.locate(tmp_path, XML, {}, TASK, prepare=True)


def test_runtime_rejects_binding_schema_missing_model_and_different_options(tmp_path):
    folder, _, _ = build_cache(tmp_path)
    path = folder / 'receipt.json'
    row = json.loads(path.read_text())
    for update in ({'schema': 'unknown'}, {'binding': dict(row['binding'], task='wrong')}):
        path.write_text(json.dumps(dict(row, **update)))
        with pytest.raises(MissingCompiledProduct):
            c.locate(tmp_path, XML, {}, TASK, prepare=False)
    path.write_text(json.dumps(row))
    with pytest.raises(MissingCompiledProduct):
        c.locate(tmp_path, XML, {'camera': True}, TASK, prepare=False)
    (folder / 'model.mjb').unlink()
    with pytest.raises(MissingCompiledProduct):
        c.locate(tmp_path, XML, {}, TASK, prepare=False)


def test_legacy_migration_only_offline_and_no_recompile(tmp_path):
    folder, built, record = build_cache(tmp_path)
    old = folder.with_name('legacy-key')
    folder.rename(old)
    row = json.loads((old / 'receipt.json').read_text())
    inputs = row.pop('offline_inputs')
    row.pop('schema')
    row['binding']['assets'] = inputs['assets']
    row['binding']['engine']['library_sha256'] = inputs['library_sha256']
    (old / 'receipt.json').write_text(json.dumps(row))
    with pytest.raises(MissingCompiledProduct):
        c.locate(tmp_path, XML, {}, TASK, prepare=False)
    target, compiled, row, record = c.locate(tmp_path, XML, {}, TASK, prepare=True)
    assert target == folder and compiled == old / 'model.mjb'
    c.finish(target, built, row, record, prepare=True, xml=XML)
    assert c.locate(tmp_path, XML, {}, TASK, prepare=False)[1] == folder / 'model.mjb'
    assert compiled.exists()  # Other processes may still refer to the old product.


def test_processed_xml_verified_offline_and_failed_publication_cleans_temp(tmp_path):
    folder, built, record = build_cache(tmp_path)
    row = json.loads((folder / 'receipt.json').read_text())
    built.xml += ' '
    with pytest.raises(MissingCompiledProduct, match='processed XML'):
        c.finish(folder, built, row, record, prepare=True, xml=XML)
    built.metadata['model_sha256'] = 'wrong'
    failed = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='serialization'):
        c.finish(failed, built, None, record, prepare=True, xml=XML)
    assert list(failed.iterdir()) == []
    with pytest.raises(MissingCompiledProduct, match='publication'):
        c.finish(failed, built, None, record)
