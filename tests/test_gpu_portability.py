"""Preparation identity and resource relocation checks without CUDA execution."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from libero.gpu.host.build import task_catalog, substep_catalog
from libero.gpu.host.build.compilation import compilation_scope, MissingCompiledProduct
from libero.gpu.host.build.project_paths import relative_record
from libero.gpu.host.runtime.core import product_paths


class PortabilityTests(unittest.TestCase):
    def test_scene_seed_is_explicit_and_data_paths_follow_the_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row=dict(key='suite/task.bddl', seed=42,directory='scenes/fixed',constraint_package='specializations/s')
            (root/'TASK_INSTANCES.json').write_text(json.dumps(dict(tasks=[row])))
            selected=task_catalog.scene_record(root,'/tasks/suite/task.bddl')
            self.assertEqual(selected['seed'],42)
            self.assertEqual(selected['directory'],str(root/'scenes/fixed'))
            self.assertEqual(task_catalog.scene_record(root,'/tasks/suite/task.bddl',42),selected)
            with self.assertRaises(MissingCompiledProduct):
                task_catalog.scene_record(root,'/tasks/suite/task.bddl',7)
            stored=relative_record(dict(library=str(root/'products/kernel.so')),root)
            self.assertEqual(stored,dict(library='products/kernel.so'))

    def test_fp32_catalog_reuses_selected_cpu_data_and_preserves_other_tasks(self):
        from libero.gpu.host.backends.cuda_fp32 import catalog
        import mujoco
        from libero.gpu import model
        import sys
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scene=root/'scenes/fixed';scene.mkdir(parents=True)
            bddl=root/'task.bddl';bddl.write_text('fixed task')
            files={}
            for name in ('model.mjb','scene.xml','initial_states.npy'):
                (scene/name).write_bytes(name.encode());files[name]=catalog.digest(scene/name)
            row=dict(key='suite/task.bddl',suite='suite',task_id=0,seed=0,directory='scenes/fixed',
                model_sha256=files['model.mjb'],files=files,bddl_sha256=catalog.digest(bddl))
            other=dict(key='suite/other.bddl',seed=0)
            index=root/'TASK_INSTANCES.json';index.write_text(json.dumps(dict(tasks=[row,other])))
            inputs=root/'mjwarp/scene-inputs/fixed.npz';inputs.parent.mkdir(parents=True);inputs.write_bytes(b'CPU inputs')
            inputs.with_suffix('.json').write_text(json.dumps(dict(seed=0,model_sha256=files['model.mjb'],
                scene_sha256=files['scene.xml'],init_sha256=files['initial_states.npy'],data_sha256=catalog.digest(inputs))))
            task=dict(row,bddl=str(bddl))
            with patch.object(task_catalog,'task_catalog',return_value=[task]), \
                    patch.object(mujoco,'__version__','2.3.7'), \
                    patch.object(model,'build_task_model',side_effect=AssertionError('cached scene rebuilt')), \
                    patch.object(sys,'argv',['catalog','--build-root',str(root),'--tasks','suite:0']):
                catalog.main()
            self.assertEqual(json.loads(index.read_text())['tasks'],[row,other])

    def test_equal_nv_families_select_the_full_shape(self):
        first=substep_catalog.SubstepVariant(21,23,86,11)
        second=substep_catalog.SubstepVariant(21,24,87,12)
        with patch.object(substep_catalog,'variants',return_value=(first,second)):
            self.assertIs(substep_catalog.select_variant(21,87,nq=24,njnt=12,nu=9),second)
            self.assertNotEqual(first.family_suffix,second.family_suffix)
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                substep_catalog.select_variant(21)

    def test_load_only_sdk_scope_does_not_inspect_or_download_development_files(self):
        with compilation_scope(False), patch.object(product_paths,'_sdk_valid',side_effect=AssertionError('SDK inspection')), \
                patch.object(product_paths.urllib.request,'urlopen',side_effect=AssertionError('SDK download')):
            self.assertIsNone(product_paths.mathdx_root())
            self.assertIsNone(product_paths.selected_mathdx_root())
            with product_paths.toolchain_scope(None):pass

    def test_direction_admission_matches_implemented_small_and_padded_kernels(self):
        from libero.gpu.host.runtime.newton import mjwarp_direction as direction
        model=NS(nv=35,block_dim=NS(update_gradient_cholesky=64,update_gradient_cholesky_blocked=32))
        self.assertIsNone(direction.unsupported_reason(model))
        model.nv=65
        with patch.object(direction,'FAMILIES',direction.FAMILIES+(65,)):
            self.assertIn('no small template',direction.unsupported_reason(model))

    def test_native_product_identity_ignores_deployment_and_sdk_prefixes(self):
        from libero.gpu.host.build.product_cache import _selection
        old=dict(flags=['-I','/old/checkout/libero/libero/gpu/cuda/common',
                        '-I','/sdk/include','-I','/sdk/external/cutlass/include'],
                 archive=dict(path='/sdk/lib/libcusolverdx.a',sha256='old'))
        new=dict(flags=['-I','/new/wheel/libero/gpu/cuda/common'],archive=True)
        self.assertEqual(_selection(old),_selection(new))
        old['archive']='/sdk/lib/libcusolverdx.a'
        self.assertEqual(_selection(old),_selection(new))

    def test_assets_use_configured_roots_and_ignore_checkout_names(self):
        from libero.gpu.host.backends.cuda_fp32.scene import relocate_assets
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);assets=root/'resources';assets.mkdir();(assets/'mesh.obj').write_text('asset')
            suite=root/'suite-assets';suite.mkdir()
            with patch.dict('sys.modules',{'libero.libero':NS(get_libero_path=lambda key:str(assets)),
                    'robosuite.models':NS(assets_root=str(suite))}):
                xml='<mujoco><asset><mesh file="/gone/arbitrary-checkout/assets/mesh.obj"/></asset></mujoco>'
                relative=relocate_assets(xml,relative=True)
                self.assertIn('libero-assets://mesh.obj',relative)
                self.assertIn(str(assets/'mesh.obj'),relocate_assets(relative))
                self.assertEqual(relocate_assets(relocate_assets(relative),relative=True),relative)
