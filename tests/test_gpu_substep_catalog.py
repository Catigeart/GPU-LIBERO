"""Task coverage and dimension-derived workspace safety, without a GPU."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SubstepCatalogTests(unittest.TestCase):
    def test_reference_accepts_each_retained_position_producer(self):
        from types import SimpleNamespace
        from libero.gpu.host.bindings.constraints import constraint_feature_selection
        from libero.gpu.host.build.substep_catalog import variants
        from libero.gpu.host.runtime.step.substep_binding import _check_variant
        for variant in variants():
            flags=constraint_feature_selection('fast',variant.nv,2,'single_cta',environ={})['flags']
            dimensions=dict(CF_S_DX_SM=800,NF_CAP=variant.nv,NJ=variant.njnt,
                NL_CAP=2*variant.njnt,NQ=variant.nq,NV=variant.nv,
                S_CONTACTS=variant.contacts,S_NBODY=38,S_ROWS=variant.rows)
            defines=dict(dimensions,CF_S_COMPATIBLE=0,CF_S_KBIP_PLANES=1,
                         **{'CF_S_'+key:value for key,value in flags.items()})
            def plan(values):
                return SimpleNamespace(manifest=dict(name='constraints_reference',
                    compile_commands=[['nvcc',*[f'-D{key}={value}' for key,value in values.items()]]]))
            with self.subTest(nv=variant.nv,flags=flags):
                _check_variant(plan(defines),dimensions)
                for key,value in [('CF_S_COMPATIBLE',1),('CF_S_KBIP_PLANES',0),
                                  ('CF_S_BATCH_COMPACT',2),('NV',variant.nv+1)]:
                    with self.assertRaisesRegex(ValueError,key):
                        _check_variant(plan(dict(defines,**{key:value})),dimensions)

    def test_numerical_binding_checks_actual_model_defines(self):
        from types import SimpleNamespace
        from libero.gpu.host.runtime.step.substep_binding import _check_variant
        dimensions=dict(D_NM=87,D_NU=9,D_NV=22,D_SM=800)
        plan=SimpleNamespace(manifest=dict(name='CUDA_D_launch_force_group0',
            compile_commands=[['nvcc','-DD_FAST_DATAFLOW=1',
                *[f'-D{k}={v}' for k,v in dimensions.items()]]]))
        _check_variant(plan,dimensions)
        with self.assertRaisesRegex(ValueError,'D_NV'):
            _check_variant(plan,dict(dimensions,D_NV=39))

    def test_all_tasks_resolve_to_exact_precompiled_dimensions(self):
        from libero.gpu.host.build.substep_catalog import PARAMETERS, variants, select_variant
        tasks=json.loads(PARAMETERS.read_text())['tasks']
        self.assertEqual(len(tasks),130)
        self.assertEqual(len(variants()),13)
        for task in tasks:
            d=task['dimensions']
            v=select_variant(d['nv'],d['nM'],nq=d['nq'],njnt=d['njnt'],nu=d['nu'])
            self.assertEqual(v.quaternion_joints,d['nq']-d['nv'])
            self.assertEqual(v.rows,30000+d['nv']+d['njnt'])
        with self.assertRaises(ValueError): select_variant(39,149,nq=45)
        with self.assertRaises(ValueError): select_variant(39,149,rows=170)

    @unittest.skipUnless(shutil.which('g++'),'C++ compiler required')
    def test_every_instantiation_has_disjoint_shared_regions(self):
        from libero.gpu.host.build.substep_catalog import variants
        root=Path(__file__).resolve().parents[1]/'libero/libero/gpu/cuda/step'
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'layout.cpp';binary=Path(directory)/'layout'
            source.write_text('#include "newton_workspace.cuh"\n#include <iostream>\n'
                'int main(){using namespace flash_libero;std::cout << SubstepShape::nv << " "'
                '<< NewtonLayout::bytes << " " << NewtonLayout::mathdx_bytes;}\n')
            for v in variants():
                flags=[f'-DFLASH_SUBSTEP_{k}={value}' for k,value in dict(
                    NV=v.nv,NQ=v.nq,NM=v.nM,NJ=v.njnt,NU=v.nu,CONTACTS=v.contacts).items()]
                subprocess.run(['g++','-std=c++17','-I'+str(root),*flags,str(source),'-o',str(binary)],
                    check=True,capture_output=True,text=True)
                nv,shared,mathdx=map(int,subprocess.check_output([str(binary)],text=True).split())
                self.assertEqual(nv,v.nv)
                self.assertGreater(shared,mathdx)
                self.assertGreaterEqual(mathdx,(v.nv*v.nv+2*v.nv*v.hessian_tile_rows)*8)


if __name__=='__main__': unittest.main()
