"""Cold ordinal-head ABI and source contracts; no GPU or native compilation.

These tests execute only the stdlib source generator. CUDA gate semantics still
require the separately authorized compiled numerical qualification.
"""
import ast
from pathlib import Path
import re
import runpy
from types import SimpleNamespace
import unittest


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
HOST = GPU / 'host/runtime/newton/handwritten_cuda.py'
CUDA = GPU / 'cuda/newton/constraint.cuh'


def load_source_helper():
    tree = ast.parse(HOST.read_text())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_head_metadata_source')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HOST), 'exec'), namespace)
    return namespace['_head_metadata_source']


class Array:
    def __init__(self, ptr, size=8, itemsize=8, shape=None):
        self.ptr, self.size = ptr, size
        self.strides = (itemsize,)
        self.shape = (size,) if shape is None else shape
        self.dtype = 'float64' if itemsize == 8 else 'int32'
        self.device = 'cuda:0'
        self.is_contiguous = True


class Descriptor(dict):
    def __init__(self, ws):
        self.nworld, self.rows, self.contacts, self.nv = ws.nworld, ws.rows, ws.contacts, ws.qacc.shape[1]
        self.world_strides = dict(source_counts=5, source_ncon=1, compact_heads=ws.rows,
            compact_count=1, compact_valid=1, head_type=ws.rows, head_id=ws.rows,
            head_dim=ws.rows, head_mu=ws.rows, head_friction=5*ws.rows, valid=1)
        super().__init__((key, Array((index+100)*65536, ws.nworld*pitch))
                         for index, (key, pitch) in enumerate(self.world_strides.items()))
        self.validation_count = 0
        self.invalid = False

    def validate(self):
        self.validation_count += 1
        if self.invalid:
            raise ValueError('descriptor seal changed')


class NewtonHeadMetadataTests(unittest.TestCase):
    def setUp(self):
        self.generate = load_source_helper()
        self.ws = SimpleNamespace(nworld=8, rows=19, contacts=7, device='cuda:0',
                                  qacc=Array(65536, 8*43, shape=(8,43)))
        keys = ('active','status','count','nefc','ne','nf','ncon','J','aref','D','R',
                'loss','efc_type','efc_id','cone_dim','cone_mu','cone_friction','qM',
                'qfrc_smooth','qacc_smooth','contact_dim','contact_mu','contact_friction')
        for index, key in enumerate(keys):
            setattr(self.ws, key, Array((index+2)*65536, 8*19))
        self.heads = {key: Array((index+50)*65536, 8*19)
                      for index, key in enumerate(('group_head','head_rows','head_count'))}
        self.metadata = Descriptor(self.ws)
        self.bound = []
        def bind(value, name, world_stride=None):
            self.bound.append((value.ptr, name, world_stride))
            return name
        self.bind = bind
        self.options = dict(diagnostic_profile='fast',
                            execution_config=SimpleNamespace(strategy='single_cta'),
                            has_blas=True, combined_entry=True)

    def emit(self, metadata=None, mutable=(), **overrides):
        return self.generate(self.metadata if metadata is None else metadata,
                             self.bind, self.ws, self.heads, mutable,
                             **dict(self.options, **overrides))

    def test_default_has_no_metadata_abi_or_gate(self):
        result = self.generate(None, self.bind, self.ws, self.heads, (), **self.options)
        self.assertFalse(result['enabled'])
        self.assertEqual(result['arguments'], ['nullptr']*3)
        self.assertEqual(result['gate'], '')
        self.assertEqual(result['shared'], '')
        self.assertEqual(self.bound, [])

    def test_generated_gate_matches_cuda_parameter_order_and_world_pitches(self):
        result = self.emit()
        self.assertTrue(result['enabled'])
        self.assertEqual(self.metadata.validation_count, 1)
        match = re.search(r'void constraint_head_metadata_gate\((.*?)\)\s*\{', CUDA.read_text(), re.S)
        parameters = [field.split()[-1].lstrip('*') for field in match[1].split(',')]
        arguments = result['gate'].removeprefix('constraint_head_metadata_gate(').removesuffix(');').split(',')
        expected = dict(active='active', status='status', invariant_valid='&invariant_valid',
            count='count', nefc='nefc', ne='ne', nf='nf', ncon='ncon', typ='efc_type',
            cid='efc_id', dim='contact_dim', mu='contact_mu', fr='contact_friction',
            row_dim='cone_dim', row_mu='cone_mu', row_fr='cone_friction',
            group_head='group_head', head_rows='head_rows', head_count='head_count',
            head_valid='ordinal_valid', selected='&ordinal_head_selected',
            rows='ROWS', contacts='7', nv='NV')
        for key in self.metadata.world_strides:
            if key != 'valid':expected[key] = 'ordinal_'+key
        self.assertEqual(arguments, [expected[key] for key in parameters])
        for key, pitch in self.metadata.world_strides.items():
            self.assertIn((self.metadata[key].ptr, 'ordinal_'+key, pitch), self.bound)
        self.assertEqual(result['arguments'], [
            '(ordinal_head_selected?ordinal_'+key+':nullptr)'
            for key in ('head_dim','head_mu','head_friction')])
        self.assertIn('if(threadIdx.x==0)ordinal_head_selected=0;', result['shared'])

    def test_explicit_candidate_rejects_unqualified_scope(self):
        for worlds in (1,2,64):
            with self.subTest(worlds=worlds):
                self.ws.nworld = worlds
                with self.assertRaisesRegex(ValueError, 'fast N=8 single_cta'):
                    self.emit()
        self.ws.nworld = 8
        for options in (dict(diagnostic_profile='compatible'), dict(has_blas=False),
                        dict(combined_entry=False),
                        dict(execution_config=SimpleNamespace(strategy='multi_cta'))):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'fast N=8 single_cta'):
                self.emit(**options)

    def test_descriptor_drift_is_rejected_before_pointer_binding(self):
        self.metadata.invalid = True
        with self.assertRaisesRegex(ValueError, 'seal changed'):
            self.emit()
        self.assertEqual(self.bound, [])
        self.metadata.invalid = False
        self.metadata.world_strides['head_friction'] = self.ws.rows
        with self.assertRaisesRegex(ValueError, 'world pitches differ'):
            self.emit()
        self.assertEqual(self.bound, [])

    def test_foreign_device_is_rejected_before_pointer_binding(self):
        for value in self.metadata.values():
            value.device = 'cuda:1'
        with self.assertRaisesRegex(ValueError, 'device differs from Newton'):
            self.emit()
        self.assertEqual(self.bound, [])

    def test_sealed_descriptor_module_connects_without_runtime_imports_or_views(self):
        descriptor_class = runpy.run_path(str(GPU/'host/runtime/newton/head_metadata.py'))['HeadMetadata']
        arrays = {key: Array(value.ptr, value.size,
                             itemsize=8 if key in ('head_mu','head_friction') else 4)
                  for key, value in self.metadata.items()}
        descriptor = descriptor_class.create(owner=object(), nworld=8, rows=19,
            contacts=7, nv=43, provider=lambda: arrays, validate_owner=lambda: None,
            array_type=Array, int32_dtype='int32', float64_dtype='float64')
        result = self.emit(metadata=descriptor)
        self.assertTrue(result['enabled'])
        self.assertIs(descriptor['head_mu'], arrays['head_mu'])
        arrays['head_mu'].ptr += 8
        with self.assertRaisesRegex(ValueError, 'owner/layout changed'):
            self.emit(metadata=descriptor)

    def test_mutable_alias_uses_original_fallback_without_new_admission_error(self):
        for operand in (self.metadata['head_mu'], self.metadata['head_friction'],
                        self.ws.contact_mu, self.ws.cone_friction, self.ws.J):
            for shift in (0,8):
                with self.subTest(ptr=operand.ptr, shift=shift):
                    result = self.emit(mutable=[Array(operand.ptr+shift, 2)])
                    self.assertFalse(result['enabled'])
                    self.assertEqual(result['arguments'], ['nullptr']*3)
                    self.assertEqual(result['gate'], '')
                    self.assertIn('overlaps', result['fallback_reason'])
        self.assertEqual(self.bound, [])

    def test_gate_is_advisory_and_checks_current_topology_and_both_numeric_views(self):
        source = CUDA.read_text()
        begin = source.index('__device__ void constraint_head_metadata_gate(')
        end = source.index('// All threads enter this collective.', begin)
        gate = source[begin:end]
        self.assertLess(gate.index('selected[0]=0;'), gate.index('active[0]!=1'))
        self.assertLess(gate.index('if(next!=n||limits!=source_counts[2])return;'),
                        gate.index('selected[0]=1;'))
        for field in ('status','active','invariant_valid','compact_valid','head_valid'):
            self.assertNotRegex(gate, r'\b'+field+r'\s*\[[^]]+\]\s*=(?!=)')
        for check in ('nefc[0]!=n', 'compact_count[0]!=nh', 'compact_heads[k]!=r',
                      'group_head[r]!=r', 'head_type[k]!=t', 'head_id[k]!=c',
                      'head_dim[k]!=d||row_dim[r]!=d',
                      'm_bits!=__double_as_longlong(mu[c])||m_bits!=__double_as_longlong(row_mu[r])',
                      'f_bits!=__double_as_longlong(fr[c*5+j])||f_bits!=__double_as_longlong(row_fr[r*5+j])',
                      'group_head[r+j]!=-1||typ[r+j]!=t||cid[r+j]!=c'):
            self.assertIn(check, gate)
        # Raw/S flags are necessary hints, never a substitute for those current
        # content comparisons or the completed ordinary N validation.
        self.assertIn('invariant_valid[0]!=1', gate)
        self.assertIn('head_friction[j*rows+k]', gate)

    def test_first_update_and_callback_boundary_are_retained(self):
        source = HOST.read_text()
        self.assertNotIn("head_metadata=(external_prefix or {}).get('newton_head_metadata')", source)
        self.assertIn('head_gate_inserted=False', source)
        first = source.index("calls[mode].append(code+'\\n__syncthreads();')")
        late = source.index("calls[mode].append(head_source['gate']+'\\n__syncthreads();')")
        self.assertLess(first, late)
        self.assertIn("if mode!='prefix':raise ValueError", source[first:late])
        self.assertIn('self.external_binding_validators+=(head_metadata.validate,)', source)
        self.assertIn('(4 if head_metadata is not None else 0)', source)


if __name__ == '__main__':
    unittest.main()
