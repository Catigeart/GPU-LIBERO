"""Exercise generated Newton publication extents without Warp or CUDA.

The real PUBLIC_FIELDS generator is extracted as an AST. Bytewise examples
compare its commit to the previous full commit under the existing writer/hold
contract. These tests do not execute Newton arithmetic or qualify GPU results.
"""
import ast
import copy
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
CORE = GPU / 'host/runtime/newton/handwritten_cuda.py'
ADAPTER = GPU / 'host/runtime/newton/runtime_adapter.py'
PUBLIC = ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'force', 'state',
          'solver_iter', 'solver_nnz', 'solver_stats_f', 'solver_stats_i')
CALL = re.compile(r'libero_newton_io::copy_block\((\w+),(\w+),([^;]+)\);')


def generated_publication(worlds, nv=43, rows=30061, profile="compatible"):
    """Execute only the current source generator, using plain owner metadata."""
    tree = ast.parse(CORE.read_text())
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Name) and node.iter.id == 'PUBLIC_FIELDS']
    if len(loops) != 1:
        raise AssertionError('exactly one PUBLIC_FIELDS generator required')
    wrappers = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'adapter_commit'
                        for t in node.targets)
                and isinstance(node.value, ast.List) and node.value.elts]
    if len(wrappers) != 1:
        raise AssertionError('exactly one valid-masked commit wrapper required')
    lengths = dict(qacc=nv, qacc_warmstart=nv, qfrc_constraint=nv,
                   force=rows, state=rows, solver_iter=1, solver_nnz=1,
                   solver_stats_f=3000, solver_stats_i=4000)
    owners = {name: SimpleNamespace(size=worlds * size, ptr=i + 1)
              for i, (name, size) in enumerate(lengths.items())}
    ws = SimpleNamespace(nworld=worlds, **owners,
                         nefc=SimpleNamespace(), iteration=SimpleNamespace())
    adapter = SimpleNamespace(_canonical={name: SimpleNamespace() for name in PUBLIC},
                              valid=SimpleNamespace())
    bind = lambda owner, key: key
    namespace = dict(PUBLIC_FIELDS=PUBLIC, ws=ws, adapter=adapter, bind=bind,
                     local=lambda owner, key: bind(getattr(owner, key), key),
                     adapter_begin=[], adapter_commit=[], seeded=set(),
                     diagnostic_profile=profile)
    nodes = [copy.deepcopy(loops[0]), copy.deepcopy(wrappers[0])]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CORE), 'exec'), namespace)
    return namespace, lengths


def copy_extents(source, n, iterations):
    """Evaluate only generated integer copy lengths, not C++ or solver code."""
    result = {}
    for src, dst, extent in CALL.findall(source):
        result[dst] = (src, eval(extent, {'__builtins__': {}, 'min': min, 'max': max},
                                 {'nefc': [n], 'iteration': [iterations]}))
    return result


class NewtonCommitExtentTests(unittest.TestCase):
    def test_generated_full_seed_and_small_fields_keep_their_extents(self):
        for worlds in (1, 2, 8, 64):
            with self.subTest(worlds=worlds):
                generated, sizes = generated_publication(worlds)
                seed = copy_extents('\n'.join(generated['adapter_begin']), -1, -1)
                self.assertEqual(seed, {name: ('canonical_' + name, size)
                                        for name, size in sizes.items()})
                commit = copy_extents(generated['adapter_commit'][0], 17, 4)
                for name in ('qacc', 'qacc_warmstart', 'qfrc_constraint',
                             'solver_iter', 'solver_nnz'):
                    self.assertEqual(commit['canonical_' + name], (name, sizes[name]))
                self.assertEqual(generated['seeded'], set(range(1, len(PUBLIC) + 1)))

    def test_fast_seed_is_bounded_and_omits_write_only_statistics(self):
        for worlds in (1, 2, 8, 64):
            generated, sizes = generated_publication(worlds, profile='fast')
            for n in (-2147483648, -1, 0, 1, 17, 30061, 30062, 2147483647):
                seed = copy_extents('\n'.join(generated['adapter_begin']), n, 0)
                for name in ('force', 'state'):
                    self.assertEqual(seed[name], ('canonical_' + name, max(0, min(n, sizes[name]))))
                for name in ('solver_stats_f', 'solver_stats_i'):
                    self.assertNotIn(name, seed)
                for name in ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'solver_iter', 'solver_nnz'):
                    self.assertEqual(seed[name], ('canonical_' + name, sizes[name]))
            self.assertEqual(generated['seeded'], set(range(1, len(PUBLIC) + 1)))

    def test_fast_seed_and_commit_hold_public_tails_with_dirty_private_scratch(self):
        generated, sizes = generated_publication(8, nv=7, rows=13, profile='fast')
        public = {name: [i * 37 + len(name) for i in range(size)] for name, size in sizes.items()}
        private = {name: [-1729] * size for name, size in sizes.items()}
        sequence = ((10, 7, 'success'), (2, 1, 'success'), (0, 0, 'empty'),
                    (5, 0, 'disabled'), (12, 3, 'failure'), (-1, 0, 'failure'),
                    (14, 0, 'failure'), (7, 0, 'success'), (13, 1001, 'success'))
        for step, (n, iterations, mode) in enumerate(sequence):
            # Poison private memory anew: absent seeds must not leak via held tails.
            for value in private.values():
                value[:] = [-1729-step] * len(value)
            full = {key: value.copy() for key, value in public.items()}
            for dst, (src, count) in copy_extents('\n'.join(generated['adapter_begin']), n, 0).items():
                private[dst][:count] = public[src.removeprefix('canonical_')][:count]
            dirty = dict(sizes)
            dirty.update(force=max(0, min(n, 13)), state=max(0, min(n, 13)),
                         solver_stats_f=3 * min(iterations, 1000), solver_stats_i=4 * min(iterations, 1000))
            if mode == 'empty':
                dirty = {name: sizes[name] for name in ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'solver_iter')}
            for name, count in dirty.items():
                for i in range(count):
                    private[name][i] = full[name][i] = step * 19 + i
            valid = mode in ('success', 'empty')
            expected = full if valid else {key: value.copy() for key, value in public.items()}
            if valid:
                for dst, (src, count) in copy_extents(generated['adapter_commit'][0], n, iterations).items():
                    public[dst.removeprefix('canonical_')][:count] = private[src][:count]
            self.assertEqual(public, expected)

    def test_dense_row_extent_is_nefc_and_stats_use_actual_record_count(self):
        generated, _ = generated_publication(8)
        for rows, iterations in ((0, 0), (1, 0), (23, 1), (30061, 100),
                                 (4, 999), (4, 1000), (4, 1001)):
            with self.subTest(rows=rows, iterations=iterations):
                result = copy_extents(generated['adapter_commit'][0], rows, iterations)
                self.assertEqual(result['canonical_force'][1], rows)
                self.assertEqual(result['canonical_state'][1], rows)
                self.assertEqual(result['canonical_solver_stats_f'][1], 3 * min(iterations, 1000))
                self.assertEqual(result['canonical_solver_stats_i'][1], 4 * min(iterations, 1000))
        self.assertNotIn('head', generated['adapter_commit'][0])
        self.assertTrue(generated['adapter_commit'][0].startswith('if(valid[0]){\n'))
        self.assertTrue(generated['adapter_commit'][0].endswith('\n}\n__syncthreads();'))

    def test_prefix_commit_matches_full_commit_across_holds_and_shrinking_prefixes(self):
        generated, sizes = generated_publication(8, nv=7, rows=13)
        itemsize = {name: 4 if name in ('state', 'solver_iter', 'solver_nnz',
                                      'solver_stats_i') else 8 for name in PUBLIC}
        public = {name: bytearray((i * 37 + len(name) * 13) & 255
                                  for i in range(size * itemsize[name]))
                  for name, size in sizes.items()}
        # Arbitrary bytes include NaN/sign-bit patterns; no floating point math
        # converts the held tails. A failure can dirty work but cannot publish.
        sequence = ((10, 7, 'success'), (2, 1, 'success'), (0, 0, 'empty'),
                    (5, 0, 'disabled'), (12, 3, 'failure'), (7, 0, 'success'),
                    (13, 100, 'success'), (1, 1001, 'success'))
        for step, (n, iterations, mode) in enumerate(sequence):
            with self.subTest(n=n, iterations=iterations, mode=mode):
                work = {name: bytearray(value) for name, value in public.items()}
                dirty = dict(sizes)
                dirty.update(force=n, state=n, solver_stats_f=3 * min(iterations, 1000),
                             solver_stats_i=4 * min(iterations, 1000))
                if mode == 'empty':
                    dirty = {name: sizes[name] for name in
                             ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'solver_iter')}
                for name, count in dirty.items():
                    for i in range(count * itemsize[name]):
                        work[name][i] = (i * 19 + step + 1) & 255
                valid = mode in ('success', 'empty')
                expected = {name: bytearray(work[name] if valid else value)
                            for name, value in public.items()}
                actual = {name: bytearray(value) for name, value in public.items()}
                if valid:
                    for dst, (src, count) in copy_extents(
                            generated['adapter_commit'][0], n, iterations).items():
                        name = dst.removeprefix('canonical_')
                        length = count * itemsize[name]
                        actual[name][:length] = work[src][:length]
                self.assertEqual(actual, expected)
                public = expected

    def test_reviewed_public_schema_and_count_proof_still_bound_the_generator(self):
        adapter_source = ADAPTER.read_text()
        tree = ast.parse(adapter_source)
        namespace = {}
        nodes = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ('OUTPUT_FIELDS', 'PUBLIC_FIELDS')
                         for t in node.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER), 'exec'), namespace)
        self.assertEqual(namespace['PUBLIC_FIELDS'], PUBLIC)
        source = CORE.read_text()
        self.assertIn("f'{local(ws,\"solver_iter_base\")}[0]=0;'", source)
        self.assertIn("for name in RESET_INT+RESET_FLOAT", source)
        self.assertIn("'iteration'", ast.get_source_segment(adapter_source, next(
            node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'RESET_INT' for t in node.targets))))
        # The profiling-only wrapper also joins this list during preparation.
        # Check the actual final kernel emission, after publication.
        self.assertLess(source.index("calls['tail']+adapter_commit"), source.rindex(r"'\n'.join(stores)"))


if __name__ == '__main__':
    unittest.main()
