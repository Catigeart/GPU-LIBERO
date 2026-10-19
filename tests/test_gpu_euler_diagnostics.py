"""Euler diagnostic-profile source contracts; no CUDA imports or compilation.

These checks establish stage selection and retained handoff boundaries. They do
not execute Euler arithmetic or establish numerical equivalence on a GPU.
"""
from pathlib import Path
import re
import unittest


HEADER = (Path(__file__).resolve().parents[1]
          / 'libero/libero/gpu/cuda/integration/euler.cuh')


def function_body(source, name):
    start = source.index('__device__ void ' + name + '(')
    opening = source.index('{', start)
    depth = 1
    for index in range(opening + 1, len(source)):
        depth += (source[index] == '{') - (source[index] == '}')
        if depth == 0:
            return source[opening + 1:index]
    raise AssertionError('Unterminated function: ' + name)


def diagnostic_profile(body, enabled):
    """Project only the diagnostic conditionals in this function, without cpp."""
    included = [True]
    output = []
    for line in body.splitlines():
        directive = line.strip()
        if directive == '#if CF_EULER_DIAGNOSTICS':
            included.append(included[-1] and enabled)
        elif directive == '#endif':
            if len(included) == 1:
                raise AssertionError('Unmatched diagnostic #endif')
            included.pop()
        elif directive.startswith('#'):
            raise AssertionError('Unreviewed preprocessor condition: ' + directive)
        elif included[-1]:
            output.append(line)
    if len(included) != 1:
        raise AssertionError('Unterminated diagnostic conditional')
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', '\n'.join(output), flags=re.S)


def call(source, name):
    match = re.search(r'\b' + re.escape(name) + r'\s*\(', source)
    if match is None:
        raise AssertionError('Missing call: ' + name)
    opening = source.index('(', match.start())
    depth = 1
    for index in range(opening + 1, len(source)):
        depth += (source[index] == '(') - (source[index] == ')')
        if depth == 0:
            return source[match.start():index + 1]
    raise AssertionError('Unterminated call: ' + name)


class EulerDiagnosticsSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = function_body(HEADER.read_text(), 'evaluate_shared')
        cls.fast = diagnostic_profile(cls.body, False)
        cls.compatible = diagnostic_profile(cls.body, True)

    def test_replay_and_its_exports_are_compatible_only(self):
        self.assertNotRegex(self.fast, r'\breplay\s*\(')
        self.assertEqual(len(re.findall(r'\breplay\s*\(', self.compatible)), 1)
        for field, destination in (('zero_work', 'work'), ('zero_skip', 'skip')):
            export = ('libero_vec::copy_block(scratch.' + field + ','
                      + destination + '+w*nv,nv);')
            self.assertNotIn(export, self.fast)
            self.assertIn(export, self.compatible)

    def test_both_profiles_keep_the_numerical_stage_handoffs(self):
        stages = ('prepare_block', 'mass_block', 'integrate_block', 'commit_block')
        for profile in (self.fast, self.compatible):
            positions = [profile.index(call(profile, name)) for name in stages]
            self.assertEqual(positions, sorted(positions))
            for name in stages:
                self.assertEqual(len(re.findall(r'\b' + name + r'\s*\(', profile)), 1)
        for name in stages:
            self.assertEqual(call(self.fast, name), call(self.compatible, name))
        integration = call(self.fast, 'integrate_block')
        self.assertIn('scratch.solution,scratch.mass_status,scratch.damped', integration)
        self.assertNotIn('zero_work', integration)
        self.assertNotIn('zero_skip', integration)

    def test_fast_keeps_barriers_and_publication_evidence(self):
        self.assertEqual(self.fast.count('__syncthreads();'),
                         self.compatible.count('__syncthreads();'))
        mass_end = self.fast.index(call(self.fast, 'mass_block')) + len(call(self.fast, 'mass_block'))
        integrate_begin = self.fast.index(call(self.fast, 'integrate_block'))
        self.assertIn('__syncthreads();', self.fast[mass_end:integrate_begin])
        for retained in (
                'status[w]=scratch.status[0];ms[w]=scratch.mass_status[0];',
                'wc[w]=scratch.warning_count[0];wl[w]=scratch.warning_last[0];',
                'libero_vec::copy_block(scratch.used,used+w*nv,nv);'):
            self.assertIn(retained, self.fast)
            self.assertIn(retained, self.compatible)
        # Final position validation and the existing transaction remain ahead
        # of scalar status/warning and used-acceleration publication.
        self.assertLess(self.fast.index('isfinite(scratch.pos[i])'),
                        self.fast.index(call(self.fast, 'commit_block')))
        self.assertLess(self.fast.index(call(self.fast, 'commit_block')),
                        self.fast.index('status[w]=scratch.status[0]'))


if __name__ == '__main__':
    unittest.main()
