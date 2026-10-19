"""CPU-only checks for diagnostic parsing and bounded per-world records."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/runtime/newton/device_phase_profile.py'
spec = importlib.util.spec_from_file_location('newton_phase_profile_tested', MODULE)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileSourceTests(unittest.TestCase):
    def test_alternative_loop_headers_share_one_body(self):
        source = '''__device__ void sample(){
#if FIRST
for(int i=0;i<4;++i){
#else
for(int i=0;i<2;++i){
#endif
work(i);
}
}
__device__ void following(){}
'''
        start, brace, end = profile._function(source, 'sample')
        self.assertEqual(source[end:].strip(), '__device__ void following(){}')
        self.assertEqual(source[start:brace].strip(), '__device__ void sample()')

    def test_braces_in_comments_and_literals(self):
        source = '''__device__ int sample(){
// }
const char* text="}\\\"{"; /* } */
return 2;
}
NEXT'''
        self.assertEqual(source[profile._function(source, 'sample')[2]:].strip(), 'NEXT')

    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler required')
    def test_world_stride_leader_ownership_and_overflow(self):
        n = len(profile.PHASES) * 2
        store = profile.trace_store_source('values', 'trace', 'counts', 'metrics', 2, ['nefc[0]'])
        source = '''#include <cstdint>
#include <cassert>
struct {int x;} threadIdx;
struct Parameters{int64_t *values,*trace,*counts,*metrics;};
int main(){
''' + f'''
int64_t values[{2*n}], trace_guard[{4*n+2}], counts[2]={{0,0}}, metrics[4];
for(int i=0;i<{2*n};++i)values[i]=1000+i;
for(auto& v:trace_guard)v=-9;
Parameters params{{values,trace_guard+1,counts,metrics}};
auto* p=&params;
auto launch=[&](int w,int x){{threadIdx.x=x;int nefc[1]={{w+7}};{store}}};
for(int w=0;w<2;++w){{launch(w,0);launch(w,1);launch(w,0);launch(w,0);}}
assert(counts[0]==3&&counts[1]==3);
assert(trace_guard[0]==-9&&trace_guard[{4*n+1}]==-9);
for(int w=0;w<2;++w)for(int k=0;k<2;++k){{
assert(metrics[w*2+k]==w+7);
for(int i=0;i<{n};++i)assert(params.trace[(w*2+k)*{n}+i]==values[w*{n}+i]);
}}
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'trace.cpp').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', str(path/'trace.cpp'), '-o', str(path/'trace')], check=True, capture_output=True)
            subprocess.run([str(path/'trace')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
