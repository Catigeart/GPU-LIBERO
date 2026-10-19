"""Execute the CUDA metadata helper as host C++ over random valid row groups."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

GPU=Path(__file__).resolve().parents[1]/'libero/libero/gpu'
spec=importlib.util.spec_from_file_location('phase_parser',GPU/'host/runtime/newton/device_phase_profile.py')
parser=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


class TileHeadTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'host C++ compiler required')
    def test_live_offsets_match_ordered_scan_and_hold_unused_tail(self):
        source=(GPU/'cuda/newton/constraint.cuh').read_text()
        start,_,end=parser._function(source,'constraint_tile_head_offsets_block')
        helper=source[start:end]
        program='''#include <vector>
#include <random>
#include <cassert>
#define __device__
struct {int x;} threadIdx,blockDim;
'''+helper+'''
int main(){
std::mt19937 random(1729);
for(int trial=0;trial<1000;++trial){
  int n=trial<40?trial:int(random()%30054),eq=n>20?7:0,fc=n>20?9:0;
  std::vector<int> typ(n,0),cid(n,0),dim(n+1,1),heads;
  for(int r=0;r<n;){
    int d=1;
    if(r>=eq+fc){int dims[4]={1,3,4,6};d=dims[random()%4];if(r+d>n)d=1;}
    heads.push_back(r);dim[r]=d;
    if(r>=eq+fc){for(int j=0;j<d;++j){typ[r+j]=7;cid[r+j]=r;}}
    r+=d;
  }
  int tiles=(n+15)/16;
  std::vector<int> expected(tiles+1),actual(tiles+19,-1729);
  int k=0;
  for(int t=0;t<=tiles;++t){
    while(k<int(heads.size())){int r=heads[k],d=r>=eq+fc&&typ[r]==7?dim[cid[r]]:1;
      if((r+d-1)/16>=t)break;++k;}
    expected[t]=k;
  }
  for(int threads: {32,64,128}){
    std::fill(actual.begin(),actual.end(),-1729);blockDim.x=threads;
    for(int lane=0;lane<threads;++lane){threadIdx.x=lane;
      constraint_tile_head_offsets_block(n,eq,fc,0,0,typ.data(),cid.data(),dim.data(),heads.data(),int(heads.size()),actual.data());}
    for(int t=0;t<=tiles;++t)assert(actual[t]==expected[t]);
    for(int t=tiles+1;t<int(actual.size());++t)assert(actual[t]==-1729);
  }
}
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'heads.cpp').write_text(program)
            subprocess.run(['g++','-O2','-std=c++17',str(path/'heads.cpp'),'-o',str(path/'heads')],check=True,capture_output=True)
            subprocess.run([str(path/'heads')],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
