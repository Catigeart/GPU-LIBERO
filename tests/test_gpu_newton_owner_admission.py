"""Run the actual ordered cone admission as host C++, including duplicate ids."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

GPU = Path(__file__).resolve().parents[1] / 'libero/libero/gpu'
spec = importlib.util.spec_from_file_location('phase_parser', GPU/'host/runtime/newton/device_phase_profile.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
SOURCE = GPU/'cuda/newton/constraint.cuh'


class OwnerAdmissionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler required')
    def test_ordered_admission_owns_last_middle_cone_without_numeric_writes(self):
        source = SOURCE.read_text()
        functions = []
        for name in ('cone_norm', 'constraint_update'):
            start, _, end = parser._function(source, name)
            functions.append(source[start:end])
        program = '''#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#define __device__
using std::isfinite;
'''+ '\n'.join(functions) + '''
int main(){
  const int rows=64,nv=2;int nc=4;
  for(int d: {1,3,4,6})for(int cached: {0,1})for(int pattern=0;pattern<81;++pattern){
    int active=1,n=4*d,eq=0,fc=0,flag=1,status=0,invariant=cached;
    std::vector<double> jar(rows,0),D(rows,2),R(rows,1),loss(rows,0),J(rows*nv,1),mu(nc,1),fr(nc*5,1);
    std::vector<int> dim(nc,d),typ(rows,7),cid(rows,0),state(rows,-1729),head(rows,-1729),owner(nc,-1),expected(nc,-1);
    std::vector<double> force(rows,-1729),H(nc*36,-1729);double cost=-1729;
    int p=pattern;
    for(int k=0;k<4;++k){
      int zone=p%3;p/=3;int r=k*d;
      // Alternate duplicate contact ids, including later top/bottom cones.
      int c=k%2;for(int j=0;j<d;++j)cid[r+j]=c;
      jar[r]=zone==0?2:(zone==1?-2:0);
      if(d>1)jar[r+1]=1;
      double T=cone_norm(jar.data()+r,fr.data()+c*5,d),N=jar[r];
      bool top=N>=T||(T<=0&&N>=0),bottom=N+T<=0||(T<=0&&N<0);
      if(!top&&!bottom)expected[c]=r;
    }
    constraint_update(0,&active,&n,&eq,&fc,&nc,&flag,jar.data(),D.data(),R.data(),loss.data(),typ.data(),cid.data(),J.data(),mu.data(),fr.data(),dim.data(),force.data(),state.data(),H.data(),&cost,&status,rows,nc,nv,&invariant,nullptr,nullptr,nullptr,nullptr,true,head.data(),owner.data(),false,true,true);
    assert(status==0&&invariant==1&&owner==expected&&cost==-1729);
    for(double v:force)assert(v==-1729);for(double v:H)assert(v==-1729);for(int v:state)assert(v==-1729);
    // Malformed cone values keep the original ordered admission error.
    if(d>1){jar[0]=0;jar[1]=1e-13;status=0;invariant=cached;
      constraint_update(0,&active,&n,&eq,&fc,&nc,&flag,jar.data(),D.data(),R.data(),loss.data(),typ.data(),cid.data(),J.data(),mu.data(),fr.data(),dim.data(),force.data(),state.data(),H.data(),&cost,&status,rows,nc,nv,&invariant,nullptr,nullptr,nullptr,nullptr,true,head.data(),owner.data(),false,true,true);
      assert(status==10);for(double v:force)assert(v==-1729);for(double v:H)assert(v==-1729);
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'owner.cpp').write_text(program)
            subprocess.run(['g++', '-O2', '-std=c++17', str(path/'owner.cpp'), '-o', str(path/'owner')], check=True, capture_output=True)
            subprocess.run([str(path/'owner')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
