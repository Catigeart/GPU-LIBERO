// Geometric regression for the actual FP32 task-0 face-selection tie.
// Also preserve first-vertex ownership at exact face/edge/corner boundaries.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "../../libero/libero/gpu/cuda/common/vector.cuh"
#include "../../libero/libero/gpu/cuda/collision/native/native_source.cuh"

struct Geometry { real_t size[6], pos[6], mat[18]; };
struct Result { int count, axis; real_t points[50][4]; };
static void check(cudaError_t code) {
  if (code != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(code)); std::exit(2); }
}
static std::vector<double> read_csv(const char *path) {
  std::ifstream input(path); std::vector<double> values;
  for (std::string line; std::getline(input, line);) {
    std::replace(line.begin(), line.end(), ',', ' '); std::istringstream row(line);
    for (double value; row >> value;) values.push_back(value);
  }
  return values;
}
__global__ void exercise(Geometry *geometry, Result *result) {
  const int query = blockIdx.y, width = int(blockIdx.y % 4);
  const int lanes = width == 0 ? 0 : (width == 1 ? 8 : (width == 2 ? 16 : 32));
  if (threadIdx.x >= (lanes ? lanes : 1)) return;
  Geometry &g = geometry[query / 4];
  int diag[16]{}, frame = 0;
  mjModel model{}; mjData data{}; mjContact contact[50]{};
  model.geom_size = g.size; model.adapter_diag = diag;
  model.adapter_frame0 = &frame; model.adapter_subgroup_width = lanes;
  data.geom_xpos = g.pos; data.geom_xmat = g.mat;
  const int count = mjc_BoxBox(&model, &data, contact, 0, 1, real_t(0));
  if (threadIdx.x == 0) {
    result[query].count = count; result[query].axis = diag[6];
    for (int j = 0; j < count; ++j) {
      result[query].points[j][0] = contact[j].dist;
      for (int k = 0; k < 3; ++k) result[query].points[j][k+1] = contact[j].pos[k];
    }
  }
}
int main(int argc, char **argv) {
  const auto values = read_csv(argc > 1 ? argv[1] : FLASH_BOX_CORNER_FIXTURE);
  if (values.size() != 30) { std::fputs("Invalid BoxBox fixture\n", stderr); return 2; }
  Geometry captured{};
  for (int i = 0; i < 6; ++i) { captured.size[i] = real_t(values[i]); captured.pos[i] = real_t(values[6+i]); }
  for (int i = 0; i < 18; ++i) captured.mat[i] = real_t(values[12+i]);
  std::vector<Geometry> cases; std::vector<int> expected; std::vector<double> expected_depth;
  auto add = [&](Geometry g, int count, double depth=std::numeric_limits<double>::quiet_NaN()) {
    cases.push_back(g); expected.push_back(count); expected_depth.push_back(depth);
    for (int k = 0; k < 3; ++k) { std::swap(g.size[k], g.size[3+k]); std::swap(g.pos[k], g.pos[3+k]); }
    for (int k = 0; k < 9; ++k) std::swap(g.mat[k], g.mat[9+k]);
    cases.push_back(g); expected.push_back(count); expected_depth.push_back(depth);
  };
  add(captured, 4);
  // The same initial physical configuration, with its actual per-product
  // transforms. Tiny FP32 projection errors must not switch to the edge path's
  // different signed-depth convention at a face/edge SAT tie.
  const auto tie = read_csv(argc > 2 ? argv[2] : FLASH_BOX_SAT_FIXTURE);
  if (tie.size()!=55) { std::fputs("Invalid SAT fixture\n",stderr); return 2; }
  Geometry sat{}; const int offset=FLASH_REAL_BITS==32 ? 6 : 30;
  for (int k=0;k<6;++k) { sat.size[k]=real_t(tie[k]); sat.pos[k]=real_t(tie[offset+k]); }
  for (int k=0;k<18;++k) sat.mat[k]=real_t(tie[offset+6+k]);
  for (double scale : {1e-3, 1.0, 1e3}) {
    Geometry scaled=sat;
    for(int k=0;k<6;++k) { scaled.size[k]*=real_t(scale); scaled.pos[k]*=real_t(scale); }
    add(scaled, 4, tie[54]*scale);
  }
  const double positions[][4] = {{0,0,1.9,4}, {0,0,2,4}, {2,0,2,2}, {2,2,2,1}, {0,0,2.2,0}};
  for (const auto &p : positions) {
    Geometry g{}; for (int k=0;k<6;++k) g.size[k]=real_t(1);
    for (int b=0;b<2;++b) for (int k=0;k<3;++k) g.mat[9*b+4*k]=real_t(1);
    for (int k=0;k<3;++k) g.pos[3+k]=real_t(p[k]);
    add(g, int(p[3]));
  }
  Geometry *device; Result *result;
  check(cudaMallocManaged(&device, cases.size()*sizeof(Geometry)));
  check(cudaMallocManaged(&result, cases.size()*4*sizeof(Result)));
  std::copy(cases.begin(), cases.end(), device);
  exercise<<<dim3(1, unsigned(cases.size()*4)), 32>>>(device, result);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  double worst = 0;
  for (size_t c=0;c<cases.size();++c) for (int w=0;w<4;++w) {
    const auto &g=cases[c]; const auto &r=result[c*4+w];
    if (r.count<0 || r.count>50) return 1;
    double scale=1; for (int k=0;k<6;++k) scale=std::max(scale, std::abs(double(g.pos[k]))+double(g.size[k]));
    const double tolerance=16*std::numeric_limits<float>::epsilon()*scale;
    std::vector<std::array<double,3>> unique;
    for (int j=0;j<r.count;++j) {
      std::array<double,3> p{};
      for (int k=0;k<4;++k) if (!std::isfinite(double(r.points[j][k]))) return 1;
      if (std::isfinite(expected_depth[c]) &&
          std::abs(double(r.points[j][0])-expected_depth[c])>tolerance) {
        std::fprintf(stderr,"SAT depth: case=%zu width=%d got=%g expected=%g\n",c,w,double(r.points[j][0]),expected_depth[c]); return 1;
      }
      for (int k=0;k<3;++k) p[k]=double(r.points[j][k+1]);
      for (int b=0;b<2;++b) for (int k=0;k<3;++k) {
        double local=0; for (int q=0;q<3;++q) local+=(p[q]-double(g.pos[3*b+q]))*double(g.mat[9*b+3*q+k]);
        const double violation=std::abs(local)-double(g.size[3*b+k]); worst=std::max(worst,violation);
        if (violation>tolerance) { std::fprintf(stderr,"Outside OBB: case=%zu width=%d error=%g\n",c,w,violation); return 1; }
      }
      bool duplicate=false;
      for (const auto &old:unique) { double error=0; for(int k=0;k<3;++k) error=std::max(error,std::abs(old[k]-p[k])); duplicate|=error<=tolerance; }
      if (!duplicate) unique.push_back(p);
    }
    if (int(unique.size())!=expected[c]) { std::fprintf(stderr,"Contact count: case=%zu width=%d unique=%zu expected=%d\n",c,w,unique.size(),expected[c]); return 1; }
  }
  check(cudaFree(device)); check(cudaFree(result));
  std::printf("real_bits=%d queries=%zu worst_obb_violation=%g\n", FLASH_REAL_BITS,cases.size()*4,worst);
}
