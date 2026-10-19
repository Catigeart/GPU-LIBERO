// Two-world OSC against fixtures from the installed CPU reference. Exercise
// full rank, rank deficiency, scale changes, zero Jacobians and cache refresh.
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../libero/libero/gpu/cuda/control/osc_solver.cuh"
#include "../../libero/libero/gpu/cuda/step/substep_types.cuh"

using flash_libero::ControllerSolveData;
struct Field {
  const char *name;
  size_t offset;
  int width;
};
#define FIELD(name, width)                                                                                   \
  Field { #name, offsetof(ControllerSolveData, name), width }
static const Field doubles[] = {FIELD(mass, 49),
                                FIELD(jacobian, 42),
                                FIELD(q, 7),
                                FIELD(qvel, 7),
                                FIELD(bias, 7),
                                FIELD(q0, 7),
                                FIELD(ee_pos, 3),
                                FIELD(ee_rot, 9),
                                FIELD(ee_velocity, 6),
                                FIELD(action, 7),
                                FIELD(goal_pos, 3),
                                FIELD(goal_rot, 9),
                                FIELD(gripper, 2),
                                FIELD(gains, 12),
                                FIELD(range, 18),
                                FIELD(ctrl, 9),
                                FIELD(raw_tau, 7),
                                FIELD(mass_inv, 49),
                                FIELD(A_full, 36),
                                FIELD(A_pos, 9),
                                FIELD(A_ori, 9),
                                FIELD(lambda_full, 36),
                                FIELD(lambda_pos, 9),
                                FIELD(lambda_ori, 9),
                                FIELD(nullspace_matrix, 49),
                                FIELD(position_error, 3),
                                FIELD(orientation_error, 3),
                                FIELD(quat_input, 4),
                                FIELD(quat_result, 9),
                                FIELD(nullspace_torque, 7),
                                FIELD(clipped, 7),
                                FIELD(derived_key, 91),
                                FIELD(derived_data, 206),
                                FIELD(pm, 49),
                                FIELD(pj, 42),
                                FIELD(pq, 7),
                                FIELD(pv, 7),
                                FIELD(pp, 3),
                                FIELD(pr, 9),
                                FIELD(pe, 6)};
static const Field integers[] = {FIELD(active, 1),        FIELD(error, 1),         FIELD(ranks, 3),
                                 FIELD(policy, 1),        FIELD(needs, 1),         FIELD(quat_valid, 1),
                                 FIELD(derived_valid, 1), FIELD(derived_ranks, 3), FIELD(cache_hits, 1),
                                 FIELD(cache_valid, 1)};
static const Field inputs[] = {FIELD(pm, 49),   FIELD(pj, 42),      FIELD(pq, 7),       FIELD(pv, 7),
                               FIELD(bias, 7),  FIELD(q0, 7),       FIELD(pp, 3),       FIELD(pr, 9),
                               FIELD(pe, 6),    FIELD(goal_pos, 3), FIELD(goal_rot, 9), FIELD(gains, 12),
                               FIELD(range, 18)};
static const Field outputs[] = {FIELD(mass_inv, 49),
                                FIELD(A_full, 36),
                                FIELD(A_pos, 9),
                                FIELD(A_ori, 9),
                                FIELD(lambda_full, 36),
                                FIELD(lambda_pos, 9),
                                FIELD(lambda_ori, 9),
                                FIELD(nullspace_matrix, 49),
                                FIELD(position_error, 3),
                                FIELD(orientation_error, 3),
                                FIELD(nullspace_torque, 7),
                                FIELD(raw_tau, 7),
                                FIELD(clipped, 7),
                                FIELD(ctrl, 9),
                                FIELD(goal_pos, 3),
                                FIELD(goal_rot, 9)};
#undef FIELD

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}
static void read(FILE *file, void *data, size_t size) {
  if (std::fread(data, 1, size, file) != size) {
    std::fputs("Truncated CPU fixture\n", stderr);
    std::exit(1);
  }
}
template <class T> static T *&view(ControllerSolveData &data, size_t offset) {
  return *reinterpret_cast<T **>(reinterpret_cast<char *>(&data) + offset);
}
__global__ void exercise(ControllerSolveData inputs) {
  extern __shared__ __align__(32) unsigned char shared[];
  physics_controller::solve(inputs, blockIdx.y, shared);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fputs("Usage: flash_osc_reference_test CPU-fixture.bin\n", stderr);
    return 1;
  }
  FILE *file = std::fopen(argv[1], "rb");
  if (!file) {
    std::perror(argv[1]);
    return 1;
  }
  uint64_t magic;
  uint32_t cases, worlds;
  read(file, &magic, sizeof(magic));
  read(file, &cases, sizeof(cases));
  read(file, &worlds, sizeof(worlds));
  if (magic != 0x4f53435f4e325631ull || cases != 4 || worlds != 2)
    return 1;
  ControllerSolveData data{};
  std::vector<void *> owners;
  for (const auto &f : doubles) {
    auto &p = view<double>(data, f.offset);
    check(cudaMallocManaged(&p, 2 * f.width * sizeof(double)));
    owners.push_back(p);
  }
  for (const auto &f : integers) {
    auto &p = view<int>(data, f.offset);
    check(cudaMallocManaged(&p, 2 * f.width * sizeof(int)));
    owners.push_back(p);
  }
  check(cudaMallocManaged(&data.quat_cast, 8 * sizeof(float)));
  owners.push_back(data.quat_cast);
  data.diagnostics = data.uncouple = 1;
  double largest_relative_error = 0.0;
  for (unsigned c = 0; c < cases; ++c) {
    for (const auto &f : doubles)
      for (int i = 0; i < 2 * f.width; ++i)
        view<double>(data, f.offset)[i] = 0.;
    for (const auto &f : integers)
      for (int i = 0; i < 2 * f.width; ++i)
        view<int>(data, f.offset)[i] = 0;
    for (const auto &f : inputs)
      read(file, view<double>(data, f.offset), 2 * f.width * sizeof(double));
    data.active[0] = data.active[1] = data.needs[0] = data.needs[1] = 1;
    for (int pass = 0; pass < 3; ++pass) {
      // The third call changes only world one's physical mass. World zero
      // must retain its exact-input cache hit while world one recomputes.
      if (pass == 2)
        data.pm[49] += .125;
      exercise<<<dim3(1, 2), flash_libero::substep_threads, flash_libero::OscLayout::bytes>>>(data);
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      for (const auto &f : outputs) {
        std::vector<double> expected(2 * f.width);
        read(file, expected.data(), expected.size() * sizeof(double));
        const auto *actual = view<double>(data, f.offset);
        for (size_t i = 0; i < expected.size(); ++i) {
          const double error = std::abs(actual[i] - expected[i]);
          largest_relative_error = std::fmax(largest_relative_error, error / (1 + std::abs(expected[i])));
          if (!std::isfinite(actual[i]) || error > 1e-8 * (1 + std::abs(expected[i]))) {
            std::fprintf(stderr, "case=%u pass=%d %s[%zu]: actual=%.17g CPU=%.17g\n", c, pass, f.name, i,
                         actual[i], expected[i]);
            return 1;
          }
        }
      }
      int expected_ranks[6];
      read(file, expected_ranks, sizeof(expected_ranks));
      for (int i = 0; i < 6; ++i)
        if (data.ranks[i] != expected_ranks[i])
          return 1;
      for (int w = 0; w < 2; ++w) {
        const int hits = pass == 0 ? 0 : (pass == 1 ? 1 : (w == 0 ? 2 : 1));
        if (data.error[w] || data.cache_hits[w] != hits || !data.cache_valid[w] || data.needs[w] != 1)
          return 1;
      }
    }
  }
  if (std::fgetc(file) != EOF)
    return 1;
  std::fclose(file);
  for (auto *p : owners)
    check(cudaFree(p));
  std::printf("OSC CPU reference: 12 two-world calls passed; max scaled error %.3g\n",
              largest_relative_error);
}
