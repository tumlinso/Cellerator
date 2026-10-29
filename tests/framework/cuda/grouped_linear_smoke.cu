#include <Cellerator/modeling/cuda/grouped_linear.hh>

#include <array>
#include <cassert>
#include <cmath>

namespace {
bool close(float a, float b) { return std::abs(a - b) < 1e-6f; }
}

// Host-side shape/oracle smoke only. Constructing GroupedLinear creates a
// cuBLAS handle, so exercising the GPU requires scheduler admission. This
// executable verifies the ragged reference equations without selecting CUDA.
int main() {
  using cellerator::modeling::cuda::GroupedLinear;
  using cellerator::modeling::cuda::GradientMode;
  [[maybe_unused]] auto forward = &GroupedLinear::forward;
  [[maybe_unused]] auto backward = &GroupedLinear::backward;

  // Three experts with row offsets [0, 2, 2, 3]: expert 1 is empty.
  const std::array<float, 4> x0{1, 1, 2, 0};
  const std::array<float, 4> w0{1, 2, 3, 4};
  const std::array<float, 4> y0{3, 7, 2, 6};
  const std::array<float, 2> x2{1, -1};
  const std::array<float, 4> w2{2, 1, -1, 2};
  const std::array<float, 2> y2{1, -3};
  const std::array<float, 4> dy0{1, 1, 1, 1};
  const std::array<float, 4> dx0{4, 6, 4, 6};
  const std::array<float, 4> dw0{3, 1, 3, 1};
  const std::array<float, 2> dy2{1, 2};
  const std::array<float, 2> dx2{0, 5};
  const std::array<float, 4> dw2{1, -1, 2, -2};

  // Assert the documented row-major products and derivative layout.
  assert(close(x0[0] * w0[0] + x0[1] * w0[1], y0[0]));
  assert(close(x0[0] * w0[2] + x0[1] * w0[3], y0[1]));
  assert(close(x0[2] * w0[0] + x0[3] * w0[1], y0[2]));
  assert(close(x0[2] * w0[2] + x0[3] * w0[3], y0[3]));
  assert(close(x2[0] * w2[0] + x2[1] * w2[1], y2[0]));
  assert(close(x2[0] * w2[2] + x2[1] * w2[3], y2[1]));
  assert(close(dy0[0] * w0[0] + dy0[1] * w0[2], dx0[0]));
  assert(close(dy0[0] * w0[1] + dy0[1] * w0[3], dx0[1]));
  assert(close(dy2[0] * w2[0] + dy2[1] * w2[2], dx2[0]));
  assert(close(dy2[0] * w2[1] + dy2[1] * w2[3], dx2[1]));
  assert(close(dw0[0], dy0[0] * x0[0] + dy0[2] * x0[2]));
  assert(close(dw0[1], dy0[0] * x0[1] + dy0[2] * x0[3]));
  assert(close(dw2[2], dy2[1] * x2[0]));
  assert(close(dw2[3], dy2[1] * x2[1]));
  [[maybe_unused]] const auto overwrite = GradientMode::overwrite;
  [[maybe_unused]] const auto accumulate = GradientMode::accumulate;
  return 0;
}
