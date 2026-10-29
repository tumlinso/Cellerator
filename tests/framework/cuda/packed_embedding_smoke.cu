#include <Cellerator/modeling/cuda/packed_embedding.hh>

#include <array>
#include <cassert>
#include <cmath>

namespace {
bool close(float a, float b) { return std::abs(a - b) < 1e-6f; }
}

// Host reference smoke; no CUDA device is selected and no GPU launch occurs.
int main() {
  using namespace cellerator::modeling::cuda;
  [[maybe_unused]] auto forward = &packed_embedding_bag_forward;
  [[maybe_unused]] auto backward = &packed_embedding_bag_backward;

  // One profile feature with levels 0/1 and two learned 2-D embeddings.
  // The exact value 0.5 interpolates the two rows to [0.5, 1].
  const std::array<float, 4> table{1, 0, 0, 2};
  const std::array<std::uint32_t, 3> occurrence_indices{0, 1, 1};
  const std::array<float, 3> occurrence_weights{1, 2, 1};
  const std::array<std::uint32_t, 3> segments{0, 2, 3};
  const std::array<float, 4> expected_output{2, 2, 0.5f, 1};
  for (std::size_t d = 0; d < 2; ++d) {
    const float exact_embedding = (table[d] + table[2 + d]) * 0.5f;
    const float bag0 = table[d] + occurrence_weights[1] * exact_embedding;
    const float bag1 = occurrence_weights[2] * exact_embedding;
    assert(close(bag0, expected_output[d]));
    assert(close(bag1, expected_output[2 + d]));
  }

  // dY=[1,1] for bag0 and [2,2] for bag1. Code-row gradients are additive,
  // split evenly for exact-value interpolation; weight gradients are dots.
  const std::array<float, 3> expected_weight_gradients{1, 1.5f, 3};
  assert(close(1.0f * table[0] + 1.0f * table[1], expected_weight_gradients[0]));
  assert(close(1.0f * expected_output[2] + 1.0f * expected_output[3], expected_weight_gradients[1]));
  assert(close(2.0f * expected_output[2] + 2.0f * expected_output[3], expected_weight_gradients[2]));
  // Expected table gradient by code row, flattened [code, dim].
  const std::array<float, 4> expected_table_grad{3, 3, 2, 2};
  assert(close(expected_table_grad[0], 1 + 0.5f * 2 + 0.5f * 2));
  assert(close(expected_table_grad[2], 0.5f * 2 + 0.5f * 2));
  (void)occurrence_indices;
  (void)segments;
  return 0;
}
