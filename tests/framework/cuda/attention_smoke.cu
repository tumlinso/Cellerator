#include <Cellerator/modeling/cuda/attention.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {
// Independent host oracle for the contract's small sparse relation. This is
// not a CUDA launch; scheduler admission is required for device execution.
void host_reference() {
  const std::vector<float> q{1, 0, 0, 1};
  const std::vector<float> k{1, 0, 0, 1, 1, 1};
  const std::vector<float> v{2, 1, 0, 3, 4, 2};
  const std::vector<int> offsets{0, 2, 3}, keys{0, 2, 1};
  const float scale = 0.5f;
  std::vector<float> output(4, 0.0f);
  for (int row=0; row<2; ++row) {
    std::vector<float> logits;
    for (int e=offsets[row]; e<offsets[row+1]; ++e) {
      float dot=0;
      for (int d=0; d<2; ++d) dot += q[row*2+d]*k[keys[e]*2+d];
      logits.push_back(dot*scale);
    }
    float maximum=*std::max_element(logits.begin(),logits.end()), denominator=0;
    for (auto& x:logits) { x=std::exp(x-maximum); denominator+=x; }
    for (auto& x:logits) x/=denominator;
    for (int j=0; j<2; ++j)
      for (int e=offsets[row]; e<offsets[row+1]; ++e)
        output[row*2+j] += logits[e-offsets[row]]*v[keys[e]*2+j];
  }
  assert(std::abs(output[0] - 3.0f) < 1e-5f);
  assert(std::abs(output[1] - 1.5f) < 1e-5f);
  assert(output[2] == 0.0f && output[3] == 3.0f); // second query has one relation
}
}

int main() {
  using namespace cellerator::modeling::cuda;
  (void)&sparse_attention_fp16;
  (void)&sparse_attention_fp32;
  (void)&sparse_attention_fused_small_fp16;
  (void)&sparse_attention_fused_small_fp32;
  (void)&sparse_attention_backward_fp16;
  (void)&sparse_attention_backward_fp32;
  SparseAttentionPlan plan{};
  SparseAttentionWorkspace workspace{};
  host_reference();
  return static_cast<int>(plan.query_count + workspace.logits_capacity);
}
