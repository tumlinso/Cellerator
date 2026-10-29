#include <Cellerator/modeling/cuda/dispatch.hh>

// Compile/link fixture for the optional CUDA dispatch primitives. It checks
// that the public plan view and all typed launch entrypoints are usable without
// selecting or launching a GPU; hardware execution requires scheduler admission.
int main() {
  using namespace cellerator::modeling::cuda;
  DispatchPlanView plan{};
  (void)&gather_fp16;
  (void)&gather_fp32;
  (void)&combine_fp16;
  (void)&combine_fp32;
  (void)&combine_backward_fp16;
  (void)&combine_backward_fp32;
  (void)&gather_backward_fp16;
  (void)&gather_backward_fp32;
  return static_cast<int>(plan.selection_count + plan.origin_count);
}
