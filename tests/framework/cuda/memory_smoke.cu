#include <Cellerator/modeling/cuda/memory.hh>
#include <Cellerator/modeling/event_memory.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

// Host reference smoke plus CUDA compile/link fixture. No device is selected
// and no kernel is launched; execution of CUDA paths requires scheduler admission.
int main() {
  using namespace cellerator::modeling;
  using namespace cellerator::modeling::cuda;
  (void)&apply_events_fp16;
  (void)&apply_events_fp32;
  (void)&read_traces_fp16;
  (void)&read_traces_fp32;

  TraceBank<float> traces(EntityLayout({7, 9}), {2.0, 4.0});
  traces.apply({{7, 0, 1.0, 2.0, 100}, {7, 1, 2.0, 4.0, 101}}, 2.0);
  const auto at_two = traces.read(7, 2.0);
  assert(std::abs(at_two[0] - 2.0 * std::exp(-0.5)) < 1e-6);
  assert(std::abs(at_two[1] - 4.0) < 1e-6);
  traces.apply({{7, 0, 3.0, 1.0, 102}}, 3.0);
  const auto at_three = traces.read(7, 3.0);
  assert(std::abs(at_three[0] - (at_two[0] * std::exp(-0.5) + 1.0)) < 1e-6);
  assert(std::abs(at_three[1] - (4.0 * std::exp(-0.25))) < 1e-6);
  assert(traces.anchor_times()[traces.layout().slot(9)] == 0.0);  // untouched entity stays lazy
  return 0;
}
