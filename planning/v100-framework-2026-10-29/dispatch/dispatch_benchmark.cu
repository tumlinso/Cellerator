// Resident dispatch-backward microbenchmark. Compile once against a baseline
// dispatch CUDA object and once against a candidate, keeping this harness and
// all other library objects fixed. Invoke only in a Device Scheduler job.
#include <Cellerator/modeling/cuda/plans.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cellerator::modeling::cuda;

int main(int argc, char** argv) {
  const int features = argc > 1 ? std::atoi(argv[1]) : 32;
  constexpr std::int64_t origins = 64;
  constexpr std::int64_t selections = 256;
  constexpr int warmups = 100;
  constexpr int repetitions = 7;
  constexpr int iterations = 300;
  assert(features > 0);

  Stream stream(0);
  std::vector<std::int64_t> selection_origin(selections), origin_offsets(origins + 1),
      origin_dispatch;
  std::vector<float> weights(selections);
  origin_dispatch.reserve(selections);
  for (std::int64_t origin = 0; origin < origins; ++origin) {
    origin_offsets[origin] = origin_dispatch.size();
    for (std::int64_t k = 0; k < selections / origins; ++k) {
      const auto d = k * origins + origin;
      selection_origin[d] = origin;
      weights[d] = (k & 1) ? -0.125f : 0.25f;
      origin_dispatch.push_back(d);
    }
  }
  origin_offsets[origins] = origin_dispatch.size();
  DispatchPlan plan(origins, std::move(selection_origin), std::move(weights),
                    std::move(origin_offsets), std::move(origin_dispatch),
                    {0, selections}, stream);

  const auto origin_bytes = origins * features * sizeof(float);
  const auto selection_bytes = selections * features * sizeof(float);
  DeviceBuffer grad_origin(origin_bytes, stream.device());
  DeviceBuffer expert_output(selection_bytes, stream.device());
  DeviceBuffer grad_expert(selection_bytes, stream.device());
  DeviceBuffer grad_weight(selections * sizeof(float), stream.device());
  grad_origin.zero_async(stream);
  expert_output.zero_async(stream);
  stream.synchronize();

  for (int i = 0; i < warmups; ++i)
    plan.combine_backward(grad_origin.view(), expert_output.view(),
                          grad_expert.view(), grad_weight.view(), features,
                          DType::float32, stream);
  stream.synchronize();

  std::vector<float> milliseconds;
  milliseconds.reserve(repetitions);
  for (int sample = 0; sample < repetitions; ++sample) {
    Event start(stream.device(), true), stop(stream.device(), true);
    start.record(stream);
    for (int i = 0; i < iterations; ++i)
      plan.combine_backward(grad_origin.view(), expert_output.view(),
                            grad_expert.view(), grad_weight.view(), features,
                            DType::float32, stream);
    stop.record(stream);
    stop.synchronize();
    milliseconds.push_back(start.elapsed_ms(stop) / iterations);
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  std::cout << "device=0,sm=70,dtype=f32,origins=" << origins
            << ",selections=" << selections << ",features=" << features
            << ",warmups=" << warmups << ",iterations=" << iterations
            << ",median_us=" << milliseconds[milliseconds.size() / 2] * 1000.0f
            << ",min_us=" << milliseconds.front() * 1000.0f << '\n';
}
