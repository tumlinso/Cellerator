#pragma once

// Compatibility surface for pre-remap consumers. New code includes the
// canonical runtime headers directly.
#include <Cellerator/runtime/multi_gpu/fleet.cuh>

namespace cellerator::compute::runtime {

using namespace ::cellerator::runtime;

} // namespace cellerator::compute::runtime
