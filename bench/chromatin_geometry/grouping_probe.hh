#pragma once

#include "workloads.hh"
#include <array>
#include <stdexcept>

namespace cgs {
// Separate from the frozen manifest: same seed, gate topology and numerical
// maps as its coherent 64-group case, but no value changes or learning.
inline std::array<WorkloadSpec, 2> grouping_probe_specs() {
    const auto& manifest = workload_manifest();
    const char* base = "dag7-64root-complete-consumer-coherent-dense-high-update-r16-change1";
    for (const auto& s : manifest) if (s.id == base) {
        auto one = s;
        one.id = "probe-grouping-coherent-hard-frozen-r1-change0";
        one.parameter_pattern = ParameterPattern::frozen;
        one.predicate_change_percent = 0;
        one.reuse_horizon = 1;
        auto sixteen = one;
        sixteen.id = "probe-grouping-coherent-hard-frozen-r16-change0";
        sixteen.reuse_horizon = 16;
        return {one, sixteen};
    }
    throw std::logic_error("grouping probe base case missing from frozen manifest");
}
} // namespace cgs
