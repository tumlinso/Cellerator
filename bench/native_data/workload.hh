#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace cellerator::bench::native_data {

// A host-owned, canonical fixture for the gated ordered-product operation.
// Index arrays retain incidence order and multiplicity. The benchmark packs
// these into each backend's local representation and charges that work.
struct Workload {
    std::string_view name;
    std::size_t batch = 1;
    std::size_t coordinates = 0;
    std::size_t conditions = 0;
    std::size_t parameters = 0;
    float baseline_scale = -0.25f;

    // Dense values are row-major: x[batch, coordinates],
    // z[batch, conditions, {z0,z1}], and upstream[batch, coordinates].
    std::vector<float> x;
    std::vector<float> z;
    std::vector<float> theta;
    std::vector<float> phi;
    std::vector<float> tau;
    std::vector<float> upstream;

    // Group ranges index the flattened incidence arrays below.
    std::vector<std::uint64_t> argument_begin;
    std::vector<std::uint64_t> argument_count;
    std::vector<std::uint64_t> output_begin;
    std::vector<std::uint64_t> output_count;
    std::vector<std::uint64_t> parameter_index;
    std::vector<std::uint64_t> condition_index;
    std::vector<std::uint64_t> argument_coordinate;
    std::vector<std::uint64_t> output_coordinate;
    std::vector<float> output_scale;

    // The selected shard owns each group's contribution. Two-device fixtures
    // use 0 or 1; the small reference fixture preserves its declared split.
    std::vector<int> group_owner;

    std::size_t groups() const noexcept { return argument_begin.size(); }
    std::uint64_t identity_hash() const noexcept;
    std::uint64_t quantized_x_hash() const noexcept;
};

Workload make_low_activity_disjoint();
Workload make_high_activity_overlap();
Workload make_reference_active();
Workload make_reference_inactive();

} // namespace cellerator::bench::native_data
