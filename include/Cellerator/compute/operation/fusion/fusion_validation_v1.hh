#pragma once

#include <Cellerator/compute/operation/fusion/prepared_stage_graph_v1.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::operation::fusion {

struct registry_entry_v1 {
    std::uint64_t stable_candidate_id = 0u;
    const char *unique_name = nullptr;
    composition_kind_v1 composition =
        composition_kind_v1::value_generation_to_pack;
    bool fused = false;
    bool experimental = true;
    bool requires_measurement = true;
    bool explicitly_selectable = true;
    bool auto_promoted = false;
    bool unfused_stages_available = true;
    bool exact = true;
};

struct equivalence_request_v1 {
    const float *unfused_output = nullptr;
    const float *fused_output = nullptr;
    std::uint64_t global_output_begin = 0u;
    std::uint32_t local_output_count = 0u;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
};

struct equivalence_result_v1 {
    std::uint64_t checked_output_count = 0u;
    std::uint64_t first_failing_global_output = 0u;
    double maximum_absolute_error = 0.0;
    double maximum_relative_error = 0.0;
    bool exact_match = false;
    bool within_tolerance = false;
};

const registry_entry_v1 *fusion_registry_v1(std::size_t *count) noexcept;
status_v1 validate_fusion_registry_v1() noexcept;
status_v1 validate_fused_unfused_equivalence_v1(
    const equivalence_request_v1 &request,
    equivalence_result_v1 *result) noexcept;

} // namespace cellerator::compute::operation::fusion
