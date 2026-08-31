#pragma once

#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

inline constexpr std::uint32_t segment_mechanism_schema_version_v2 = 2u;
inline constexpr std::uint32_t segment_mechanism_name_capacity_v2 = 48u;

struct segment_mechanism_descriptor_v2 {
    std::uint32_t schema_version = segment_mechanism_schema_version_v2;
    segment_mechanism_v2 mechanism = segment_mechanism_v2::cta_per_output;
    std::uint8_t reserved0[3]{};
    std::uint64_t mechanism_identity = 0u;
    char static_name[segment_mechanism_name_capacity_v2]{};
    std::uint32_t threads_per_cta = 0u;
    std::uint32_t warps_per_cta = 0u;
    std::uint64_t preferred_maximum_segment_length = 0u;
    std::uint32_t launches_per_component = 1u;
    std::uint32_t dynamic_shared_bytes = 0u;
    bool graph_capture_compatible = true;
    bool requires_measurement = true;
    bool production_promoted = false;
    std::uint8_t reserved1[5]{};
};

struct segment_mechanism_portfolio_view_v2 {
    const segment_mechanism_descriptor_v2 *mechanisms = nullptr;
    std::uint32_t count = 0u;
    std::uint32_t reserved = 0u;
};

segment_mechanism_portfolio_view_v2 built_in_segment_mechanisms_v2() noexcept;

segment_result_v2 validate_segment_mechanism_portfolio_v2(
    segment_mechanism_portfolio_view_v2 portfolio) noexcept;

const segment_mechanism_descriptor_v2 *find_segment_mechanism_v2(
    segment_mechanism_portfolio_view_v2 portfolio,
    segment_mechanism_v2 mechanism) noexcept;

static_assert(std::is_trivially_copyable<segment_mechanism_descriptor_v2>::value,
    "segment mechanism descriptor must remain pointer-free");

} // namespace cellerator::compute::segment
