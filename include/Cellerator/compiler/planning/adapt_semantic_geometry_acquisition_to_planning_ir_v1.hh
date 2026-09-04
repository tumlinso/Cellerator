#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

inline constexpr std::uint32_t semantic_geometry_acquisition_schema_v1 = 1u;

struct planning_identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return low != 0u || high != 0u;
    }
};

enum class geometry_acquisition_kind_v1 : std::uint8_t {
    compile_now = 1u,
    precompiled_semantic_geometry = 2u,
    external_exact_cover = 3u,
    conventional_fallback = 4u,
};

enum geometry_compatibility_flag_v1 : std::uint32_t {
    compatible_semantics_v1 = 1u << 0u,
    compatible_profile_v1 = 1u << 1u,
    compatible_target_v1 = 1u << 2u,
    exact_logical_coverage_v1 = 1u << 3u,
};

struct geometry_acquisition_cost_v1 {
    std::uint64_t discovery_ns = 0u;
    std::uint64_t construction_ns = 0u;
    std::uint64_t validation_ns = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct geometry_acquisition_request_v1 {
    std::uint32_t schema_version = semantic_geometry_acquisition_schema_v1;
    std::uint32_t record_bytes = sizeof(geometry_acquisition_request_v1);
    geometry_acquisition_kind_v1 kind = geometry_acquisition_kind_v1::compile_now;
    std::array<std::uint8_t, 7> reserved{};
    planning_identity_v1 request_identity{};
    planning_identity_v1 semantic_problem_identity{};
    planning_identity_v1 profile_identity{};
    planning_identity_v1 target_identity{};
    planning_identity_v1 supplied_geometry_identity{};
    std::uint32_t required_compatibility = compatible_semantics_v1;
    std::uint32_t reserved1 = 0u;
    geometry_acquisition_cost_v1 maximum_acquisition_cost{};
};

enum class geometry_acquisition_status_v1 : std::uint8_t {
    acquired = 1u,
    incompatible = 2u,
    cost_limit_exceeded = 3u,
    unavailable = 4u,
};

struct geometry_acquisition_result_v1 {
    std::uint32_t schema_version = semantic_geometry_acquisition_schema_v1;
    std::uint32_t record_bytes = sizeof(geometry_acquisition_result_v1);
    geometry_acquisition_kind_v1 kind = geometry_acquisition_kind_v1::compile_now;
    geometry_acquisition_status_v1 status = geometry_acquisition_status_v1::unavailable;
    std::array<std::uint8_t, 6> reserved{};
    planning_identity_v1 request_identity{};
    planning_identity_v1 semantic_geometry_identity{};
    planning_identity_v1 provider_identity{};
    std::uint32_t satisfied_compatibility = 0u;
    std::uint32_t reserved1 = 0u;
    geometry_acquisition_cost_v1 measured_acquisition_cost{};
};

enum class geometry_acquisition_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    invalid_record_bytes,
    invalid_kind,
    invalid_identity,
    invalid_compatibility,
    exact_cover_not_required,
    supplied_geometry_missing,
    result_mismatch,
    malformed_csg1,
};

[[nodiscard]] geometry_acquisition_validation_code_v1
validate_geometry_acquisition_request_v1(
    const geometry_acquisition_request_v1& request) noexcept;

[[nodiscard]] geometry_acquisition_validation_code_v1
validate_geometry_acquisition_result_v1(
    const geometry_acquisition_request_v1& request,
    const geometry_acquisition_result_v1& result) noexcept;

[[nodiscard]] std::vector<std::byte> encode_csg1_request_v1(
    const geometry_acquisition_request_v1& request);

[[nodiscard]] std::vector<std::byte> encode_csg1_result_v1(
    const geometry_acquisition_result_v1& result);

[[nodiscard]] geometry_acquisition_validation_code_v1 decode_csg1_request_v1(
    const std::byte* data,
    std::size_t bytes,
    geometry_acquisition_request_v1* request) noexcept;

[[nodiscard]] geometry_acquisition_validation_code_v1 decode_csg1_result_v1(
    const std::byte* data,
    std::size_t bytes,
    geometry_acquisition_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::planning
