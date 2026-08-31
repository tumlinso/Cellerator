#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution::acquisition_v2 {

inline constexpr std::uint32_t schema_version = 2u;
inline constexpr std::uint32_t maximum_projection_requirements = 64u;

struct stable_identity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

enum class route : std::uint8_t {
    compile_now = 1,
    load_csg1 = 2,
    load_cpe2 = 3,
    adapt_cpk1 = 4
};

enum class cpe2_disposition : std::uint8_t {
    not_applicable = 0,
    compatible = 1,
    incompatible = 2,
    invalid = 3
};

enum class fallback_policy : std::uint8_t {
    reject = 0,
    rebuild_from_embedded_csg1 = 1
};

enum value_mode_flag : std::uint8_t {
    logical_primary_values = 1u << 0u,
    projection_primary_values = 1u << 1u
};

struct byte_span {
    void *data = nullptr;
    std::uint64_t bytes = 0;
};

struct immutable_byte_span {
    const void *data = nullptr;
    std::uint64_t bytes = 0;
};

struct projection_requirement {
    stable_identity candidate{};
    std::uint32_t projection_kind = 0;
    std::uint32_t provider_kind = 0;
    std::uint64_t logical_work_items = 0;
    std::uint8_t accepted_value_modes = logical_primary_values;
    std::uint8_t reserved[7]{};
};

struct acquisition_request {
    std::uint32_t version = schema_version;
    std::uint32_t record_bytes = sizeof(acquisition_request);
    route preferred_route = route::compile_now;
    cpe2_disposition cpe2 = cpe2_disposition::not_applicable;
    fallback_policy fallback = fallback_policy::reject;
    std::uint8_t required_value_modes = logical_primary_values;
    structure_id structure{};
    structure_epoch epoch{};
    immutable_byte_span source{};
    const projection_requirement *projection_requirements = nullptr;
    std::uint64_t projection_requirement_count = 0;
};

struct buffer_requirement {
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 1;
};

struct acquisition_requirements {
    std::uint32_t version = schema_version;
    std::uint32_t record_bytes = sizeof(acquisition_requirements);
    route selected_route = route::compile_now;
    bool rebuilt_from_embedded_csg1 = false;
    std::uint8_t reserved0[6]{};
    std::uint64_t projection_count = 0;
    std::uint64_t projection_chunk_count = 0;
    buffer_requirement semantic_geometry{};
    buffer_requirement projections{};
    buffer_requirement catalog{};
    buffer_requirement planner{};
    buffer_requirement program{};
    buffer_requirement transient{};
    buffer_requirement diagnostics{};
};

struct acquisition_buffers {
    byte_span semantic_geometry{};
    byte_span projections{};
    byte_span catalog{};
    byte_span planner{};
    byte_span program{};
    byte_span transient{};
    byte_span diagnostics{};
};

struct acquired_geometry {
    std::uint32_t version = schema_version;
    std::uint32_t record_bytes = sizeof(acquired_geometry);
    stable_identity semantic_geometry{};
    immutable_byte_span semantic_geometry_image{};
    immutable_byte_span projection_records{};
    std::uint64_t projection_count = 0;
    immutable_byte_span prepared_program{};
    immutable_byte_span diagnostics{};
};

enum class status_code : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_header,
    invalid_identity,
    invalid_route,
    incompatible_cpe2_rejected,
    invalid_source,
    invalid_requirements,
    insufficient_capacity,
    callback_unavailable,
    callback_failed,
    invalid_result
};

struct status {
    status_code code = status_code::success;
    std::uint64_t index = 0;

    constexpr explicit operator bool() const noexcept {
        return code == status_code::success;
    }
};

using requirements_query_function = status (*)(
    const acquisition_request &, acquisition_requirements *) noexcept;
using acquisition_function = status (*)(const acquisition_request &,
    const acquisition_requirements &, const acquisition_buffers &,
    acquired_geometry *) noexcept;

struct acquisition_facade {
    requirements_query_function query = nullptr;
    acquisition_function acquire = nullptr;
};

constexpr bool valid_stable_identity(stable_identity identity) noexcept {
    return identity.low != 0 || identity.high != 0;
}

status validate_request(const acquisition_request &request) noexcept;
status validate_requirements(
    const acquisition_request &request,
    const acquisition_requirements &requirements) noexcept;
status query_requirements(const acquisition_facade &facade,
    const acquisition_request &request,
    acquisition_requirements *requirements) noexcept;
status acquire(const acquisition_facade &facade,
    const acquisition_request &request,
    const acquisition_requirements &requirements,
    const acquisition_buffers &buffers,
    acquired_geometry *result) noexcept;

static_assert(std::is_trivially_copyable_v<projection_requirement>);
static_assert(std::is_trivially_copyable_v<acquisition_request>);
static_assert(std::is_trivially_copyable_v<acquisition_requirements>);
static_assert(std::is_trivially_copyable_v<acquisition_buffers>);
static_assert(std::is_trivially_copyable_v<acquired_geometry>);
static_assert(std::is_trivially_copyable_v<acquisition_facade>);

}  // namespace cellerator::execution::acquisition_v2
