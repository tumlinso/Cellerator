#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::acquisition_v2 {

struct compiled_provider {
    stable_identity identity{};
    std::uint32_t provider_kind = 0;
    std::uint16_t architecture_major = 0;
    std::uint16_t architecture_minor = 0;
    bool primary = false;
    bool experimental = false;
    std::uint8_t reserved[6]{};
};

struct provider_registry {
    stable_identity identity{};
    const compiled_provider *providers = nullptr;
    std::uint64_t provider_count = 0;
};

struct catalog_candidate {
    stable_identity identity{};
    std::uint32_t provider_kind = 0;
    std::uint32_t projection_kind = 0;
    bool experimental = false;
    bool requires_measurement = false;
    std::uint8_t reserved[6]{};
};

struct candidate_catalog {
    stable_identity identity{};
    const catalog_candidate *candidates = nullptr;
    std::uint64_t candidate_count = 0;
};

struct planner_binding {
    stable_identity identity{};
    acquisition_facade facade{};
};

struct default_assembly {
    provider_registry registry{};
    candidate_catalog catalog{};
    planner_binding planner{};
    bool include_experimental = false;
    std::uint8_t reserved[7]{};
};

status validate_default_assembly(const default_assembly &assembly) noexcept;
status query_default_assembly(const default_assembly &assembly,
    const acquisition_request &request,
    acquisition_requirements *requirements) noexcept;
status acquire_default_assembly(const default_assembly &assembly,
    const acquisition_request &request,
    const acquisition_requirements &requirements,
    const acquisition_buffers &buffers,
    acquired_geometry *result) noexcept;

static_assert(std::is_trivially_copyable_v<compiled_provider>);
static_assert(std::is_trivially_copyable_v<provider_registry>);
static_assert(std::is_trivially_copyable_v<catalog_candidate>);
static_assert(std::is_trivially_copyable_v<candidate_catalog>);
static_assert(std::is_trivially_copyable_v<planner_binding>);
static_assert(std::is_trivially_copyable_v<default_assembly>);

}  // namespace cellerator::execution::acquisition_v2
