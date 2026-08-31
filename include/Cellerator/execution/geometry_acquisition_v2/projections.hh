#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::acquisition_v2 {

struct route_resolution {
    route requested = route::compile_now;
    route selected = route::compile_now;
    bool rebuilt_from_embedded_csg1 = false;
    std::uint8_t reserved[5]{};
};

struct projection_record {
    stable_identity candidate{};
    stable_identity physical_projection{};
    std::uint64_t logical_work_items = 0;
    std::uint64_t physical_slots = 0;
    std::uint64_t first_chunk = 0;
    std::uint64_t chunk_count = 0;
    std::uint8_t value_modes = logical_primary_values;
    bool preserves_permanent_holes = true;
    std::uint8_t reserved[6]{};
};

struct projection_chunk {
    std::uint64_t projection_index = 0;
    std::uint64_t chunk_index = 0;
    std::uint64_t logical_begin = 0;
    std::uint32_t logical_count = 0;
    std::uint8_t local_index_bits = 32;
    std::uint8_t reserved0[3]{};
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_bytes = 0;
};

struct projection_set {
    const projection_record *projections = nullptr;
    std::uint64_t projection_count = 0;
    const projection_chunk *chunks = nullptr;
    std::uint64_t chunk_count = 0;
    std::uint64_t payload_bytes = 0;
};

status resolve_route(
    const acquisition_request &request, route_resolution *resolution) noexcept;
status validate_projection_set(
    const acquisition_request &request, const projection_set &set) noexcept;

static_assert(std::is_trivially_copyable_v<route_resolution>);
static_assert(std::is_trivially_copyable_v<projection_record>);
static_assert(std::is_trivially_copyable_v<projection_chunk>);
static_assert(std::is_trivially_copyable_v<projection_set>);

}  // namespace cellerator::execution::acquisition_v2
