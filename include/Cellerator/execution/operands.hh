#pragma once

#include <Cellerator/execution/identity.hh>

#include <type_traits>

namespace cellerator::execution {

enum class numeric_type : u8 {
    invalid = 0u,
    bit = 1u,
    u8 = 2u,
    u16 = 3u,
    u32 = 4u,
    i32 = 5u,
    f16 = 6u,
    bf16 = 7u,
    f32 = 8u,
    f64 = 9u
};

enum class operand_kind : u8 {
    dense_tensor = 1u,
    bit_plane = 2u,
    event_stream = 3u,
    segment_stream = 4u,
    sparse_relation = 5u,
    scalar_or_small_parameter = 6u
};

struct sequence_domain {
    domain_handle genome_domain;
    u32 contig_id;
    u32 chunk_id;
    u64 global_base_begin;
    u32 local_base_count;
    u32 owned_begin;
    u32 owned_end;
    u16 halo_left;
    u16 halo_right;
};

struct dense_tensor_view {
    void *data;
    device_location location;
    numeric_type value_type;
    u8 rank;
    u16 reserved;
    axis_identity axes[biological_operand_max_axes];
    u64 shape[biological_operand_max_axes];
    i64 stride[biological_operand_max_axes];
};

struct bit_plane_view {
    axis_identity coordinate_axis;
    const u32 *low;
    const u32 *high;
    const u32 *validity;
    device_location location;
    u32 word_count;
    u32 base_count;
};

enum class event_ordering : u8 {
    unordered = 1u,
    coordinate_stable = 2u,
    predicate_then_coordinate = 3u
};

struct event_stream_view {
    axis_identity event_axis;
    sequence_domain source_domain;
    const u32 *local_position;
    const u16 *rule_id;
    const u8 *attributes;
    const u8 *strand;
    device_location location;
    u64 total_matches;
    u64 stored_records;
    u64 dropped_records;
    event_ordering ordering;
    u8 reserved[7];
};

struct segment_stream_view {
    axis_identity segment_axis;
    sequence_domain source_domain;
    const u32 *begin;
    const u32 *end;
    const u32 *class_id;
    device_location location;
    u64 segment_count;
};

// Projection data is opaque at this layer. Its typed physical view belongs to
// the projection catalog, while source/destination axes retain semantics.
struct sparse_relation_view {
    axis_identity source_axis;
    axis_identity destination_axis;
    structure_handle structure;
    projection_handle projection;
    structure_epoch epoch;
    const void *projection_data;
    u64 projection_bytes;
    u64 logical_edge_count;
    device_location location;
};

struct scalar_parameter_view {
    const void *data;
    device_location location;
    numeric_type value_type;
    u8 reserved[7];
    u64 element_count;
};

union biological_operand_storage {
    dense_tensor_view dense;
    bit_plane_view bits;
    event_stream_view events;
    segment_stream_view segments;
    sparse_relation_view relation;
    scalar_parameter_view parameter;
};

struct biological_operand_view {
    operand_kind kind;
    u8 reserved[7];
    biological_operand_storage storage;
};

static_assert(sizeof(biological_operand_view) <= 256u,
    "launch operand envelope exceeded the reviewed ABI v1 size budget");
static_assert(std::is_trivially_copyable<biological_operand_view>::value,
    "biological operands must remain trivially copyable");
static_assert(std::is_standard_layout<biological_operand_view>::value,
    "biological operands must remain standard layout");
static_assert(std::is_trivially_copyable<sequence_domain>::value,
    "sequence domain must remain device-copyable");

} // namespace cellerator::execution
