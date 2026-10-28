#pragma once

#include <cstdint>
#include <limits>

namespace cellerator::compiler::sema::v1 {

using logical_edge_id = std::uint64_t;
using physical_slot = std::uint64_t;
inline constexpr logical_edge_id no_logical_edge =
    std::numeric_limits<logical_edge_id>::max();
inline constexpr physical_slot no_physical_slot =
    std::numeric_limits<physical_slot>::max();

struct exact_support_member {
    logical_edge_id edge = no_logical_edge;
    std::uint64_t source_position = 0;
    std::uint64_t destination_position = 0;
};

struct projection_slot_binding {
    physical_slot slot = no_physical_slot;
    logical_edge_id edge = no_logical_edge;
    bool hole = true;
};

struct active_support_member {
    logical_edge_id edge = no_logical_edge;
    std::uint64_t generation = 0;
    bool active = false;
};

bool valid_exact_support(const exact_support_member *members,
                         std::uint64_t count) noexcept;
bool valid_projection_slots(const projection_slot_binding *slots,
                            std::uint64_t count) noexcept;
bool active_support_applies(const active_support_member &member,
                            logical_edge_id edge,
                            std::uint64_t generation) noexcept;

}  // namespace cellerator::compiler::sema::v1
