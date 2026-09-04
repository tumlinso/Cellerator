#include <Cellerator/compiler/sema/implement_support_and_logical_edge_identity_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {

bool valid_exact_support(const exact_support_member *members,
                         std::uint64_t count) noexcept {
    if (count != 0 && members == nullptr)
        return false;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (members[i].edge == no_logical_edge)
            return false;
        for (std::uint64_t j = 0; j < i; ++j) {
            if (members[i].edge == members[j].edge)
                return false;
        }
    }
    return true;
}

bool valid_projection_slots(const projection_slot_binding *slots,
                            std::uint64_t count) noexcept {
    if (count != 0 && slots == nullptr)
        return false;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (slots[i].slot == no_physical_slot)
            return false;
        if (slots[i].hole != (slots[i].edge == no_logical_edge))
            return false;
    }
    return true;
}

bool active_support_applies(const active_support_member &member,
                            logical_edge_id edge,
                            std::uint64_t generation) noexcept {
    return member.edge == edge && member.generation == generation
        && member.active;
}

}  // namespace cellerator::compiler::sema::v1
