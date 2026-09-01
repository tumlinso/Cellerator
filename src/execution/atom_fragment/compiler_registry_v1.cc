#include <Cellerator/execution/atom_fragment/compiler_registry_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

using identity = joint_compiler::persistent_identity_v1;

bool valid(identity value) noexcept {
    return static_cast<bool>(
        joint_compiler::validate_persistent_identity_v1(value));
}

bool less(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool same(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

bool entry_less(const fragment_compiler_entry_v1 &lhs,
    const fragment_compiler_entry_v1 &rhs) noexcept {
    return less(lhs.source_identity, rhs.source_identity)
        || (same(lhs.source_identity, rhs.source_identity)
            && lhs.candidate_id < rhs.candidate_id);
}

} // namespace

fragment_compiler_registry_status_v1 validate_fragment_compiler_registry_v1(
    const fragment_compiler_registry_v1 &registry) noexcept {
    using code = fragment_compiler_registry_status_code_v1;
    if (registry.entry_count == 0u || registry.entries == nullptr)
        return {code::missing_entries, 0u};
    for (std::uint64_t index = 0u; index < registry.entry_count; ++index) {
        const auto &entry = registry.entries[index];
        if (!valid(entry.source_identity))
            return {code::invalid_source_identity, index};
        if (entry.candidate_id == 0u)
            return {code::invalid_candidate, index};
        if (!valid(entry.compiler_identity))
            return {code::invalid_compiler_identity, index};
        if (entry.source_context == nullptr)
            return {code::missing_source_context, index};
        if (entry.compile == nullptr)
            return {code::missing_compile_function, index};
        if (index != 0u && !entry_less(registry.entries[index - 1u], entry))
            return {code::duplicate_or_unordered_entry, index};
    }
    return {};
}

const fragment_compiler_entry_v1 *find_fragment_compiler_v1(
    const fragment_compiler_registry_v1 &registry,
    identity source_identity, std::uint64_t candidate_id) noexcept {
    if (!validate_fragment_compiler_registry_v1(registry)
        || !valid(source_identity) || candidate_id == 0u)
        return nullptr;
    fragment_compiler_entry_v1 key{};
    key.source_identity = source_identity;
    key.candidate_id = candidate_id;
    std::uint64_t first = 0u;
    std::uint64_t count = registry.entry_count;
    while (first < count) {
        const std::uint64_t middle = first + (count - first) / 2u;
        const auto &entry = registry.entries[middle];
        if (same(entry.source_identity, source_identity)
            && entry.candidate_id == candidate_id)
            return &entry;
        if (entry_less(entry, key))
            first = middle + 1u;
        else
            count = middle;
    }
    return nullptr;
}

} // namespace cellerator::execution::atom_fragment
