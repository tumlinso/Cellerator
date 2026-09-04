#pragma once

#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

enum class identity_origin : std::uint8_t {
    inferred = 1,
    declared_persistent,
    user_forced,
    cloned,
    ephemeral
};

struct compiler_identity_handle {
    std::uint32_t slot = 0;
    std::uint32_t generation = 0;
};

struct identity_binding {
    semantic_identity persistent{};
    identity_origin origin = identity_origin::ephemeral;
    compiler_identity_handle handle{};
    std::uint64_t semantic_generation = 0;
    bool unsafe_assertion_warning = false;
};

identity_binding make_identity_binding(semantic_identity identity,
                                       identity_origin origin,
                                       compiler_identity_handle handle,
                                       std::uint64_t generation) noexcept;
bool identity_is_persistable(const identity_binding &binding) noexcept;
bool identity_cache_entry_reusable(const identity_binding &cached,
                                   const identity_binding &requested) noexcept;

}  // namespace cellerator::compiler::sema::v1
