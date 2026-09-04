#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

inline constexpr std::uint32_t realization_ir_contract_version_v1 = 1u;

struct stable_identity_v1 {
    std::uint64_t high = 0u;
    std::uint64_t low = 0u;
};

[[nodiscard]] constexpr bool valid(stable_identity_v1 identity) noexcept {
    return identity.high != 0u || identity.low != 0u;
}

[[nodiscard]] constexpr bool operator==(
    stable_identity_v1 lhs, stable_identity_v1 rhs) noexcept {
    return lhs.high == rhs.high && lhs.low == rhs.low;
}

enum class realization_object_kind_v1 : std::uint8_t {
    function = 1u,
    kernel,
    host_stub,
    data_artifact,
    stage,
    binding,
    native_fragment,
};

// A target scope names a compiler target and one concrete profile variant.
// Capability details remain separate and are introduced by the target
// description contract; a target name alone never implies capabilities.
struct target_scope_v1 {
    stable_identity_v1 identity{};
    std::string target_name;
    std::string profile_variant;
};

// Lineage is cold compiler metadata. It is retained for inspection and
// resumption but is not parsed by a prepared-program hot path.
struct realization_lineage_v1 {
    stable_identity_v1 source_identity{};
    stable_identity_v1 semantic_identity{};
    stable_identity_v1 planning_identity{};
};

struct realization_object_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 target_scope{};
    realization_object_kind_v1 kind = realization_object_kind_v1::function;
    std::string name;
    realization_lineage_v1 lineage{};
};

struct realization_module_v1 {
    std::uint32_t contract_version = realization_ir_contract_version_v1;
    stable_identity_v1 identity{};
    std::string name;
    std::vector<target_scope_v1> targets;
    std::vector<realization_object_v1> objects;
};

enum class realization_module_status_v1 : std::uint8_t {
    valid = 0u,
    unsupported_version,
    missing_identity,
    missing_name,
    missing_target,
    duplicate_target,
    duplicate_object,
    unknown_target,
    missing_lineage,
};

[[nodiscard]] realization_module_status_v1 validate_realization_module_v1(
    const realization_module_v1& module,
    std::string* error = nullptr) noexcept;

[[nodiscard]] bool equivalent_realization_module_v1(
    const realization_module_v1& lhs,
    const realization_module_v1& rhs) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
