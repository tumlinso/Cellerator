#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

struct semantic_identity_v1 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return low != 0 || high != 0; }
};

struct domain_ir_type_v1 {
    semantic_identity_v1 identity{};
    std::string nominal_tag;
};

struct order_ir_type_v1 {
    semantic_identity_v1 identity{};
    semantic_identity_v1 domain{};
    bool canonical = false;
};

struct geometry_ir_type_v1 {
    semantic_identity_v1 identity{};
    semantic_identity_v1 domain{};
};

struct partition_ir_type_v1 {
    semantic_identity_v1 identity{};
    semantic_identity_v1 domain{};
    semantic_identity_v1 hierarchy{};
};

enum class extent_knowledge_kind_v1 : std::uint8_t {
    unknown = 1,
    bounded,
    exact,
};

struct extent_knowledge_v1 {
    extent_knowledge_kind_v1 kind = extent_knowledge_kind_v1::unknown;
    std::uint64_t lower_bound = 0;
    std::uint64_t upper_bound = 0;
};

enum class axis_identity_space_v1 : std::uint8_t {
    global = 1,
    partition_local,
};

enum class identity_recovery_kind_v1 : std::uint8_t {
    identity = 1,
    affine,
    explicit_map,
};

struct axis_identity_recovery_v1 {
    axis_identity_space_v1 stored_space = axis_identity_space_v1::global;
    identity_recovery_kind_v1 kind = identity_recovery_kind_v1::identity;
    std::uint64_t global_extent = 0;
    std::uint64_t affine_base = 0;
    std::vector<std::uint64_t> local_to_global;
};

struct axis_ir_type_v1 {
    semantic_identity_v1 identity{};
    domain_ir_type_v1 domain{};
    order_ir_type_v1 order{};
    geometry_ir_type_v1 geometry{};
    partition_ir_type_v1 partition{};
    extent_knowledge_v1 extent{};
    axis_identity_recovery_v1 recovery{};
};

enum class axis_ir_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_axis_identity,
    invalid_domain,
    invalid_order,
    invalid_geometry,
    invalid_partition,
    invalid_extent,
    invalid_recovery,
    biological_abi_mismatch,
};

[[nodiscard]] axis_ir_validation_code_v1
validate_domain_ir_type_v1(const domain_ir_type_v1& domain) noexcept;

[[nodiscard]] axis_ir_validation_code_v1
validate_axis_ir_type_v1(const axis_ir_type_v1& axis) noexcept;

[[nodiscard]] axis_ir_validation_code_v1
validate_axis_ir_against_biological_abi_v1(
    const axis_ir_type_v1& axis,
    const cellerator::execution::persistent_axis_identity& abi_axis) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
