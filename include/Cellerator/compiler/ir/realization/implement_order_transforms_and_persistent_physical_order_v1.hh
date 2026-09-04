#pragma once
#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class order_class_v1 : std::uint8_t { logical=1, canonical, projection_native, persistent_physical };
enum class order_stage_kind_v1 : std::uint8_t { gather=1, scatter, canonicalize };
struct order_identity_v1 { stable_identity_v1 identity{}; order_class_v1 order=order_class_v1::logical; };
struct order_transform_v1 {
    stable_identity_v1 identity{}; order_identity_v1 input{},output{};
    order_stage_kind_v1 kind=order_stage_kind_v1::gather;
    std::vector<std::uint64_t> output_to_input;
};
struct relation_order_stage_v1 { stable_identity_v1 operation{}; order_identity_v1 input{},output{}; };
struct persistent_order_chain_v1 {
    std::vector<relation_order_stage_v1> relations;
    std::vector<order_transform_v1> boundary_transforms;
    std::uint64_t transforms_reused=0;
};
enum class order_status_v1 : std::uint8_t { valid=0, invalid_identity, invalid_permutation, disconnected_chain, redundant_canonicalize };
[[nodiscard]] order_status_v1 validate_order_transform_v1(const order_transform_v1&,std::string*error=nullptr) noexcept;
[[nodiscard]] order_status_v1 validate_persistent_order_chain_v1(const persistent_order_chain_v1&,std::string*error=nullptr) noexcept;
[[nodiscard]] std::vector<double> apply_order_transform_v1(const std::vector<double>&,const order_transform_v1&);
} // namespace cellerator::compiler::ir::realization::v1
